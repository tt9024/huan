import numpy as np
import mts_util
import dill
import ar1_md
import ar1_eval
import datetime
import yaml
import os
import copy

def get_sw(strat_name, sw_file = '/home/mts/upload/STRATEGY_WEIGHTS.yaml', str_code = 'TSC'):
    with open(sw_file, 'r') as f:
        sw = yaml.safe_load(f)
        return float(sw[strat_name+'_'+str_code])

def read_ar1_config(fname_cfg, symbol) :
    """
    this is replaced by ar1_sim_ens.cfg_2_mkw(cfg_file, mkt='WTI_N1')
    difference is the new param has fscale/k0_k1_list to be [], 
    added 'persist_path' and 'name' that generated from model_id
    """
    param = {}
    cfg = mts_util.MTS_Config(fname_cfg)
    param['symbol'] = symbol
    param['strategy_key'] = cfg.get('strategy_key')
    param['sw_tsc'] = get_sw(param['strategy_key'], str_code='TSC')
    param['sw_tsd'] = get_sw(param['strategy_key'], str_code='TSD')
    param['trd_day'] = cfg.get('trade_date')
    param['strat_code'] = cfg.get('strategy_code')
    mts_sym = ar1_md.mts_repo_mapping[symbol]
    cstr = 'live_output.'+mts_sym+'.'
    param['name'] = cfg.get(cstr+'.name',type_func=str)
    param['maxpos'] = cfg.get(cstr+'.max_position',type_func=int)
    param['min_trade'] = cfg.get(cstr+'.min_trade', type_func=int)
    param['trading_hour'] = cfg.getArr(cstr+'.trading_hour', type_func=float)
    param['persist_path'] = cfg.get(cstr+'.persist_path')
    cstr += 'parameters.'
    pkeys = [('n', int),('barsec', int),('trigger_cnt',int),('tcost',float),('wpos', int)]
    pkeys_arr = [('fscale',float),('vscale',float), ('ixf',int)]
    for (key,tp) in pkeys :
        param[key] = cfg.get(cstr+key, type_func=tp)
    for (key,tp) in pkeys_arr:
        param[key] = cfg.getArr(cstr+key, type_func=tp)
    param['tcost_st'] = None  # used for checking st direction

    exec_key = cstr+'execution'
    exec_len = cfg.arrayLen(exec_key)
    param['k0_k1_list'] = []
    ekeys = [('k0',int), ('k1', int), ('tscale', float), ('wpos',int)]
    ekeys_arr = [('vscale',float)]
    for i in np.arange(int(exec_len)) :
        ed = {}
        for (key,tp) in ekeys:
            k = exec_key + '[%i].%s'%(i,key)
            ed[key] = cfg.get(k,type_func=tp)
        for (key,tp) in ekeys_arr:
            k = exec_key + '[%i].%s'%(i,key)
            ed[key] = cfg.getArr(k,type_func=tp)

        k0k1=[(ed['k0'],ed['k1']),ed['vscale'],ed['tscale'],ed['wpos']]
        param['k0_k1_list'].append(k0k1)
    return param

def gen_exec_param(param, stdv, lpx_day=None):
    """
    Generate:
         'lpx_day': length > k daily lpx, or so far, including the k
         'v_day': shape[n,nf] v from previous day's stdv, scaled with k0_k1
         'vs_day': shape[n,3] vscale each with (vs,maxv,expv), note vs shouldn't be used 
         't_day': shape[n,nf+1] trading cost scale day
         'w_day': shape[n,nf] weight for pop
    Need from param:
         'tcost': in price for unscaled trading cost
         'maxpos': maximum position
         'min_trade': minimum trade size
         'vscale': (vs,maxv,expv)
         'wpos': integer or None
         'k0_k1_list': list of [(k0,k1),(vs,maxv,expv),tcost,wpos]
    """
    n = len(stdv)
    assert param['n']==n
    stdvh = stdv.reshape((1,n))
    ixf = param['ixf']
    fscale = param['fscale']
    if len(fscale)==0:
        fscale=np.ones(len(ixf))
    vscale = param['vscale']
    wpos = param['wpos']
    tcost = param['tcost']
    k0k1 = param['k0_k1_list']
    v_day,vs_day,t_day,w_day = ar1_eval.gen_daily_vvtw(ixf,stdvh,vscale,wpos,k0k1)

    ep = {'n':n}
    if lpx_day is None:
        lpx_day = np.ones(n)*1e-8 # to be inited by snap
    ep['lpx_day']=lpx_day.copy()
    ep['fscale']=fscale
    ep['v_day']=v_day
    ep['vs_day']=vs_day
    ep['t_day']=t_day
    ep['w_day']=w_day
    ep['min_trade'] = param['min_trade']
    ep['tcost'] = param['tcost']
    ep['maxpos'] = param['maxpos']
    return ep

def gen_trigger_time(stdv, barsec, cnt, min_diff=15) :
    n = len(stdv)
    N = n*barsec
    cnt0 = int(N/min_diff/2)  # pick a block of every other one
    pdf = np.tile(stdv**2,(barsec,1)).T.flatten()
    pdf = pdf[(np.linspace(1,N-1,cnt0)+0.5).astype(int)]
    pdf/=np.sum(pdf)
    k = np.random.choice(np.arange(cnt0).astype(int), size=cnt, replace=False, p=pdf)
    x = (k*(min_diff*2)+np.random.rand(cnt)*min_diff).astype(int)
    zix = np.nonzero(x%barsec==0)[0]
    x=np.delete(x,zix)
    x.sort()
    return x

def get_pos_param(f, p0, exec_param_daily, use_k=None, logfunc=ar1_eval.NullLogger) :
    """run f,v for one day, using exec_param_daily.
       
       p0, previous position
       use_k, if not None, in live, suggests f shape [1,nf] at bar k
              if None, then k_arr will be np.arange(n), assuming one-day worth of f

       exec_param_daily: all parameters from param, plus:
                         'lpx_day': length > k daily lpx, or so far, including the k
                         'v_day': shape[n,nf] v from previous day's stdv, scaled with k0_k1
                         'vs_day': shape[n,3] vscale each with (vs,maxv,expv) at each bar k
                         't_day': shape[n,nf+1] trading cost scale day
                         'w_day': shape[n,nf] weight for pop
       param key use:
                         'fscale': forecast
                         'tcost': in price for unscaled trading cost
                         'maxpos': maximum position
                         'min_trade': minimum trade size
       return: vector of (nndays+1) target positions pp, 
               pp[0] is p0, pp[-1] is the latest target position for f[-1,:]
    """
    param=exec_param_daily
    n=param['n']
    lpx=param['lpx_day']
    vd=param['v_day']
    vsd=param['vs_day']
    td=param['t_day']
    wd=param['w_day']
    if use_k is not None:
        # in live
        assert f.shape[0] == 1
        k=use_k
        v=vd[k:k+1,:]
        lpx=lpx[k:k+1]
        k_arr=np.array([k]).astype(int)
    else:
        # sim one day
        assert f.shape[0] == n
        v=vd
        k_arr=np.arange(n).astype(int)

    fs=f*param['fscale']
    return ar1_eval.p0_from_fnx2_run(n,fs,v,lpx,k_arr,\
            param['tcost'], \
            param['maxpos'], \
            vsd, td, wd,\
            min_trade=param['min_trade'],\
            p0=p0,
            logfunc=logfunc)

def get_p0(k,f,exec_param,prev_pos,logfunc=ar1_eval.NullLogger) :
    pp0 = get_pos_param(f.reshape((1,nf)), prev_pos, exec_param, use_k = k, logfunc=logfunc)
    return pp0[-1]

class AR1_Model :
    def __init__(self, logger, cfg_fname, symbol, fm=None, dump_fname=None):
        self.logger = logger
        self.cfg_fname = cfg_fname
        self.symbol = symbol
        self.fm = fm
        self.dump_fd = None
        self.dump_fname =dump_fname
        self.prev_pos=0

    def sod(self, trade_day=None, check_trade_day=True, param_in=None) :
        if param_in is not None:
            self.param = copy.deepcopy(param_in)
        else:
            self.param = read_ar1_config(self.cfg_fname, self.symbol)
        self.logger.logInfo('read AR1 parameter:\n%s'%(str(self.param)))
        self.tdu = mts_util.TradingDayUtil()
        if trade_day is None:
            trade_day = self.tdu.get_trading_day_dt(datetime.datetime.now(), snap_forward=True)
        if check_trade_day :
            assert trade_day == self.param['trd_day'], 'trade day mismatch!'

        self.trd_day = trade_day
        self._load(trade_day)
        self.fm._init()

        try :
            self.prev_pos = self.fm.prev_pos
        except :
            self.fm.prev_pos = 0
            self.prev_pos = 0
        if np.isnan(self.prev_pos) or np.isinf(self.prev_pos):
            self.logger.logError('AR1 sod got invalid previous position, set to 0')
            self.prev_pos = 0
        self.logger.logInfo('AR1 sod got previous position %d'%(self.prev_pos))

        self.n = int(self.param['n'])
        self.nf = len(self.param['ixf'])

        self.md = {} # key with bar_utc: 'bar': md_bar, 'snap': [ {'bar': md_bar, 'fv': [nf,3,2], 'p0': p0
        self.fv = np.zeros((self.n,3,self.nf,2))
        self.p0 = np.zeros((self.n))

        # init the exec_params
        self._init_exec_param()

        # init fh/vh
        self.fh=[]
        self.vh=[]

    def _init_exec_param(self):
        stdv=self.fm.state_obj.std.v[0,:]
        self.exec_param = gen_exec_param(self.param,stdv)

    @staticmethod
    def _fm_fname(trade_day, param):
        return os.path.join(param['persist_path'], 'TSC-7000-%s/%s_%s'%(param['strat_code'], param['name'], trade_day))

    def _dump_fname(self, trade_day, param):
        if self.dump_fname is None :
            self.dump_fname = os.path.join(self.param['persist_path'], 'TSC-7000-%s/dump_%s_%s'%(self.param['strat_code'], self.param['name'], trade_day))
        return self.dump_fname

    def _load(self, trd_day):
        if self.fm is None:
            with open(AR1_Model._fm_fname(trd_day, self.param), 'rb') as fd:
                self.fm = dill.load(fd)
        #self.dump_fd = open(self._dump_fname(trd_day), 'at')
        self.dump_fd = None

    def _eod_dump(self) :
        trd_day=self.trd_day
        tdi = mts_util.TradingDayIterator(trd_day)
        tdi.begin()
        trd_day = tdi.next()
        with open(AR1_Model._fm_fname(trd_day,self.param), 'wb') as fd:
            dill.dump(self.fm,fd)
        if self.dump_fd is not None:
            self.dump_fd.close()

    @staticmethod
    def _to_array(k, a, p0, md_dict, f):
        curutc = int(datetime.datetime.now().strftime('%s'))
        d=np.array([curutc, k, a, p0])
        d = np.r_[d, ar1_md.md_dict_to_array(md_dict), f]
        return d

    @staticmethod
    def _from_array(d, nf=12) :
        curutc, k, a, p0 = d[:4]
        md_dict = ar1_md.md_dict_from_array(d[4:-nf])
        f=d[-nf:]
        return int(curutc),int(k),a,p0,md_dict,f

    @staticmethod
    def _from_file(dump_fname, nf=12) :
        da = np.genfromtxt(dump_fname, delimiter=',')
        utc = da[:,0].astype(int)
        k = da[:,1].astype(int)
        a = da[:,2]
        p0 = da[:,3].astype(int)
        md_dict = ar1_md.md_dict_from_array_multiday(da[:,4:-nf])
        f=da[:,-nf:]
        return utc, k, a, p0, md_dict, f

    def _dump_log(self, k, a, p0, md_dict, f):
        d=AR1_Model._to_array(k,a,p0,md_dict,f)
        d=d.reshape((1,len(d)))
        np.savetxt(self.dump_fd, d, delimiter=',')
        self.dump_fd.flush()

    def eod(self) :
        self.fm.on_eod()
        self._eod_dump()

    def on_snap(self, k, a, md_dict):
        # update the exec_param
        self.exec_param['lpx_day'][k]=ar1_md.col_from_md_dict(md_dict, self.symbol, 'lpx')
        f,v=self.fm.on_snap(k,a,md_dict)
        self.logger.logInfo('** %s ** on snap (%i,%f)'%(self.param['name'],k,a))
        pp0 = get_pos_param(f.reshape((1,self.nf)), self.fm.prev_pos, self.exec_param, use_k = k, logfunc=self.logger.logInfo)
        p0=pp0[-1]
        if np.isnan(p0) or np.isinf(p0):
            self.logger.logError('got invalid position, set position to 0. Check market data!')
            p0 = 0
        self.fm.prev_pos=p0
        self.logger.logInfo('** %s ** on snap DONE:(%i,%f): pos:(%f)'%(self.param['name'],k,a,p0))
        #self._dump_log(k,a,p0,md_dict,f)

        if a==1:
            self.fh.append(f)
            self.vh.append(v)
        return p0

    def on_bar(self, k, md_dict):
        p0 = self.on_snap(k,1,md_dict)
        self.fm.on_bar(k, md_dict)
        return p0

    def reset_prev_pos(self):
        """ used for setting trading hours, so the prev_pos starts with 0
        """
        self.fm.prev_pos = 0
