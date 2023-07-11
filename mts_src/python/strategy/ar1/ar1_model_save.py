import numpy as np
import mts_util
import dill
import Pop
import ar1_md
import datetime
import copy

def read_ar1_config(fname_cfg) :
    param = {}
    cfg = mts_util.MTS_Config(fname_cfg)
    param['trd_day'] = cfg.get('trade_date')
    param['strat_code'] = cfg.get('strategy_code')
    cstr = 'live_output.WTI_N1.'
    param['mpos'] = cfg.get(cstr+'.max_position')
    param['min_trade'] = cfg.get(cstr+'.min_trade')
    param['trading_hour'] = cfg.getArr(cstr+'.trading_hour', type_func=int)
    cstr += 'parameters.'
    pkeys = [('n', int),('barsec', int),('trigger_cnt',int),('tscale',float),('mpscale',float),('w_pos', int),('clp_ratio',float)]
    pkeys_arr = [('ixf',int),('ixf_st',int),('pick',int)]
    for (key,tp) in pkeys :
        param[key] = cfg.get(cstr+key, type_func=tp)
    for (key,tp) in pkeys_arr:
        param[key] = cfg.getArr(cstr+key, type_func=tp)
    exec_key = cstr+'execution'
    exec_len = cfg.arrayLen(exec_key)
    param['execution'] = []
    ekeys = [('k0',int), ('k1', int), ('fscale_st', float), ('tscale_st',float)]
    for i in np.arange(int(exec_len)) :
        exec_dict = {}
        for (key,tp) in ekeys:
            k = exec_key + '[%i].%s'%(i,key)
            exec_dict[key] = cfg.get(k,type_func=tp)
        param['execution'].append(exec_dict)

    return param

def gen_trigger_time(stdv, barsec, cnt, min_diff=15) :
    n = len(stdv)
    N = n*barsec
    cnt0 = int(N/min_diff/2)  # pick a block of every other one
    pdf = np.tile(stdv**2,(barsec,1)).flatten()
    pdf = pdf[(np.linspace(1,N-1,cnt0)+0.5).astype(int)]
    pdf/=np.sum(pdf)
    k = np.random.choice(np.arange(cnt0).astype(int), size=cnt, replace=False, p=pdf)
    x = (k*(min_diff*2)+np.random.rand(cnt)*min_diff).astype(int)
    zix = np.nonzero(x%barsec==0)[0]
    x=np.delete(x,zix)
    x.sort()
    return x

def get_model_pos_param(pos_in_arr, exec_param) :
    mpscale = exec_param['mpscale']
    clp_ratio = exec_param['clp_ratio']
    maxpos = exec_param['mpos']
    mintrd = exec_param['min_trade']
    return _get_model_pos(pos_in_arr, mpscale, clp_ratio, maxpos, mintrd)

def _get_model_pos(pos_in_arr, mpscale, clp_ratio, maxpos, mintrd) :
    pos_out_arr = []
    mp = 50.0*mpscale
    clp = 50*clp_ratio*mpscale
    for pos in pos_in_arr:
        pos_01 = np.clip(np.tanh(pos/mp)*mp,-clp,clp)/clp
        pos_mdl = float(pos_01*maxpos)/float(mintrd)
        print(pos_mdl)
        pos_mdl_sn = np.sign(pos_mdl)
        pos_mdl = int(pos_mdl_sn* int(pos_mdl*pos_mdl_sn + 0.5)*mintrd)
        pos_out_arr.append(pos_mdl)

    return np.array(pos_out_arr)

def get_pos(f, v, p0, tscale=1.8, mpscale=0.1, n=276, w_pos = 5, f_st=None, v_st=None, k0_k1_st=[(0,60,1,1), (144,160,1,1), (174,209,1.0,0.9),(270,276,1,1)] ,clp_ratio=0.7, use_k = None, verbose = True):
    nf = 12
    w = np.r_[np.linspace(0.5,2,w_pos), np.linspace(2, 0.1, nf-w_pos)]
    w/=np.sum(w)
    t0 = np.ones(nf+1)*0.003992*tscale

    pp0 = [p0]
    for i, (f0, v0) in enumerate(zip(f,v)):
        pp = Pop.opt(pp0[-1],f0,v0, t0, w)
        #print ('calling Pop with  ', pp0[-1],f0,v0, t0, w, ' got ', pp)
        if f_st is not None :
            if use_k is None :
                k = i%n
            else :
                k = use_k
            for (k0, k1, fscale_st, tscale_st) in k0_k1_st:
                if k in np.arange(k0,k1).astype(int):
                    #pp_st = Pop.opt(pp0[-1],f_st[i,:],v_st[i,:], t0*tscale_st, w)
                    w0 = np.ones(12)/12.0
                    pp_st = Pop.opt(pp0[-1],f_st[i,:],v_st[i,:]*tscale_st, t0, w0)
                    #print ('calling Pop_st with  ', pp0[-1],f_st[i,:],v_st[i,:], t0*tscale_st, w, ' got ', pp_st)
                    pp[0] = pp_st[0]*fscale_st + pp[0]*(1-fscale_st)
                    break
        
        mp = 50.0*mpscale
        clp = 50*clp_ratio*mpscale

        #print ('calling final position: ', mp, clp, pp)
        pp0.append(np.clip(np.tanh(pp[0]/mp)*mp,-clp,clp)/clp)

    #print ('return ', pp0)
    return np.array(pp0)

def get_pos1(f, v, p0, tscale=1.8, mpscale=0.1, n=276, w_pos = 5, f_st=None, v_st=None, k0_k1_st=[(0,60,1,1), (144,160,1,1), (174,209,1.0,0.9),(270,276,1,1)] ,clp_ratio=0.7, use_k = None, verbose = True):
    nf = 12
    w = np.r_[np.linspace(0.5,2,w_pos), np.linspace(2, 0.1, nf-w_pos)]
    w/=np.sum(w)
    t0 = np.ones(nf+1)*0.003992*tscale

    pp0 = [p0]
    for i, (f0, v0) in enumerate(zip(f,v)):
        pp = Pop.opt(pp0[-1],f0,v0, t0, w)
        #print ('calling Pop with  ', pp0[-1],f0,v0, t0, w, ' got ', pp)
        if f_st is not None :
            if use_k is None :
                k = i%n
            else :
                k = use_k
            for (k0, k1, fscale_st, tscale_st) in k0_k1_st:
                if k in np.arange(k0,k1).astype(int):
                    #pp_st = Pop.opt(pp0[-1],f_st[i,:],v_st[i,:], t0*tscale_st, w)
                    w = np.ones(12)/12.0
                    pp_st = Pop.opt(pp0[-1],f_st[i,:],v_st[i,:]*tscale_st, t0, w)
                    #print ('calling Pop_st with  ', pp0[-1],f_st[i,:],v_st[i,:], t0*tscale_st, w, ' got ', pp_st)
                    pp[0] = pp_st[0]*fscale_st + pp[0]*(1-fscale_st)
                    break
        
        pp0.append(pp[0])

    #print ('return ', pp0)
    return np.array(pp0)

def get_pos2(f, v, p0, tscale=1.8, mpscale=0.1, n=276, w_pos = 5, f_st=None, v_st=None, k0_k1_st=[(0,60,1,1), (144,160,1,2), (174,209,1.0,2),(270,276,1,2)] ,clp_ratio=0.7, use_k = None, verbose = True):
    nf = 12
    w = np.r_[np.linspace(0.5,2,w_pos), np.linspace(2, 0.1, nf-w_pos)]
    w/=np.sum(w)
    t0 = np.ones(nf+1)*0.003992*tscale

    pp0 = [p0]
    for i, (f0, v0) in enumerate(zip(f,v)):
        pp = Pop.opt(pp0[-1],f0,v0, t0, w)
        print ('calling Pop with  ', pp0[-1],f0,v0, t0, w, ' got ', pp)
        if f_st is not None :
            if use_k is None :
                k = i%n
            else :
                k = use_k
            for (k0, k1, fscale_lt, tscale_lt) in k0_k1_st:
                if k in np.arange(k0,k1).astype(int):
                    pp_lt = Pop.opt(pp0[-1],f0,v0, t0*tscale_lt, w)
                    pp[0] = pp_lt[0]*fscale_lt + pp[0]*(1-fscale_lt)
                    break
        
        mp = 50.0*mpscale
        clp = 50*clp_ratio*mpscale

        print ('calling final position: ', mp, clp, pp)
        pp0.append(np.clip(np.tanh(pp[0]/mp)*mp,-clp,clp)/clp)

    print ('return ', pp0)
    return np.array(pp0)

def get_pos_param(f, v, p0, param, f_st=None, v_st=None, use_k = None) :
    """
    f,v, shape [n,nf]
    p0, scale value
    f_st, v_st: same shape with f,v
    use_k: if None, use index into n, otherwise, given as k, as in live mode
    """
    tscale=  param['tscale']
    mpscale= param['mpscale']
    n= param['n']
    w_pos = param['w_pos']
    k0_k1_st = param['execution']
    clp_ratio = param['clp_ratio']

    k0_k1 = []
    for k01 in k0_k1_st:
        k0_k1.append( (k01['k0'], k01['k1'], k01['fscale_st'], k01['tscale_st']) )

    #return get_pos2(f, v, p0, \
    return get_pos(f, v, p0, \
            tscale=tscale, mpscale=mpscale, n=n, w_pos=w_pos, \
            f_st=f_st, v_st=v_st, \
            k0_k1_st=k0_k1, clp_ratio=clp_ratio, use_k = use_k)

def p0_from_stst(k, new_pos, prev_pos, prev_sz, exec_param, f_st, f_ss, ss_th_enter_flips=3, ss_th_cover_flips=0, ss0_th=5e-6, cover_k0=60, cover_k1=270):
    """
    apply the rule for the execution.
    new_pos, prev_pos and prev_sz in the scale of maxpos
    f_st and f_ss, vector of nf

    returns the preferred position.
    """
    p0_new = new_pos
    if np.abs(p0_new - prev_pos) > 0.5 :
        # DAMP to avoid oscilation
        cur_sz = p0_new - prev_pos
        if prev_sz != 0 and prev_sz*cur_sz < 0 and np.abs(cur_sz) <= np.abs(prev_sz) :
            print ('DAMP - prev_sz ', prev_sz, ' cur_sz ', cur_sz, 'not entering!')
            p0_new = prev_pos
        else :
            print ('check consistency (',k,') f_st:', f_st, ', f_ss: ', f_ss)
            ss = np.sum(np.sign(f_st))

            # cover throttle
            ss0_th = 5e-6
            ss0 = np.sum(np.sign(f_ss-np.sign(f_ss)*ss0_th))
            nf = len(f_ss)
            ss_th_enter = nf-ss_th_enter_flips*2
            ss_th_cover = nf-ss_th_cover_flips*2

            if abs(p0_new) > abs(prev_pos):
                # entering position
                if (p0_new > 0 and ss <= -ss_th_enter) or (p0_new < 0 and ss >= ss_th_enter) :
                    print ('short term ', str(f_st), ' ss', ss, ' not entering ', p0_new, ' from ', prev_pos)
                    p0_new = prev_pos
            else :
                # covering (allow covering at eod/sod)
                if k > cover_k0 and k < cover_k1 :
                    if (prev_pos > 0 and ss0 >= ss_th_cover) or (prev_pos < 0 and ss0 <= -ss_th_cover) :
                        print ('short term ', str(f_ss), ' ss', ss0, ' not covering ', p0_new, ' from ', prev_pos)
                        p0_new = prev_pos
    return p0_new


def get_p0(k,f,v, pop2_obj, exec_param) :
    f_lt = []
    v_lt = []
    pk_st, pk_lt = exec_param['pick']

    pop2_obj.on_bar(k,f,v)
    for pk in [pk_st, pk_lt]:
        pop2_obj.pick = -pk
        f0, v0 = pop2_obj.gen_nf(exec_param['ixf'])
        f_lt.append(f0)
        v_lt.append(v0)
    
    flt = f.copy()
    flt[:3] = f_lt[1][:3]*4
    flt[3] = f_lt[1][3]
    flt[4] = f_lt[0][4]
    vlt = v

    pop2_obj.pick = -pk_lt
    f_st,v_st = pop2_obj.gen_nf(exec_param['ixf_st'])

    pop2_obj.pick = -pk_st
    f_ss,v_ss = pop2_obj.gen_nf(exec_param['ixf_st'])

    maxpos = float(exec_param['mpos'])
    p0 = prev_pos/maxpos
    nf = len(f)
    pp0 = get_pos_param(flt.reshape((1,nf)),vlt.reshape((1,nf)),p0, exec_param, f_st=f_st.reshape((1,nf)), v_st=v_st.reshape((1,nf)), use_k = k)

    p0_new = pp0[-1] # a position from pop
    p0_new = np.sign(p0_new)*((np.abs(p0_new)*maxpos/min_trd+0.5).astype(int)*min_trd)

    # check the p0_new
    prev_pos = pop2_obj.p0
    prev_sz = pop2_obj.prev_sz
    new_pos = p0_from_stst(k, p0_new, p2_obj.p0, p2_obj.prev_sz, exec_param, f_st, f_ss)

    pop2_obj.prev_sz = new_pos - prev_pos
    pop2_obj.p0 = new_pos
    return new_pos, flt, f_st,v_st

class AR1_Model :
    def __init__(self, logger, cfg_fname, fm=None, pop=None, dump_fname=None) :
        self.logger = logger
        self.cfg_fname = cfg_fname
        self.fm = fm
        self.pop = pop
        self.dump_fd = None
        self.dump_fname =dump_fname

    def sod(self) :
        self.param = read_ar1_config(self.cfg_fname)
        self.logger.logInfo('read AR1 parameter:\n%s'%(str(self.param)))
        self._load()
        self.fm._init()
        self.trigger_time = gen_trigger_time(self.fm.state_obj.std.v[0,:], self.param['barsec'], self.param['trigger_cnt'])

        self.n = int(self.param['n'])
        self.nf = len(self.param['ixf'])

        self.md = {} # key with bar_utc: 'bar': md_bar, 'snap': [ {'bar': md_bar, 'fv': [nf,3,2], 'p0': p0
        self.fv = np.zeros((self.n,3,self.nf,2))
        self.p0 = np.zeros((self.n))

    def _fm_fname(self, trade_day) :
        return'/home/mts/run/recovery/strat/TSC-7000-%s/fm_%s'%(self.param['strat_code'], trade_day)

    def _pop_fname(self, trade_day) :
        return '/home/mts/run/recovery/strat/TSC-7000-%s/pop_%s'%(self.param['strat_code'], trade_day)

    def _dump_fname(self, trade_day) :
        if self.dump_fname is None :
            self.dump_fname = '/home/mts/run/recovery/strat/TSC-7000-%s/dump_%s'%(self.param['strat_code'], trade_day)
        return self.dump_fname

    def _load(self):
        trd_day=self.param['trd_day']
        if self.fm is None:
            with open(self._fm_fname(trd_day), 'rb') as fd:
                self.fm = dill.load(fd)
        if self.pop is None:
            with open(self._pop_fname(trd_day), 'rb') as fd:
                self.pop = dill.load(fd)
        self.dump_fd = open(self._dump_fname(trd_day), 'at')

    def _eod_dump(self) :
        trd_day=self.param['trd_day']
        tdi = mts_util.TradingDayIterator(trd_day)
        tdi.begin()
        trd_day = tdi.next()
        with open(self._fm_fname(trd_day), 'wb') as fd:
            dill.dump(self.fm,fd)
        with open(self._pop_fname(trd_day), 'wb') as fd:
            dill.dump(self.pop,fd)
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
        self.pop.eod_noscale()
        self._eod_dump()

    def on_snap(self, k, a, md_dict):
        f,v=self.fm.on_snap(k,a,md_dict)
        p0, flt, fst, vst = get_p0(k,f,v,self.pop,self.param)
        self.logger.logInfo('on snap (%i,%f): %f'%(k,a,p0))
        self._dump_log(k,a,p0,md_dict,f)
        return p0

    def on_bar(self, k, md_dict):
        p0 = self.on_snap(k,1,md_dict)
        self.fm.on_bar(k, md_dict)
        return p0

def run_1day(fm, p2, md_days, cfg_fname) :
    symbols = list(md_days.keys())
    ndays, n, l = md_days[symbols[0]].shape
    assert ndays >= 1

    logger =  mts_util.MTS_Logger('AR1_WTI', logger_file = '/tmp/ar1_test')
    mdl = AR1_Model(logger, cfg_fname, fm=fm, pop=p2, dump_fname='/tmp/ar1_test_dump.csv')
    mdl.sod()

    p0 = []
    day = -1  # take the last day as the sim day
              # previous day exists for over-night lr
    for k in np.arange(n).astype(int) :
        md_dict = {}
        for sym in symbols:
            md_dict[sym] = md_days[sym][day,k,:]
        p0.append(mdl.on_snap(k,1,md_dict))
        mdl.on_bar(k,md_dict)
    return mdl, p0

def replay_1day(fm, p2, dump_file, cfg_fname):
    d = np.genfromtxt(dump_file, delimiter=',')
    asert(len(d) > 1)

    logger =  mts_util.MTS_Logger('AR1_WTI', logger_file = '/tmp/ar1_test')
    mdl = AR1_Model(logger, cfg_fname, fm=fm, pop=p2, dump_fname='/tmp/ar1_test_dump.csv')
    mdl.sod()

    k = 0
    p0 = []
    for d0 in d:
        curutc,k,a,p0,md_dict,f=AR1_Model._from_array(d0)
        if np.abs(a-1.0)<1e-10:
            # a bar update
            p0.append([curutc, mdl.on_bar(k,a,md_dict)])
        else :
            p0.append([curutc, mdl.on_snap(k,a,md_dict)])
    return mdl, np.array(p0)

def farr_from_fv(f,v,param,p2_in=None):
    """
    f,v: shape nndays, nf
    """
    import Pop2
    ixf = param['ixf']
    n = param['n']
    nndays,nf = f.shape
    ndays = nndays//n

    out_ixf = [param['ixf'], param['ixf_st']]
    pick = -np.array(param['pick'])
    assert len(pick) == 2, 'len of pick is not 2, assuming first is less than second'

    # generate f_arr based on 
    if p2_in is None :
        # create p2
        f_arr,v_arr,p2=Pop2.run_pop(f,v,n,ixf,pick=pick, out_ixf=out_ixf)
    else :
        # iterate over the p2, assuming p2's eod already performed
        assert np.max(np.abs(p2.fcst[n-1,:,0])) == 0, 'p2 eod not called?'
        f_arr,v_arr,p2=Pop2.run_pop(f,v,n,ixf,pick=pick,out_ixf=out_ixf,p2_in=p2_in)

    # f_arr has shape of nndays, len(pick), len(out_ixf)*nf
    f_arr = f_arr.reshape((nndays,len(pick),len(out_ixf),nf))
    v_arr = v_arr.reshape((nndays,len(pick),len(out_ixf),nf))
    return f_arr, v_arr, p2

def p0_from_farr(f0, v0, f_arr, v_arr, param, p0=0, trd_k0=0, trd_k1=276):
    # f_arr has shape of [nndays, num_pick, num_ixf, nf]
    # num_pick = 2, with [st, lt]
    # num_ixf = 2, with  [lt, st]
    # take f_st as the pick_lt at ixf_st
    # p0, the starting position, if given is in scale of maxpos, i.e. 100
    f_st = f_arr[:,1,1,:]
    v_st = v_arr[:,1,1,:]

    f_ss = f_arr[:,0,1,:]
    v_ss = v_arr[:,0,1,:]

    f = f0.copy()
    v = v0.copy()

    # take f_lt as the ixf
    f_ixf_lt_pk_st = f_arr[:,0,0,:]
    f_ixf_lt_pk_lt = f_arr[:,1,0,:]

    # update the f/v 
    f[:,:3] = f_ixf_lt_pk_lt[:,:3]*4
    f[:,3] =  f_ixf_lt_pk_lt[:,3]
    f[:,4] = f_ixf_lt_pk_st[:,4]

    maxpos = float(param['mpos'])
    min_trd = int(param['min_trade'])
    n = param['n']

    prev_pos = p0
    prev_sz = 0
    nndays = f0.shape[0]
    ndays = nndays//n
    assert ndays*n == nndays, 'f0/v0 shape mismatch!'

    pp = [p0]
    for d in np.arange(ndays):
        for k in np.arange(n):
            if k < trd_k0 or k > trd_k1 :
                new_pos = 0
            else :
                p0 = prev_pos/maxpos
                ix0=d*n+k
                ix1=d*n+k+1
                pp0 = get_pos_param(f[ix0:ix1,:], v[ix0:ix1,:], p0, param, f_st=f_st[ix0:ix1,:], v_st=v_st[ix0:ix1,:], use_k=k)

                p0_new = pp0[-1] # in between [0,1]
                p0_new = np.sign(p0_new)*((np.abs(p0_new)*maxpos/min_trd+0.5).astype(int)*min_trd)
                new_pos = p0_new
                #new_pos = p0_from_stst(k, p0_new, prev_pos, prev_sz, param, f_st[ix0,:], f_ss[ix0,:])

            prev_sz = new_pos - prev_pos
            prev_pos = new_pos
            pp.append(new_pos)

    return np.array(pp), f, f_st, v_st

def p0_from_fv(f,v,param,p2_in=None,p0=0) :
    farr,varr,p2_obj = farr_from_fv(f,v,param,p2_in=p2_in)
    pp, f_lt, f_ixf_st, v_ixf_st, f_ixf_stst, v_ixf_stst = p0_from_farr(f,v,farr,varr,param,p0=p0)
    return pp, p2_obj, f_lt, f_ixf_st, v_ixf_st, f_ixf_stst, v_ixf_stst

def pnl_from_p0(p00, p0, lpx, tcost, fee = 2.0, contract_size=1000) :
    """
    p00: scalar, the starting position
    p0: len(n) integer vector, target position at bar k, k=0,1,2,...,n-1.
        p0[k] is the target position at close of bar k.  bar 0 is the first
        bar from 6:00 to  6:05pm
    lpx: len(n) vector, price at close of each bar
    tcost: scalar, usually half of spread plus a slippage, in price
    fee: scalar, fee per transaction, regardless of size

    return:
       pnl, pnl_cost - 
             length n+1 vector, pnl[k] is the value of cash+asset at close of bar k.
             the last pnl is the ending value.  The first 
    """

    p0d = p0-np.r_[p00,p0[:-1]]
    trd_cst = np.abs(p0d)*tcost*contract_size+np.abs(np.sign(p0d))*fee
    pnl = np.r_[0,(lpx[1:]-lpx[:-1])*p0[:-1]]* contract_size
    pnl_cst = pnl-trd_cst
    return pnl, pnl_cst

def update_backtest_file(p00,p0,exec_fname, pnl_fname, position_fname, md_dict=None, md_dict_key='CL', utc=None, lpx=None,  tcost=0.015, fee=2.0, n=276, symbol_str = 'WTI_N1', venue_str='NYM'):
    """
    exec file: 2021-10-25 14:30:00, 2021-10-25 00:00:00, WTI_N1,  NYM, , , -33, 33, 83.64
    pnl file: 2021-10-25, 1000.00
    position file: 2021-10-22, 0
    """

    if md_dict is not None:
        assert utc is None
        assert lpx is None
        utc = md_dict[md_dict_key][:,:,-1].flatten()
        lpx = md_dict[md_dict_key][:,:,-2].flatten()

    # write pnl and position files
    pnl, pnl_cst = pnl_from_p0(p00, p0, lpx, tcost=tcost, fee=fee)
    pnl_d = np.sum(pnl_cst.reshape((len(pnl_cst)//n,n)),axis=1)
    utc_d = utc.reshape((len(utc)//n,n))[:,0]
    pos_d = p0.reshape((len(p0)//n,n))[:,-1]

    fd_pnl = open(pnl_fname, 'at')
    fd_pos = open(position_fname, 'at')
    tdu = mts_util.TradingDayUtil()
    for pnl0, pos0, utc0 in zip(pnl_d, pos_d, utc_d) :
        td1 = tdu.get_trading_day_dt(dt=datetime.datetime.fromtimestamp(utc0))
        trd_day1 = datetime.datetime.strptime(td1,'%Y%m%d').strftime('%Y-%m-%d')
        fd_pos.write('\n%s, %d'%(trd_day1,pos0))
        fd_pnl.write('\n%s, %.2f'%(trd_day1,pnl0))
    fd_pnl.close()
    fd_pos.close()

    # write exec file
    pp00 = np.r_[p00, p0]
    ixe = np.nonzero(np.abs(pp00[1:]-pp00[:-1]) > 1e-10)[0]
    if len(ixe) == 0:
        print ('nothing changed!')
        return

    ex = []
    for ix0 in ixe:
        ex_arr = []
        utc0 = utc[ix0]
        dt0 = datetime.datetime.fromtimestamp(utc0)
        trd_day = tdu.get_trading_day_dt(dt=dt0, snap_forward=False)
        ex_arr.append(dt0.strftime('%Y-%m-%d %H:%M:%S'))
        ex_arr.append(trd_day + ' 00:00:00')
        ex_arr.append(symbol_str)
        ex_arr.append(venue_str)
        ex_arr.append('')
        ex_arr.append('')
        ex_arr.append('%i'%(p0[ix0]))
        ex_arr.append('%i'%(p0[ix0]-pp00[ix0]))
        ex_arr.append('%f'%(lpx[ix0]))
        ex.append(ex_arr)
    with open(exec_fname, 'at') as fd:
        np.savetxt(fd, np.array(ex), fmt='%s', delimiter=',')

def run_offline(fm_in, param=None, p2_in = None, md_dict=None, sday=None, eday=None, param_fname=None,p0_in=0) :
    fm_obj = copy.deepcopy(fm_in)
    import forecast

    if param is None :
        param = read_ar1_config(param_cfg)

    if md_dict is None :
        # get md_dict from sday to eday that includes the
        # overnight return of sday
        tdi = mts_util.TradingDayIterator(sday,sday)
        tdi.begin()
        sday = tdi.prev()
        md_dict0 = forecast.get_md_days(ar1_md.mts_repo_mapping.keys(), sday, eday)
        md_dict = {}
        for sym in md_dict0.keys() :
            md_dict[sym] = md_dict0[sym][1:,:,:]

    f,v,stlrh = fm_obj.run_days(md_dict)

    p0, p2_obj, f_lt, f_ixf_st, v_ixf_st, f_ixf_stst, v_ixf_stst = p0_from_fv(f,v,param,p2_in=p2_in,p0=p0_in)
    lpx = md_dict['CL'][:,:,-2].flatten()
    pnl, pnl_cst = pnl_from_p0(p0_in, p0[1:], lpx, tcost=0.01, fee = 2.0)
    return fm_obj, p2_obj, p0, pnl, pnl_cst

def load_exec(fname, strat_name) :
    """
    CLZ1,TSC-7000-370,4-0-1635953753744126188,80639:M:4648129TN0147252,2,9,81.11,20211103-15:35:53.779,,1635953753791070
    """
    v = np.genfromtxt(fname,delimiter=',',dtype='str')
    ix = np.nonzero(v[:,1] == strat_name)[0]
    v = v[ix]
    ix_ff = np.nonzero(v[:,4] == '2')[0]
    ix_f = np.r_[ix_ff, np.nonzero(v[:,4] == '1')[0]]
    ix_f.sort()
    v = v[ix_f]
    ix_f = np.searchsorted(ix_f, ix_ff)

    sz_ix = 5
    sz_cs = np.cumsum(v[:,sz_ix].astype(int))
    sz = sz_cs[ix_f] - np.r_[0, sz_cs[ix_f[:-1]]]

    s_ix = np.r_[0, ix_f[:-1]+1]
    px_ix = 6
    px = v[:,px_ix].astype(float)[s_ix]

    tm_ix = -1
    utc = v[:,tm_ix].astype(int)[s_ix]/1000000
    return utc, sz, px
