import numpy as np
import dill
import mts_util
import ar1_sim
import ar1_md
import ar1_model
import pop4
import os
import copy
import traceback
import datetime

def read_ens_cfg(cfg_file, mkt='WTI_N1'):
    """
    ens config file
    """
    mp_dict = {}
    cfg = mts_util.MTS_Config(cfg_file)
    mkey = '.'.join(['mkt',mkt,''])
    pkeys = [('n',int),('barsec',int),('symbol',str),('mts_symbol',str),('mts_venue',str),('ens_scale',float),('ens_maxpos',int),('ens_weight',float),('ref_weight',float),('contract_size',int),('pnl_tcost',float),('strat_code',int),('strategy_key',str),('min_trade',int),('trigger_cnt',int),('fee',float),('persist_path',str)]
    for (key,tp) in pkeys:
        mp_dict[key]=cfg.get(mkey+key, type_func=tp)
    mp_dict['trading_hour'] = cfg.getArr(mkey+'trading_hour',type_func=float)
    mp_dict['sw_tsc'] = ar1_model.get_sw(mp_dict['strategy_key'], str_code='TSC')
    mp_dict['sw_tsd'] = ar1_model.get_sw(mp_dict['strategy_key'], str_code='TSD')

    param = copy.deepcopy(mp_dict)
    param_dict = {}
    mp_dict['ens_model'] = {}
    model_keys = cfg.listSubKeys(mkey+'ens_model')
    for mk in model_keys:
        mk0 = '.'.join([mkey+'ens_model', mk, ''])
        mdl_id = int(mk)
        param0 = copy.deepcopy(param)
        param0['name'] = 'fm_%d'%(mdl_id)
        pkeys = [('model',str),('tcost',float),('maxpos',int)]
        for (key,tp) in pkeys:
            param0[key]=cfg.get(mk0+key, type_func=tp)
        param0['ixf'] = cfg.getArr(mk0+'ixf',type_func=int)
        param0['wpos'] = 0
        param0['fscale']=[]
        param0['vscale'] = [1,0,3]
        param0['k0_k1_list'] = []
        param_dict[mdl_id]=copy.deepcopy(param0)
        groups = []
        cnt = int(cfg.arrayLen('%s.groups'%(mk0)))
        for i in np.arange(cnt):
            k0,k1,wt=cfg.getArr('%s.groups[%d]'%(mk0,int(i)),type_func=float)
            groups.append([int(k0), int(k1), float(wt)])
        mp_dict['ens_model'][mdl_id]=copy.deepcopy(groups)
    return mp_dict, param_dict

def agg_pos_day(mp_dict, pos_dict, karr, log_func=print):
    """
       pos = agg_pos_day(n, mp_dict, pos_dict, karr)
    input:
        mp_dict: {'ens_scale', 'ens_maxpos', 'mkw_dict': {model_id: [ [ k0, k1, wt ], ..., ]}}, i.e. from read_ens_cfg(cfg_file)
        pos_dict: {mdl_id: parr } length nk of position, same for all models, mdl_id integer
        karr: length nk array of k, bar number, same for all models. 0<=k<n, can be float discontinous
    return :
       length nk aggregated positions (scaled by 'ens_scale', clipped by 'ens_maxpos') 
       on bar time given by karr (see note)
    Note - k can be a float.  at 6pm, k=0. k increases linearly until 6:05 to k=1, etc.
           The ens group boundary 'k0,k1' in mkw_dict refer to the starting bar count. 
           i.e. when k0=0, it means to include the position that would be effective 
           during bar 0, i.e. 6:00 to 6:05.
           This effectively includes the last position at n-1.
           For a group defined as (k0, k1), when k = k0-1, the inclusion 
           weight is 0, when k=k0-0.5, it's 0.5, when k=k0, it's inclusion 
           weight is 1, same for k1.

           For example, k0,k1 = (1,2), means 

         18:00          18:05          18:10            18:15
           |--------------|--------------|----------------|
           |       ^      |              |            ^   |
                   |                                  |
                bar=0,a=0.5 (k=0.5)             bar=2,a=0.8 (k=2.8)
                inclusion=0.5                   inclusion=0.2
           

    """
    nk = len(karr)
    n = mp_dict['n']
    pos = np.zeros(nk)
    mkw_dict = mp_dict['ens_model']
    ens_maxpos = mp_dict['ens_maxpos']
    ens_scale = mp_dict['ens_scale']
    for mdl_id in mkw_dict.keys():
        if mdl_id not in mkw_dict.keys():
            continue
        kw = mkw_dict[mdl_id] # [ [k0, k1, wt] ]
        if len(kw) == 0:
            continue

        parr = pos_dict[mdl_id]
        assert len(parr) == nk
        for i, (k, p) in enumerate(zip(karr, parr)):
            if abs(p) < 1e-10:
                continue
            # allowing for a trigger time as fraction of k
            # i.e. 0.5 is between bar 0 and bar 1
            k_prev = int(k)
            k_next = k_prev+1
            for kw0 in kw:
                k0,k1,wt=kw0
                for kk, frac in zip([k_prev, k_next], [k_next-k, k-k_prev]):
                    if frac*wt < 1e-10:
                        continue
                    kk = (kk%n)
                    if kk>=k0 and kk<=k1:
                        #debug
                        log_func('!!! Got position: mdl_id(%d), ens_group(%s), pos(%f), k(%f), kk(%d), k0(%d), k1(%d), pos(%f), wt(%f), frac(%f)'%(int(mdl_id), str(kw0), float(p), float(k), int(kk), int(k0), int(k1), float(pos[i]), float(wt), float(frac)))
                        pos[i]+=(p*wt*frac)
                        log_func('!!! Got position: resulting pos(%f)'%(float(pos[i])))

    return np.clip(pos*ens_scale,-ens_maxpos,ens_maxpos)

def run_offline_model(fm_in, param, md_dict):
    """
    fm_out, p0, pnl, f,v = run_offline_model(fm_in, param, md_dict)
    Run one model for md_dict, returns the fm/p0/pnl/f/v
    input:
        fm_in: the starting model.  this object is NOT modified, as a deepcopy is used and returned
        param: read from the model's LiveHeader
        md_dict: the market data for the days to be run
    return:
        fm_out: the object after run_days()
        p0: length nndays+1 vector, with p0[0] being the p00. Note, the p00 is from the fm_in.prev_pos
        pnl: length nndays vector, with lr0 from md_dict's first lr, last p0[-1] is not counted
        f/v: [ndays,nf] f/v from fm.run_days()
        stlrh: [ni,nh] last state input for the model
    """
    fm = copy.deepcopy(fm_in)
    f,v,stlrh = fm.run_days(md_dict)
    symbol = param['symbol']
    md = md_dict[symbol]
    lpx=md[:,:,3].flatten()
    lr=md[:,:,0].flatten()
    try :
        prev_pos = fm.prev_pos
    except :
        prev_pos=0

    # pp length nndays+1, pnl length nndays
    pp, pnl = pop4.p0_from_fv(param['n'], f, v, lpx, param['ixf'], param['tcost'], param['contract_size'], param['pnl_tcost'], param['maxpos'], p0=prev_pos, fscale=None, vscale=param['vscale'], no_trade_k0_k1=None, lr0=lr[0])

    # set the prev_pos for fm_out
    fm.prev_pos = pp[-1]
    return fm, pp, pnl, f, v, stlrh

def run_offline_ens(mp_dict,  fm_dict, param_dict, md_dict, fm_out_dict=None, pos_dict=None):
    """
        posa, pos_dict, fm_out_dict = run_offline_ens()

    Input:
        mp_dict: {'ens_scale', 'ens_maxpos', 'mkw_dict': {model_id: [ [ k0, k1, wt ], ..., ]}}, i.e. from read_ens_cfg(cfg_file)
        fm/param_dict: {mdl_id: fm_obj/live_header_param}
        md_dict: {symbol: [ndays,n,>5]}
    return:
        posa, the aggregated position of length nndays+1, posa[0] is the first (incoming position from fm)
        pos_dict: {model_id: nndays+1 model_position}
        fm_out_dict: {model_id: {'fm_out','f','v','stlrh','pnl'}
    """

    # collect model position
    symbol = mp_dict['symbol']
    n = mp_dict['n']
    ndays, n0, _ = md_dict[symbol].shape
    assert n==n0
    
    # run all the models that are 'picked' in fm_dict
    # i.e. models without ens_model.groups are not loaded (and run)
    if fm_out_dict is None:
        pos_dict = {}
        fm_out_dict = {}
        for mdl_id in fm_dict.keys():
            print('getting f,v, from model ', mdl_id)
            fm_in = fm_dict[mdl_id]
            param = param_dict[mdl_id]
            fm_out,pp,pnl,f,v,stlrh = run_offline_model(fm_in, param, md_dict) #fm_in not modified
            pos_dict[mdl_id] = pp.flatten().copy() # pp[0] is p00, the entering position from fm
            fm_out_dict[mdl_id] = {'fm_out':fm_out, 'f':f, 'v':v, 'stlrh':stlrh, 'pnl':pnl}

        # for debug
        #return fm_out_dict, pos_dict

    # generate posa and pnl
    print('aggregating positions')
    karr = np.arange(n+1).astype(int) # k = [0,1,2,...,n-1,n], go with pp, the at bar open
    posa = []
    for d in np.arange(ndays).astype(int):
        # get a pos_dict for day d
        pos_dict0 = {}
        for mdl_id in pos_dict.keys():
            pos_dict0[mdl_id] = pos_dict[mdl_id][d*n:(d+1)*n+1]
        # run it with ens
        posa.append(agg_pos_day(mp_dict, pos_dict0, karr).copy())
    posa = np.r_[np.array(posa)[:,:-1].flatten(),posa[-1][-1]] # back to nndays+1

    # get pnl for posa
    print('calculating aggregated pnl')
    lpx = md_dict[mp_dict['symbol']][:,:,3]
    lr = md_dict[mp_dict['symbol']][:,:,0]
    lpx_ra = pop4.get_lpx_from_lr(lpx, lr)
    stdv_ref = fm_out_dict[list(fm_out_dict.keys())[0]]['fm_out'].state_obj.std.v[0].copy()
    tcost_bar = pop4.tcost_from_std(stdv_ref, mp_dict['pnl_tcost'])

    _, pnla, shpa = pop4.pnl_from_pos_tcost(lpx_ra, posa, tcost_bar, mp_dict['ens_maxpos'], mp_dict['contract_size'], lr0=lr[0,0], fee=mp_dict['fee'], clip_maxpos=True)

    print('done, pnl:', np.sum(pnla), ' shp:', shpa)
    return posa, pnla, shpa, pos_dict, fm_out_dict

def load_fm(trd_day, mp_dict, param_dict):
    """
    models without groups defined in mkw_dict are not loaded
    """
    mkw_dict = mp_dict['ens_model']
    fm_dict = {}
    for mdl_id in mkw_dict.keys():
        groups = mkw_dict[mdl_id]
        if len(groups) == 0:
            continue
        param = param_dict[mdl_id]
        fm_file0 = ar1_model.AR1_Model._fm_fname(trd_day, param)
        fm_dict[mdl_id] = dill.load(open(fm_file0,'rb'))
    return fm_dict

def backtest_eod_ens(trd_day=None,\
                 config_path = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US', \
                 ens_cfg_file = 'ENS.cfg',\
                 fm_file = 'fm',\
                 mkt='WTI_N1',\
                 ref_cfg_fname='/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US/LiveHO-INTRADAY_MTS_AR1_WTI_ENS.cfg', \
                 ref_fm_dir='/home/mts/run/recovery/strat/TSC-7000-370', \
                 run_days=1, \
                 write_backoffice=True, \
                 backoffice_scale=1):
    if trd_day is None :
        dt = datetime.datetime.now()
        if dt.weekday() >= 5:
            print('backtest_eod not running on a weekend: %s'%(str(dt)))
            return None
        # assuming it is run in between 5pm to 6pm
        tdu = mts_util.TradingDayUtil()
        trd_day = tdu.get_trading_day(snap_forward=True)
        tdi = mts_util.TradingDayIterator(trd_day)
        tdi.begin()
        trd_day = tdi.prev()

    tdi = mts_util.TradingDayIterator(trd_day,trd_day)
    trd_day_end=tdi.begin()
    while run_days > 1:
        trd_day_end = tdi.next()
        run_days-=1

    mp_dict, param_dict = read_ens_cfg(os.path.join(config_path, ens_cfg_file), mkt=mkt)
    ens_weight = mp_dict['ens_weight']
    ref_weight = mp_dict['ref_weight']
    ens_maxpos = mp_dict['ens_maxpos']

    fm_dict = load_fm(trd_day, mp_dict, param_dict)
    md_dict = ar1_md.get_md_days(ar1_md.mts_repo_mapping.keys(), trd_day,trd_day_end, mp_dict['barsec'])

    print('running offline model from ', trd_day, ' to ', trd_day_end)
    posa, pnla, shpa, pos_dict, fm_out_dict = run_offline_ens(mp_dict, fm_dict, param_dict, md_dict)

    # match with the online results, saved as the next trading day
    next_day = tdi.next()
    try :
        print('matching, loading fm on ', next_day)
        for mdl_id in fm_dict.keys():
            fm_file = ar1_model.AR1_Model._fm_fname(next_day, param_dict[mdl_id])
            fm_live = dill.load(open(fm_file,'rb'))
            fm_offline = fm_out_dict[mdl_id]['fm_out']
            if not ar1_sim.match_fm(fm_live, fm_offline):
                print('failed to match on ', mdl_id)
            else :
                print('model ', mdl_id, ' matched')
            fm_live = None
    except:
        traceback.print_exc()

    ref_trd_k0_k1_all=(0,276)
    ref_trd_k0_k1_us =(156,276)
    p0_arr_ref, pnl_arr_ref, _, f_ref, v_ref, fm_ref_out = ar1_sim.backtest_eod(trd_day=trd_day, cfg_fname=ref_cfg_fname, fm_dir=ref_fm_dir, trd_k0_k1_all=ref_trd_k0_k1_all, trd_k0_k1_us=ref_trd_k0_k1_us, fail_check=False, update_backoffice=False, md_dict=md_dict)
    fm_ref_out_dict={'fm_out': fm_ref_out, 'f':f_ref, 'v':v_ref, 'p0_arr_ref':p0_arr_ref, 'pnl_arr_ref':pnl_arr_ref}

    # write backoffice for all bar positions
    posa_all = np.clip(posa*ens_weight + p0_arr_ref[0]*ref_weight, -ens_maxpos, ens_maxpos)
    pnla_all = pnla*ens_weight + pnl_arr_ref[0]*ref_weight
    posa_us = np.clip(posa*ens_weight + p0_arr_ref[1]*ref_weight, -ens_maxpos, ens_maxpos)
    pnla_us = pnla*ens_weight + pnl_arr_ref[1]*ref_weight

    if write_backoffice:
        write_backoffice_files(mp_dict, posa_all, posa_us, md_dict, backoffice_scale)

    return posa, pnla, p0_arr_ref, pnl_arr_ref, shpa, pos_dict, md_dict, fm_out_dict, fm_ref_out_dict

def dump_fm_out(fm_out_dict, trd_day, param_dict):
    for mdl_id in fm_out_dict.keys():
        fm = fm_out_dict[mdl_id]['fm_out']
        fm_file0 = ar1_model.AR1_Model._fm_fname(trd_day, param_dict[mdl_id])
        print('saving model ', mdl_id, ' to ', fm_file0)
        with open(fm_file0, 'wb') as fp:
            dill.dump(fm, fp)

def write_backoffice_files(mp_dict, posa, posa_us, md_dict, pos_scale, us_k0_k1 = (147,276), overnight=(0, 72), backoffice_path = '/home/mts/upload/ZFU_STRATS'):
    n = mp_dict['n']
    pnl_tcost = mp_dict['pnl_tcost']
    fee = mp_dict['fee']
    md_dict_key = mp_dict['symbol']
    symbol_str = mp_dict['mts_symbol']
    venue_str = mp_dict['mts_venue']

    ndays, n0, _ = md_dict[md_dict_key].shape
    assert n==n0
    p00 = posa[0]
    p0 = posa[1:]
    nndays = len(p0)
    assert nndays == ndays*n
    p0 = p0.reshape((ndays,n))
    p0_us = posa_us[1:].reshape((ndays,n))

    # all good, update the backtest files
    exec_fname    = os.path.join(backoffice_path, 'INTRADAY_MTS_AR1_WTI/ExecutionLog-INTRADAY_MTS_AR1_WTI')
    exec_fname_us = os.path.join(backoffice_path, 'INTRADAY_MTS_AR1_WTI_US/ExecutionLog-INTRADAY_MTS_AR1_WTI_US')
    pnl_fname     = os.path.join(backoffice_path, 'INTRADAY_MTS_AR1_WTI/Pnl-INTRADAY_MTS_AR1_WTI')
    pnl_fname_us  = os.path.join(backoffice_path, 'INTRADAY_MTS_AR1_WTI_US/Pnl-INTRADAY_MTS_AR1_WTI_US')
    pos_fname     = os.path.join(backoffice_path, 'INTRADAY_MTS_AR1_WTI/Positions-INTRADAY_MTS_AR1_WTI')
    pos_fname_us  = os.path.join(backoffice_path, 'INTRADAY_MTS_AR1_WTI_US/Positions-INTRADAY_MTS_AR1_WTI_US')

    exec_fn = [exec_fname, exec_fname_us]
    pnl_fn = [pnl_fname, pnl_fname_us]
    pos_fn = [pos_fname, pos_fname_us]
    p0_arr = [p0.flatten(), p0_us.flatten()]
    for p0, exef, pnlf, posf in zip (p0_arr, exec_fn, pnl_fn, pos_fn):
        ar1_sim.update_backtest_file(p00,p0,exef,pnlf,posf,md_dict=md_dict, md_dict_key=md_dict_key, tcost = pnl_tcost, fee=fee, n=n, symbol_str=symbol_str, venue_str=venue_str)

