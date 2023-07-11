import numpy as np
import mts_util
import dill
import Pop
import ar1_model
import datetime
import yaml
import copy
import os

def run_1day(fm, md_days, symbol, cfg_fname, trd_day) :
    symbols = list(md_days.keys())
    ndays, n, l = md_days[symbols[0]].shape
    assert l >= 5

    logger =  mts_util.MTS_Logger('AR1_WTI', logger_file = '/tmp/ar1_test')
    mdl = ar1_model.AR1_Model(logger, cfg_fname, symbol, fm=fm, dump_fname='/tmp/ar1_test_dump.csv')
    mdl.sod(trd_day, check_trade_day=False)

    p0 = []
    day = -1  # take the last day as the sim day
              # previous day exists for over-night lr
    for k in np.arange(n).astype(int) :
        md_dict = {}
        for sym in symbols:
            md_dict[sym] = md_days[sym][day,k,:]
        #p0.append(mdl.on_snap(k,1,md_dict))
        #mdl.on_bar(k,md_dict)
        p0.append(mdl.on_bar(k,md_dict))
    mdl.eod()
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

def match_fm(fm_1, fm_2) :
    # check f0 and stlrh
    f1=fm_1.fcst_obj
    f2=fm_2.fcst_obj
    f0_diff=np.max(np.abs(f1.f0-f2.f0))
    st_diff=np.max(np.abs(f1.stlrh-f2.stlrh))
    pos_diff=fm_1.prev_pos-fm_2.prev_pos

    print ('fm_f0_diff: ', f0_diff, 'fm_stlrh_diff: ', st_diff, 'pos_diff',pos_diff)

    try :
        assert f0_diff < 1e-9, 'f0_diff'
        assert st_diff < 1e-9, 'st_diff'
        assert pos_diff < 1e-9, 'pos_diff'
    except :
        return False

    return True

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
        p2=copy.deepcopy(p2_in)
        # iterate over the p2, assuming p2's eod already performed
        assert np.max(np.abs(p2.fcst[n-1,:,0])) == 0, 'p2 eod not called?'
        f_arr,v_arr,p2=Pop2.run_pop(f,v,n,ixf,pick=pick,out_ixf=out_ixf,p2_in=p2_in)

    # f_arr has shape of nndays, len(pick), len(out_ixf)*nf

    f_arr = f_arr.reshape((nndays,len(pick),len(out_ixf),nf))
    v_arr = v_arr.reshape((nndays,len(pick),len(out_ixf),nf))
    return f_arr, v_arr, p2

def p0_from_farr(f0, v0, param, lpx, p0=0, trd_k0=0, trd_k1=276, barsec=300):
    """
    This reads the param['trd_hour'] for overnight plus the trd_k0_k1. 
    If the 'trd_hour' is (7, 21) and k0_k1 is (140,276) then the overnight is set to be 21,
    allowing trade before 21 and from k=140 to 276.  Mainly setup for running US profile.
    """
    f = f0.copy()
    v = v0.copy()
    n = param['n']
    prev_pos = p0
    nndays = f0.shape[0]
    ndays = nndays//n
    assert ndays*n == nndays, 'f0/v0 shape mismatch!'

    stdvh = []
    # use previous day's v0 as stdv
    for d in np.arange(ndays):
        sd = np.sqrt(v0[d*n:(d+1)*n,0])
        stdvh.append(np.r_[sd[-1],sd[:-1]])

    trd_hours = param['trading_hour']  # i.e. [7, 21.5]
    overnight_k = n+1
    if trd_hours[1]>=18:
        overnight_k = int((trd_hours[1]-18)*(3600/barsec)+0.5)

    pp = [p0]
    for d in np.arange(ndays):
        ep = ar1_model.gen_exec_param(param,stdvh[d],lpx[d*n:(d+1)*n])
        for k in np.arange(n):
            if (k < trd_k0 or k > trd_k1) and k>=overnight_k: # allow over-night cover
                new_pos = 0
            else :
                p0 = prev_pos
                ix0=d*n+k
                ix1=d*n+k+1
                pp0 = ar1_model.get_pos_param(f[ix0:ix1,:], p0, ep, use_k=k)
                new_pos = pp0[-1]
            prev_pos = new_pos
            pp.append(new_pos)

    #return np.array(pp), f, f_st, v_st
    return np.array(pp)

def p0_from_fv(f,v,param,lpx,trd_k0_k1=[(0,276)],p0=0):
    pp_arr = []
    for (k0, k1) in trd_k0_k1:
        pp = p0_from_farr(f,v,param,lpx,p0=p0,trd_k0=k0, trd_k1=k1)
        pp_arr.append(pp)
    return pp_arr

def pnl_from_p0(p00, p0, lpx, tcost, fee = 2.0, contract_size=1000, lr0=0) :
    """
    p00: scalar, the starting position
    p0: len(n) integer vector, target position at bar k, k=0,1,2,...,n-1.
        p0[k] is the target position at close of bar k.  bar 0 is the first
        bar from 6:00 to  6:05pm
    lpx: len(n) vector, price at close of each bar
    tcost: scalar, usually half of spread plus a slippage, in price
    fee: scalar, fee per transaction, regardless of size
    lr0: the first log return at 6:05, used to apply p00 for initial pnl

    return:
       pnl, pnl_cost - 
             length n vector, pnl[k] is the value of cash+asset at close of bar k.
             the last pnl is the ending value.  pnl[0] is pnl achieved by p00 and lr0,
             pnl[-1] is the pnl achieved by p0[-2] with lpx[-1]-lpx[-2]. 
             Note p0[-1] not included in pnl calculation as there is no price.
             
    """
    p0d = p0-np.r_[p00,p0[:-1]]
    trd_cst = np.abs(p0d)*tcost*contract_size+np.abs(np.sign(p0d))*fee
    lpx0 = np.r_[lpx[0]/np.exp(lr0),lpx]
    pnl = (lpx0[1:]-lpx0[:-1])*np.r_[p00,p0[:-1]]* contract_size
    #pnl = np.r_[0,(lpx[1:]-lpx[:-1])*p0[:-1]]* contract_size
    pnl_cst = pnl-trd_cst
    return pnl, pnl_cst

def update_backtest_file(p00,p0,exec_fname, pnl_fname, position_fname, md_dict, md_dict_key='CL', tcost=0.025, fee=2.0, n=276, symbol_str = 'WTI_N1', venue_str='NYM'):
    """
    exec file: 2021-10-25 14:30:00, 2021-10-25 00:00:00, WTI_N1,  NYM, , , -33, 33, 83.64
    pnl file: 2021-10-25, 1000.00
    position file: 2021-10-22, 0
    """

    utc = md_dict[md_dict_key][:,:,4].flatten()
    lpx = md_dict[md_dict_key][:,:,3].flatten()
    lr = md_dict[md_dict_key][:,:,0].flatten()

    # write pnl and position files
    pnl_raw, pnl_net = pnl_from_p0(p00, p0, lpx, tcost=tcost, fee=fee, lr0=lr[0])
    pnl, pnl_twap_raw = pnl_from_p0(p00, p0, lpx, tcost=tcost/5, fee=fee, lr0=lr[0])
    pnl, pnl_twap_net = pnl_from_p0(p00, p0, lpx, tcost=tcost/2, fee=fee, lr0=lr[0])

    pnl_d1 = np.sum(pnl_raw.reshape((len(pnl_raw)//n,n)),axis=1)
    pnl_d2 = np.sum(pnl_twap_raw.reshape((len(pnl_twap_raw)//n,n)),axis=1)
    pnl_d3 = np.sum(pnl_net.reshape((len(pnl_net)//n,n)),axis=1)
    pnl_d4 = np.sum(pnl_twap_net.reshape((len(pnl_twap_net)//n,n)),axis=1)

    utc_d = utc.reshape((len(utc)//n,n))[:,0]
    pos_d = p0.reshape((len(p0)//n,n))[:,-1]

    fd_pnl = open(pnl_fname, 'at')
    fd_pos = open(position_fname, 'at')
    tdu = mts_util.TradingDayUtil()
    for pnl1, pnl2, pnl3, pnl4, pos0, utc0 in zip(pnl_d1, pnl_d2, pnl_d3, pnl_d4, pos_d, utc_d) :
        td1 = tdu.get_trading_day_dt(dt=datetime.datetime.fromtimestamp(utc0))
        trd_day1 = datetime.datetime.strptime(td1,'%Y%m%d').strftime('%Y-%m-%d')
        fd_pos.write('%s, %d\n'%(trd_day1,pos0))
        fd_pnl.write('%s, %.0f, %.0f, %.0f, %.0f\n'%(trd_day1,pnl1,pnl2,pnl3,pnl4))
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

def run_offline(fm_in, param=None, md_dict=None, sday=None, eday=None, param_fname=None, trd_k0_k1=[(0,276)], pos_scale=1.0):
    fm_obj = copy.deepcopy(fm_in)
    if param is None :
        symbol = fm_in.state_obj.ind_array[0][0]
        param = ar1_model.read_ar1_config(param_fname, symbol)

    if md_dict is None :
        import ar1_md
        # over-night lr covered in repo's get_bars()
        md_dict = ar1_md.get_md_days(ar1_md.mts_repo_mapping.keys(), sday, eday, int(param['barsec']))

    f,v,stlrh = fm_obj.run_days(md_dict)
    lpx=md_dict[param['symbol']][:,:,3].flatten()
    lr=md_dict[param['symbol']][:,:,0].flatten()
    try :
        prev_pos = fm_in.prev_pos
    except :
        prev_pos=0

    p0_arr = p0_from_fv(f,v,param,lpx,trd_k0_k1=trd_k0_k1,p0=prev_pos)
    fm_obj.prev_pos = p0_arr[0][-1]
    p0_arr=np.array(p0_arr)*pos_scale
    pnl_arr = []
    for p0 in p0_arr :
        pnl, pnl_cst = pnl_from_p0(p0[0], p0[1:], lpx, tcost=0.025, fee = 2.0, lr0=lr[0])
        pnl_arr.append(pnl_cst)
    return fm_obj, p0_arr, pnl_arr, f,v, md_dict

def backtest_eod(trd_day=None, \
                 cfg_fname='/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US/LiveHO-INTRADAY_MTS_AR1_WTI_US.cfg', \
                 fm_dir='/home/mts/run/recovery/strat/TSC-7000-370', \
                 trd_k0_k1_all=(0,276), \
                 trd_k0_k1_us =(156,276), \
                 fail_check=False, \
                 update_backoffice=True, \
                 md_dict=None,
                 pos_scale=1 \
    ):

    if trd_day is None :
        # assuming it is run in between 5pm to 6pm
        tdu = mts_util.TradingDayUtil()
        trd_day = tdu.get_trading_day(snap_forward=False)

    with open(fm_dir+'/fm_'+trd_day,'rb') as f:
        fm_in = dill.load(f)

    # match with the online results, saved as the next trading day
    tdi = mts_util.TradingDayIterator(trd_day,trd_day)
    tdi.begin()
    next_day = tdi.next()

    # all time first, and then US time
    trd_k0_k1 = [trd_k0_k1_all, trd_k0_k1_us]

    # save prev_pos
    try :
        prev_pos=fm_in.prev_pos
    except:
        prev_pos=0
    fm_offline, p0_arr, pnl_arr, f,v, md_dict = run_offline(fm_in, sday=trd_day, eday=trd_day, param_fname=cfg_fname, trd_k0_k1=trd_k0_k1, md_dict=md_dict, pos_scale=pos_scale)

    try:
        fn = os.path.join(fm_dir, 'fm_'+next_day)
        with open(fn,'rb') as f:
            fm_live = dill.load(f)
        matched = match_fm(fm_live, fm_offline)
        if matched:
            print(fn + ' matched with live!')
        else:
            # reasons check fail are mostly from market data:
            # 1. over-night return mismatch
            # 2. rolled to new contract
            # 3. tp restart casugin 300S bars missing volume but not 1S bars
            print (fn + ' check failed!')
            raise 'failed to match ' + fn
    except:
        if fail_check:
            raise 'failed to match! ' + fn

    if update_backoffice:
        # all good, update the backtest files
        exec_fname    = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI/ExecutionLog-INTRADAY_MTS_AR1_WTI'
        exec_fname_us = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US/ExecutionLog-INTRADAY_MTS_AR1_WTI_US'
        pnl_fname     = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI/Pnl-INTRADAY_MTS_AR1_WTI'
        pnl_fname_us  = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US/Pnl-INTRADAY_MTS_AR1_WTI_US'
        pos_fname     = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI/Positions-INTRADAY_MTS_AR1_WTI'
        pos_fname_us  = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US/Positions-INTRADAY_MTS_AR1_WTI_US'
        md_dict_key = 'CL'

        exec_fn = [exec_fname, exec_fname_us]
        pnl_fn = [pnl_fname, pnl_fname_us]
        pos_fn = [pos_fname, pos_fname_us]
        for p0, pnl, exef, pnlf, posf in zip (p0_arr, pnl_arr, exec_fn, pnl_fn, pos_fn):
            update_backtest_file(p0[0],p0[1:],exef,pnlf,posf,md_dict=md_dict, md_dict_key=md_dict_key)

        # update config to the next trading day
        next_trd_day = mts_util.TradingDayIterator.nth_trade_day(trd_day,1)
        cfg = mts_util.MTS_Config(cfg_fname)
        cfg.set('trade_date', next_trd_day)
        cfg.set('data_date', next_trd_day)

    return p0_arr, pnl_arr, md_dict, f, v, fm_offline


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


