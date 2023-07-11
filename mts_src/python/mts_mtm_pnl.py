import numpy as np
import mts_repo
import symbol_map
import mts_util
import os
import copy
import traceback

def write_mtm_pos(pos_dict, cur_day, eod_pos_path='/tmp'):
    """
    pos_dict[mts_contract][strat] = [qty, vap, pnl, mtm, lpx, mtm_day, utc, fx, fx_rate]
    """
    print('writing ', cur_day)
    if len(pos_dict.keys()) == 0:
        return
    fn = os.path.join(eod_pos_path, 'eod_pos_'+cur_day+'_mtm.csv')
    dt = cur_day + '-17:00:03'
    ret_str = ''
    for sym in pos_dict.keys():
        pd = pos_dict[sym]
        for strat in pd.keys():
            qty, vap, pnl, mtm, lpx, mtm_day, utc, fx, fx_rate = pd[strat]
            ret_str += '%s,%s,%s,%d,%f,%.2f,%d,%s,%f,%f\n'%(dt,strat,sym,qty,vap,mtm_day,utc,fx,fx_rate,lpx)
    with open(fn,'wt') as fp:
        fp.write(ret_str)

def gen_mtm(sday=None, eod_pos_fn = 'eod_pos.csv', eod_pos_path='/home/mts/run/recovery', md_dict_in=None, eday=None):
    """
    format of 
    20220816-17:00:03,TSC-7000-380,Gilt_202209,-88,115.95310606,-22618.33,1660665359679014,GBP

    sday, eday: in format of yyyymmdd, all inclusive, the dates mtm pnl files will be generated.
                if eday is None, use the last day from eod_pos_fn.
    md_dict_in: the saved md_dict from previous run
    """
    if sday is None:
        # run previous trading day's mtm pnl
        tdu = mts_util.TradingDayUtil()
        sday = tdu.get_trading_day(snap_forward=True)
        tdi = mts_util.TradingDayIterator(sday)
        tdi.begin()
        sday=tdi.prev()

    pos = np.genfromtxt(os.path.join(eod_pos_path, eod_pos_fn), delimiter=',', dtype='str', skip_header=67)
    pos_dict = {}  # {'mts_contract': {'strategy': [qty, vap, pnl, mtm, lpx, mtm_day, lastutc, fx, fx_rate]}}
    md_dict = {} # {'mts_contract': {'days': [yyyymmdd...], 'close_px':[],'contract_size', 'tick_size', 'fx_rate'}}
    if md_dict_in is not None:
        md_dict = copy.deepcopy(md_dict_in)

    tdi = mts_util.TradingDayIterator(sday)
    tdi.begin()
    # get a start day 1 week earlier than sday, just to prime the pump
    for i in np.arange(3):
        sday0=tdi.prev()

    cur_day = ''
    prev_pos_dict = {}
    smap = symbol_map.SymbolMap()
    fx_map = symbol_map.FXMap()
    repo = mts_repo.MTS_REPO("/home/mts/run/repo/mts_live", smap)

    if eday is None:
        eday = pos[-1][0].split('-')[0]
    print('generating mtm pnl from ', sday, ' to ', eday, ' reading days starting from ', sday0)
    for pos0 in pos:
        day0 = pos0[0].split('-')[0]
        if day0 > eday:
            break
        if day0 < sday0 :
            continue
        if day0 != cur_day:
            if cur_day >= sday and cur_day <= eday:
                write_mtm_pos(pos_dict, cur_day, eod_pos_path=eod_pos_path)
            prev_pos_dict = copy.deepcopy(pos_dict)
            pos_dict = {}
            cur_day = day0

        # parse this line 
        strat = pos0[1]
        mts_contract = pos0[2]
        qty = int(pos0[3])
        vap = float(pos0[4])
        pnl = float(pos0[5])
        utc = int(pos0[6])
        fx = pos0[7]

        # get the previous adjust
        pnl_adj = 0
        qty0, vap0, pnl0, mtm0, lpx0, pnl_adj = [0,0,0,0,0,0]
        if mts_contract in prev_pos_dict.keys() and strat in prev_pos_dict[mts_contract].keys():
            qty0, vap0, pnl0, mtm0, lpx0 = prev_pos_dict[mts_contract][strat][:5]
            pnl_adj = mtm0-pnl0

        # current day's total value
        if mts_contract not in md_dict.keys():
            try:
                tinfo = smap.get_tinfo(mts_contract, day0, is_mts_symbol=True)
                contract_size = float(tinfo['point_value'])
                md_dict[mts_contract] = {'days':{}, 'contract_size':contract_size, 'tick_size':tinfo['tick_size'], 'currency':tinfo['currency']}
            except:
                print('failed to get contract size for ', mts_contract, ' on ', day0)
                traceback.print_exc()
                continue

        # getting the contract_size and lpx from md_dict
        contract_size = md_dict[mts_contract]['contract_size']
        if day0 in md_dict[mts_contract]['days'].keys():
            lpx = md_dict[mts_contract]['days'][day0]
        else:
            try:
                bar = repo.get_bars(mts_contract, day0, day0, barsec=300, is_mts_symbol=True, get_holiday=True, get_roll_adj=False, cols=['lpx'])
                lpx = bar[-1,-1,0]
            except:
                print('failed to get market data for ', mts_contract, ' on ', day0, 'using previous days lpx:', lpx0)
                #traceback.print_exc()

                lpx = lpx0
            md_dict[mts_contract]['days'][day0] = lpx

        # calculate daily mtm from current and previous pnl/mtm
        mtm = pnl + qty*(lpx-vap)*contract_size
        mtm_day = mtm-pnl_adj

        # convert pnl to us dollar
        assert md_dict[mts_contract]['currency'] == fx
        try:
            fx_rate = 1.0
            if fx != 'USD':
                fx_rate = fx_map.get(fx,day0)
        except:
            print('failed to get fx_rate for %s(%s) on %s'%(mts_contract,fx,day0))
            fx_rate = 1.0
        mtm_day *= fx_rate

        # save the line
        if mts_contract not in pos_dict.keys():
            pos_dict[mts_contract] = {}
        pos_dict[mts_contract][strat] = [qty, vap, pnl, mtm, lpx, mtm_day, utc, fx, fx_rate]

    # write the last day
    if cur_day >= sday and cur_day <= eday:
        write_mtm_pos(pos_dict, cur_day, eod_pos_path=eod_pos_path)

    return md_dict
