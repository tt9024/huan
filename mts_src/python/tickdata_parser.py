import numpy as np
import datetime
import os
import mts_util
import copy
import traceback

###################################################
# utilities to read/parse the tickdata quote/trade
###################################################

# Since tickdata timestamps are local received timestamp
# generally more than 10-milli more than exchange time,
# depending on their local load, could be upto 500 milli.
# this constant is to fix this to a very minimum degree.
#
# this doesn't seem to help the matching, tuning off.
LocalLatencyMilli=-10

def get_trade_tickdata(file_name, time_zone='US/Eastern', px_multiplier=1.0, local_latency_milli=LocalLatencyMilli, mts_venue=None):
    """
    The trade file has the following format
    01/31/2021,18:00:04.281,51.89,1,E,0,,51.89
    in the format of
    date, time, filtered_px, volume, flag, condition, exclude, raw_px

    We are going to use the raw_px instead of filterd_px
    where flag = 'E'
    return array of [utc_milli, trd_px, trd_sz]

    Note, the last 4 columns only available after 7/1/2011, so use the 
    filtered_px if needed.

    When getting lines with 'X', check the d[:,6], either include
    them all, or not at all.  Usually "LEG" has 0 price, so no effects
    For CME venues (CME, NYM, CBT, CEC, MGE)
       - no X trades from TickData,
       - but bpipe live could have BT, ORS, ORT trades, ignored
    For ICE venues (IFUS, IFEU, IFLL, IFLX, IFCA)
       - RFCL,EFP, EFS, BLK(L), BAS from Tickdata
       - ST, CT, B, BT, SBL, BL from bpipe live, matched
    For Eurex venues (EUR, EOP)
       - 1,15,16,22,23 from TickData
       - OPEN,TES,EFPF,VOLA,EFS from bpipe live, matched

    After review the overall match, it is decided to remove all
    X trades from CME and include all X trades from other venues.
    See bbtp.cpp addTCC() for detailed bpipe trade condition code setup.

    6/27 - separate special trade parsing from here initially, to be added
    """
    try :
        do_gz = False
        if file_name[-3:] == '.gz' :
            os.system('gunzip -f ' + file_name)
            file_name = file_name[:-3]
            do_gz = True

        d = np.genfromtxt(file_name, delimiter=',', dtype='|S64')
        n,m = d.shape

        dflag = d[:,4].astype(str)
        ix = np.nonzero(dflag == 'E')[0]
        if len(ix) > 0 :
            d = d[ix,:]
        sz = d[:,3].astype(int)

        pxcol = 7 if m > 7 else 2
        px = d[:,pxcol].astype(float)*px_multiplier

        ix = np.nonzero(sz*px!=0)[0]
        if len(ix) > 0:
            d = d[ix,:]
            sz = sz[ix]
            px = px[ix]

        #includeX = True 
        includeX = False

        """
        if mts_venue is not None:
            includeX=mts_venue not in ['CME', 'NYM', 'CBT', 'CEC', 'MGE']

        # X trades treated differently
        """
        if not includeX and m>6 :
            ix = np.nonzero(d[:,6].astype('str')!='X')[0]
            if len(ix) > 0:
                d = d[ix,:]
                sz = sz[ix]
                px = px[ix]
            else:
                print('no trade (without X) found')
                return None;

        dt0 = ''
        utcs0 = 0.0

        utc=[]
        for d0 in d:
            dts=d0[0].decode() + '-' + d0[1].decode()
            if '.' not in dts: 
                dts += '.000'

            dts0 = dts[:-10]
            sec0 = int(dts[-9:-7])*60 + int(dts[-6:-4])
            sec0 += float(dts[-3:])/1000.0
            if dt0 != dts0 :
                dt = datetime.datetime.strptime(dts0+":00:00.000", '%m/%d/%Y-%H:%M:%S.%f')
                utcs0 = mts_util.TradingDayUtil.dt_to_utc(dt, time_zone)
                dt0 = dts0
            utcs = utcs0 + sec0
            utc.append(utcs)

        trd = np.vstack(((np.array(utc)*1000).astype(int)+local_latency_milli, px, sz)).T

        """
        if do_gz :
            os.system('gzip ' + file_name)
        """
        return trd
    except :
        print ('problem getting trade from ',file_name)
        #traceback.print_exc()
        return None

def get_quote_tickdata(file_name, time_zone='US/Eastern', px_multiplier=1.0, local_latency_milli=LocalLatencyMilli) :
    """
    The quote file has the following format
    02/17/2021,18:00:00.000,61.71,13,61.71,11,E,
    in the format of 
    date, time, bp, bsz, ap, asz, flag, condition

    We are going to read only flag = 'E'
    return array of [utc_milli, bp, bsz, ap, asz]
    """
    try :
        do_gz = False
        if file_name[-3:] == '.gz' :
            os.system('gunzip -f ' + file_name)
            file_name = file_name[:-3]
            do_gz = True

        bp = []; bsz=[]; ap=[]; asz=[]; utc=[]
        dt0 = ''
        utcs0 = 0.0
        with open(file_name, 'rt') as f:
            while True:
                l = f.readline()
                if len(l) == 0:
                    break
                d0 = l.split(',')
                if d0[-2] != 'E':
                    continue

                try:
                    bp0=float(d0[2])*px_multiplier
                    ap0=float(d0[4])*px_multiplier
                    if bp0>ap0-1e-10:
                        # remove the crossed ticks
                        continue
                    bsz0=int(d0[3])
                    asz0=int(d0[5])

                    # time
                    dts=d0[0] + d0[1]
                    dts0 = dts[:-10]
                    sec0 = int(dts[-9:-7])*60 + int(dts[-6:-4])
                    sec0 += float(dts[-3:])/1000.0
                    if dt0 != dts0 :
                        dt = datetime.datetime.strptime(dts0+":00:00.000", '%m/%d/%Y%H:%M:%S.%f')
                        utcs0 = mts_util.TradingDayUtil.dt_to_utc(dt, time_zone)
                        dt0 = dts0
                    utcs = utcs0 + sec0
                    utc.append(utcs)
                    bp.append(bp0)
                    bsz.append(bsz0)
                    ap.append(ap0)
                    asz.append(asz0)
                except:
                    continue

        quote = np.vstack(((np.array(utc)*1000).astype(int)+local_latency_milli, bp, bsz, ap, asz)).T

        """
        if do_gz :
            os.system('gzip ' + file_name)
        """
        return quote
    except :
        print ('problem getting quote from ', file_name)
        #traceback.print_exc()
        return None

def _quote_from_trade(trd_bar) :
    """
    this is for the old days where only trades are avaliable
    trd_bar: [utc, px, sz]
    """
    mid_px = trd_bar[:,1]
    tick_sz = mid_px.mean()*0.0001
    n = len(mid_px)
    return np.vstack((trd_bar[:,0], mid_px-tick_sz, np.ones(n), mid_px+tick_sz, np.ones(n))).T

def sample_OHLC(midpx, midutc, butc) :
    """
    both midutc and butc are integer and are in same unit, i.e. milliseconds
    butc should include the start of first bar, i.e. len(butc) = bars + 1
    return shape [bars,4], each row with [open, high, low, close]
    """
    mp = midpx.copy()
    mutc = midutc.copy()
    rmidx = np.nonzero(np.abs(mp[1:]-mp[:-1])<1e-10)[0]
    if rmidx.size > 0 :
        rmidx += 1;
        mp = np.delete(mp, rmidx)
        mutc = np.delete(mutc, rmidx)

    nbars = len(butc) - 1
    qix = np.clip(np.searchsorted(mutc,butc+0.001)-1,0,1e+10).astype(int)
    ticks = qix[1:] - qix[:-1] + 1
    mt = np.max(ticks)
    px = np.tile(mp[qix[:-1]],(mt,1)).T
    px[:,-1] = mp[qix[1:]]
    for ix in np.arange(mt-2)+1 :
        p = mp[np.clip(qix[:-1]+ix,0,qix[1:])]
        px[:,ix]=p
    return np.vstack((px[:,0],np.max(px,axis=1),np.min(px,axis=1),px[:,-1])).T

def sample_extended(quote, butc, trade=None) :
    """
    quote is returned from get_quote_tickdata(), shape [nticks, 5], 
    where ts is first column, as int of 1000*utc
    butc also in milli-seconds, length [nbars + 1] vector, as the time of 
         start of the first bar to the end of last bar, i.e. len(butc) = bars + 1
    if trade is not none, the bid/ask diffs are removed with the trade quantity.
    return:
        bar: shape [n,5] 1 second bar with columns as col_name
        col_name: ['avg_bsz', 'avg_asz', 'avg_spd', 'tot_bsz_dif', 'tot_asz_diff']
    """

    # remove crossed or zero size quote
    ixg = np.nonzero(quote[:,3]-quote[:,1]>1e-10)[0]
    quote0 = quote[ixg,:]
    ixg = np.nonzero(np.abs(quote0[:,2]*quote0[:,4])>1e-10)[0]
    quote0 = quote0[ixg,:]
    ts, bp, bsz, ap, asz = quote0.T

    bdif = np.r_[0, bsz[1:] - bsz[:-1]]
    adif = np.r_[0, asz[1:] - asz[:-1]]

    # accounting for the trades, note this is the best
    # effort, as the time is in milliseconds
    if trade is not None:
        tst, pxt, szt = trade.copy().T
        tix = np.clip(np.searchsorted(ts, tst, side='left'),0, len(ts)-1).astype(int)

        # demand exact match on tix
        tix0 = np.nonzero(np.abs(ts[tix] - tst) > 0.1)[0]
        tix = np.delete(tix, tix0)
        pxt = np.delete(pxt, tix0)
        szt = np.delete(szt, tix0)

        bix = np.nonzero(np.abs(bp[tix]-pxt)<1e-10)[0]
        bdif[tix[bix]] += szt[bix]
        aix = np.nonzero(np.abs(ap[tix]-pxt)<1e-10)[0]
        adif[tix[aix]] += szt[aix]

    # nzix is index into the new ticks with a different mid px
    bnzix = np.nonzero( np.abs(bp[1:] - bp[:-1]) > 1e-10 )[0] + 1
    anzix = np.nonzero( np.abs(ap[1:] - ap[:-1]) > 1e-10 )[0] + 1

    # total bsz change and asz change in the bar
    bdif[bnzix] = 0
    adif[anzix] = 0

    qix = np.clip(np.searchsorted(ts,butc+0.1)-1,0,1e+10).astype(int)
    vdifc = np.cumsum(np.vstack((bdif, adif)), axis=1)
    vdifx = vdifc[:,qix[1:]] - vdifc[:,qix[:-1]]

    # bsz, asz, spd, need time weighted avg. 

    # time weighted avg is more complicated, it has to
    # observe boundary of a bar, even when there were no
    # changes upon them. The following adds "artificial"
    # ticks on bar time, so to allow such calclation

    spd = ap - bp

    # time weighted avg of book pressure and spread
    val = np.vstack((bsz, asz, spd))

    # add bar open/closing times to the ts, if not there yet
    tsix = np.nonzero(np.abs(butc - ts[qix]) > 1e-8)[0]
    butc_to_add = butc[tsix]
    qtix = qix[tsix]  # the value at butc[tsix]

    ts0 = np.r_[ts, butc_to_add]
    val = np.hstack((val, val[:, qtix]))
    # sort
    ix0 = np.argsort(ts0, kind='stable')
    ts0 = ts0[ix0]
    val = val[:,ix0]

    # redo the ts
    qix0 = np.clip(np.searchsorted(ts0.astype(int),butc.astype(int), side='right')-1,0,1e+10).astype(int)

    dt = ts0[1:] - ts0[:-1]
    valc = np.hstack((np.zeros((val.shape[0],1)), np.cumsum(val[:,:-1]*dt, axis=1)))
    vx = (valc[:,qix0[1:]] - valc[:,qix0[:-1]])/(butc[1:]-butc[:-1])

    col_name = ['avg_bsz', 'avg_asz', 'avg_spd', 'tot_bsz_dif', 'tot_asz_diff']
    return np.vstack((vx, vdifx)).T, col_name

def daily_mts_bar(trd0, quote0, barsec, start_utc, bar_cnt, extended_fields = False) :
    """
    write a daily bar with the trd and quote returned from quote and trade file
    barsec: bar period
    start_utc: starting time of the quote, bar_first_utc is start_utc plus barsec
    bar_cnt: the number of bars to be written

    return: 2-d array with each row being a bar line and columns as:
    BarTime: the utc of the bar generation time
    Open/High/Close/Low: OHCL
    TotalVolume: total trading volume
    LastPrice: the latest trade price seen so far
    LastPriceTime: the time of latest trade, with precision given
    VolumeImbalance: Buy trade size minus Sell trade size within this bar
    """

    if trd0 is None and quote0 is None :
        raise RuntimeError("trade and quote not found!")
    if quote0 is None :
        quote0 = _quote_from_trade(trd0)

    trd = trd0.copy()
    quote = quote0.copy()

    # remove all the crossed ticks
    crix = np.nonzero(quote[:,3]-quote[:,1]>1e-8)[0]
    if len(crix) > 0 :
        quote = quote[crix,:]

    # butc is the close time of each bar
    bar_first_utc = start_utc + barsec
    butc = np.arange(bar_cnt)*barsec+bar_first_utc

    # get OHLC
    bp = quote[:,1]
    ap = quote[:,3]
    ohlc = sample_OHLC((bp+ap)/2, quote[:,0], np.r_[start_utc,butc]*1000)

    # match trades with quote
    # find trade sign by matching trades with quotes

    # Since tickdata has 100 millisecond conflated, it is
    # therefore approximation. We look at the previous bpx/apx
    # of the matching time stamp of trade time onto quote.
    # if the trade price in doesn't touch any bpx/apx, look at
    # previous different bpx/apx to make desicion
    tix = np.clip(np.searchsorted(quote[:,0], trd[:,0], side='left') -2 ,0,quote.shape[0]-1).astype(int)

    # make sure the tpx equals either bpx/apx
    bpx = quote[tix, 1]
    apx = quote[tix, 3]
    tpx = trd[:,1]
    nzix = np.nonzero(np.min(np.abs(np.vstack((bpx,apx))-tpx),axis=0)>1e-8)[0]
    mpx = (quote[:,1] + quote[:,3])/2
    max_cnt = 100
    cnt = 0
    while len(nzix) > 0 and cnt < max_cnt:
        mpx0 = mpx[tix[nzix]]
        tix[nzix] = np.clip(tix[nzix]-1,0,1e+10)
        mpx1 = mpx[tix[nzix]]
        nzix_ix = np.nonzero(np.abs(mpx0-mpx1)>1e-8)[0]
        if len(nzix_ix) == 0 or np.max(nzix_ix) < 1:
            break
        nzix = nzix[nzix_ix]
        cnt += 1

    tpx = trd[:,1].copy()
    tsz = trd[:,2].copy()
    sellix = np.nonzero(np.abs(tpx-quote[tix, 1]) < np.abs(tpx-quote[tix,3]))[0]
    tsz[sellix]=tsz[sellix]*-1

    #tix matches latest trade on the bar second
    tix = np.clip(np.searchsorted(trd[:,0],butc*1000-0.001)-1,0,1e+10).astype(int)
    last_px = tpx[tix]
    last_px_time = trd[tix,0]
    volc = np.cumsum(np.abs(tsz))[tix]
    vbsc = np.cumsum(tsz)[tix]
    tvol = volc-np.r_[0, volc[:-1]]
    tvbs = vbsc-np.r_[0, vbsc[:-1]]

    ixname = {'BarTime':0, 'Open':1, 'High':2, 'Low':3, 'Close':4, 'TotalVolume':5, 'LastPx':6, 'LastPxTime':7, 'VolumImbalance':8}
    flds = np.vstack((butc, ohlc.T, tvol, last_px, last_px_time, tvbs)).T

    if extended_fields:
        #efld, colname = sample_extended(quote0.copy(), np.r_[start_utc,butc].astype(int)*1000)
        efld, colname = sample_extended(quote0.copy(), np.r_[start_utc,butc].astype(int)*1000, trade=trd0.copy())
        next_field = 9
        for i, col in enumerate(colname) :
            ixname[col] = next_field + i
        flds = np.hstack((flds, efld))

    return flds, ixname


def mts_bar_np(quote, trade, start_utc, end_utc, barsec, out_csv_fn, tick_size):
    # writes to mts repo with optional fields. quote, trade returned as 2d array
    #if trade is None and quote is None :
    #    raise RuntimeError("trade and quote not found!")
    if tick_size is None:
        tick_size = 0.0  #for parsing swipe level in bqd/aqd, optional
    else :
        tick_size=float(tick_size)

    if quote is None or len(quote)==0:
        raise "quote is empty, cannot parse"
        """
        TODO - 

        #if trade is None or len(trade)==0:
        if trade is None or len(trade)<2 or tick_size is None:
            raise RuntimeError("no quote updates, can't fake from trade")

        px = trade[:,1]
        sz = trade[:,2]
        n = len(px)
        quote = np.vstack((trade[:,0], px-tick_size, sz+1, px+tick_size, sz+1)).T
        """

    import td_parser_module_np as td
    td.td_parser_np(quote.copy(), trade.copy(), start_utc, end_utc, barsec, float(tick_size), out_csv_fn)

cme_month_code = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
class TickdataMap :
    def __init__(self, symbol_map_obj=None) :
        if symbol_map_obj is None:
            import symbol_map
            symbol_map_obj = symbol_map.SymbolMap()
        self.symbol_map = symbol_map_obj

    def get_tickdata_file(self, mts_symbol_no_N, contract_ym, day_ymd, add_prev_day=False) :

        # figure out the tickdata symbols
        tmap = self.symbol_map.get_tradable_map(day_ymd, mts_key = True, mts_symbols = [ mts_symbol_no_N ], add_prev_day=add_prev_day)
        found = False
        tdsym = None
        for k in tmap.keys() :
            if tmap[k]["symbol"] == mts_symbol_no_N :
                tdsym = tmap[k]["tickdata_id"]
                tzone = tmap[k]["tickdata_timezone"]
                pxmul = float(tmap[k]["tickdata_px_multiplier"])
                if tmap[k]["contract_month"] == contract_ym :
                    found = True
                    break
        if tdsym is None :
            raise RuntimeError(mts_symbol_no_N + " not defined for contract month " + contract_ym + " on the day of " + day_ymd)

        # there could be a contract that is N6 or N12, which is 
        # more than symbol_map's maxN, typically 2. In this case,
        # construct a tickdata_id by 
        # asset code + month_code + yy
        # timezone and pxmul are defined at asset level
        if not found:
            contract_month = int(contract_ym[-2:])
            contract_yy = contract_ym[2:4]
            tdsym = tdsym[:-3] + cme_month_code[contract_month-1] + contract_yy
        day_append = '_' + day_ymd[:4] + '_' + day_ymd[4:6] + '_' + day_ymd[6:8]
        qfile = tdsym + day_append + '_Q.asc.gz'
        tfile = tdsym + day_append + '.asc.gz'
        return qfile, tfile, tzone, pxmul

    def get_td_monthly_file(self, td_symbol, month_ym, tickdata_future_path, extract_to_path = None) :
        path = os.path.join(tickdata_future_path, td_symbol[0], td_symbol)
        tfname = month_ym[:4]+'_'+month_ym[4:]+'_'+td_symbol
        qfname = tfname + '_Q'

        tfile = os.path.join(path, tfname)
        qfile = os.path.join(path, 'QUOTES', qfname)

        if extract_to_path is not None :
            # do the extraction
            qpath = os.path.join(extract_to_path, 'quote')
            tpath = os.path.join(extract_to_path, 'trade')
            for p, f in zip ([qpath, tpath], [qfile, tfile]) :
                os.system('rm -fR ' + p + ' > /dev/null 2>&1')
                os.system('mkdir -p ' + p + ' > /dev/null 2>&1')
                os.system('unzip -o ' + f + ' -d ' + p)

        return qfile, tfile

    def get_td_by_mts(self, mts_symbol_no_N, day_ymd=None) :
        if day_ymd is None:
            tdu=mts_util.TradingDayUtil()
            day_ymd=tdu.get_trading_day(snap_forward=False)
        tmap = self.symbol_map.get_tradable_map(day_ymd, mts_key = True, mts_symbols = [ mts_symbol_no_N ], add_prev_day=True)
        for k in tmap.keys() :
            if tmap[k]["symbol"] == mts_symbol_no_N :
                tdsym = tmap[k]["tickdata_id"]
                venue = tmap[k]["venue"]
                return tdsym[:-3], venue
        raise RuntimeError(mts_symbol_no_N + " not recognized from symbol map ")

    def get_td_by_mts_month(self, mts_symbol_no_N, yyyymm):
        days = ['05','08','11','14']
        for d in days:
            try :
                return self.get_td_by_mts(mts_symbol_no_N, yyyymm+d)
            except :
                continue
        return self.get_td_by_mts(mts_symbol_no_N, yyyymm+'17');


    def get_quote_trade(self, mts_symbol_no_N, contract_ym, yyyymmdd, tickdata_future_path='/home/mts/run/repo/tickdata/FUT', out_csv_path=None, start_end_hhmmss=None, out_csv_utc=False):
        """
        extract tick-by-tick bid/ask and trade from tickdata raw files. such as used by backoffice to match/investigate fills
        Input:
            mts_symbol_no_N:      i.e. 'WTI'
            contract_ym:          yyyymm
            yyyymmdd:             the day to be extracted
            tickdata_future_path: the default path to tickdata raw files
            out_csv_path:         i.e. '/tmp/for_philippe', 
                                  the path to save the returned quote and trade to csv files
                                  file name is in 'WTI_202303_20221213_[quote|trade].csv'
                                  If provided, the csv files will output timestamp such as
                                  'yyyymmdd-hh:mm:ss.ffffff', unless out_csv_utc=True
            start_end_hhmmss:     i.e. ('08:25:00', '08:40:00'), 
                                  to extract only this time period of ticks/trades
        Return:
            quote:                shape [n,5] numpy array, with [utc, bid_px, bid_sz, ask_px, ask_sz]
            trade:                shape [n,3] numpy array, with [utc, trade_px, trade_sz]
        """
        td_symbol, mts_venue = self.get_td_by_mts(mts_symbol_no_N, yyyymmdd)
        ymd = datetime.datetime.now().strftime('%s.%f')
        tmp_path = '/tmp/td'+ymd
        os.system('rm -fR ' + tmp_path)
        os.system('mkdir -p ' + tmp_path + '/quote')
        os.system('mkdir -p ' + tmp_path + '/trade')

        #  extract it to a extract_to_path
        try:
            month_ym = yyyymmdd[:6]
            self.get_td_monthly_file(td_symbol, month_ym, tickdata_future_path, extract_to_path = tmp_path)
            qfile, tfile, tzone, pxmul = self.get_tickdata_file(mts_symbol_no_N, contract_ym, yyyymmdd, add_prev_day=False)
            qfile = os.path.join(tmp_path, 'quote', qfile)
            tfile = os.path.join(tmp_path, 'trade', tfile)
            print ('getting quotes from %s'%qfile)
            quote = get_quote_tickdata(qfile, time_zone=tzone, px_multiplier=pxmul)
            print ('getting trades from %s'%tfile)
            trade = get_trade_tickdata(tfile, time_zone=tzone, px_multiplier=pxmul,mts_venue=mts_venue)
            assert len(quote) > 0, 'no quotes found'
        except:
            traceback.print_exc()
            return
        finally:
            os.system('rm -fR ' + tmp_path)

        if start_end_hhmmss is not None:
            utc0 = int(datetime.datetime.strptime(yyyymmdd+start_end_hhmmss[0],'%Y%m%d%H:%M:%S').strftime('%s'))*1000
            utc1 = int(datetime.datetime.strptime(yyyymmdd+start_end_hhmmss[1],'%Y%m%d%H:%M:%S').strftime('%s'))*1000
            if utc0>utc1:
                utc0-=(3600*24*1000)
            ix0 = np.clip(np.searchsorted(quote[:,0], utc0)-1, 0, quote.shape[0]-1)
            ix1 = np.clip(np.searchsorted(quote[:,0], utc1)+1, 0, quote.shape[0]-1)
            quote = quote[ix0:ix1,:]
            if len(trade) > 0:
                ix0 = np.clip(np.searchsorted(trade[:,0], utc0)-1, 0, trade.shape[0]-1)
                ix1 = np.clip(np.searchsorted(trade[:,0], utc1)+1, 0, trade.shape[0]-1)
                trade = trade[ix0:ix1,:]

        # output to csv
        if out_csv_path is not None:
            for fn, d in zip (['quote','trade'], [quote, trade]):
                if len(d) == 0:
                    print('%s empty, not written'%(fn))
                    continue
                if not out_csv_utc:
                    dt = []
                    for t in d[:,0]:
                        dt.append(datetime.datetime.fromtimestamp(t/1000.0).strftime('%Y%m%d-%H:%M:%S.%f'))
                    ds = np.vstack(( np.array(dt) , d[:, 1:].T.astype('str') )).T
                else :
                    ds = d.astype('str')
                csv = os.path.join(out_csv_path, '%s_%s_%s_%s.csv'%(mts_symbol_no_N, contract_ym, yyyymmdd, fn))
                np.savetxt(csv, ds, delimiter=',', fmt='%s')

        return quote, trade

    def dump_tick_bar(self, mts_symbol, yyyymmdd, out_csv,\
                      dump_bar=False, barsec=1, \
                      tickdata_future_path='/home/mts/run/repo/tickdata/FUT'):

        sym,n=mts_symbol.split('_N')
        n=int(n)
        sym_n1=sym+'_N1'
        tinfo=self.symbol_map.get_tinfo(sym_n1, yyyymmdd,is_mts_symbol=True)
        mkt=tinfo['symbol']
        if n > self.symbol_map.max_N:
            contract_ym=self.symbol_map.get_contract_from_symbol(mkt, yyyymmdd, include_spread=False, extra_N=[n])[-1]
        else:
            contract_ym=tinfo['mts_contract'].split('_')[-1]
        start_hhmmss = tinfo['start_time']
        end_hhmmss = tinfo['end_time']
        utc_start=int(datetime.datetime.strptime(yyyymmdd+start_hhmmss,'%Y%m%d%H:%M:%S').strftime('%s'))
        utc_end=int(datetime.datetime.strptime(yyyymmdd+end_hhmmss,'%Y%m%d%H:%M:%S').strftime('%s'))
        if utc_start > utc_end:
            utc_start -= (3600*24)
        tick_size = tinfo['tick_size']
        quote_file = '%s_%s_%s_quote.csv'%(mkt,contract_ym, yyyymmdd)
        trade_file = '%s_%s_%s_trade.csv'%(mkt,contract_ym, yyyymmdd)
        out_csv_path='/tmp'
        self.get_quote_trade(mkt, contract_ym, yyyymmdd, out_csv_path=out_csv_path, start_end_hhmmss=(start_hhmmss, end_hhmmss), out_csv_utc=True)
        bin_path='bin/td_parser'
        import subprocess
        cmd=[bin_path, \
             os.path.join(out_csv_path,quote_file), \
             os.path.join(out_csv_path, trade_file),\
             str(utc_start), \
             str(utc_end), \
             str(barsec) if barsec is not None else '1', \
             'bar' if dump_bar else 'tick', \
             out_csv, \
             str(tick_size)]
        print('running %s'%(' '.join(cmd)))
        subprocess.run(cmd)

