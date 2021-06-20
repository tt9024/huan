import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo
import copy

# this is more or less same with KDB_hist.py, with 
# differences are 
# * hist file reader, need to figure out trade direction.  This will not be 
#   needed after June 2018 from l1 bar
# * trading hour is 24 hours instead of 23 hour (for ICE, need 17-18) 
# * barsec is 1 second instead of 5 second
# * more logic on finding hist files by names, they can overlap each other
# * more logic for filtering out out-of-order bars (repeated chunk of hist from IB)
# * retrieving by start day and end day, instead of a year
# * allow gzip of csv
# This should be the on-going, as the KDB bars be in repo for once and for all

def write_daily_bar(symbol, bar, bar_sec=5, is_front=True, last_close_px=None, get_missing=True):
    """
    bar: all bars from a hist file having the format of 
    [utc, utc_ltt, open_px, hi_px, lo_px, close_px, vwap, vol, vb, vs]
    These bars have the same contract. 
    The bar is in increasing utc, but may have gaps, or other invalid values
    The first day of that contract bar, due to prev_close_px unknown, it is
    usually covered by having the previous contract day. 
    Note there is a limitation that the start end time has to be on a whole hour
    i.e. cannot stop on 4:30, just make it 5, which will write some zero bars.
    However, it can handle 24 hour trading, i.e. start/end at 18:00, for fx venues.
    Note 2, the first bar of a day should be 1 bar_sec after the starting utc and
    the last bar of a day should be at the ending utc.

    if get_missing is set to true, then try to get the bar on a bad day

    Output: 
    array of daily_bar for each day covered in the bar (hist file)
    Each daily_bar have the following format: 
    [obs_utc, lr, trd_vol, vbs, lrhl, lrvwap, ltt, lpx]
    where: 
        obs_utc is the checking time stamp
        lr is the log return between this checking price and last checking price
           i.e. the lr of the previous bar that ended at this checking time (obs_utc)

      (May extend in the future)
    Note that the Trading Hours set to 24 for ICE hours
    In addition, it does the following:
    1. loop the close px to the first open px, 
    2. convert the price to lr, removing bars with maxlr more than 0.2 (CME circuit breaker)
    3. replace all inf/nan values with zero
    4. cacluate the ltt and lpx
    """
    import pandas as pd
    dt=datetime.datetime.fromtimestamp(bar[0,0])  # fromtimestamp is safe for getting local representation of utc

    start_hour, end_hour = l1.get_start_end_hour(symbol)
    TRADING_HOURS=end_hour-start_hour
    start_hour = start_hour % 24

    # get the initial day, last price
    day_start=dt.strftime('%Y%m%d')
    utc_s = int(l1.TradingDayIterator.local_ymd_to_utc(day_start, start_hour, 0, 0))
    if last_close_px is None :
        x=np.searchsorted(bar[1:,0], float(utc_s)-1e-6)

        # only take the last price within 5 minutes of utc_s
        if x+1 >= bar.shape[0] or bar[x+1, 0] - utc_s > 300 :
            if x+1>=bar.shape[0] :
                print 'no bars found after the start utc of ', day_start
            else :
                print 'start up utc (%d) more than 5 minutes later than start utc (%d) on %s'%(bar[x+1,0], utc_s, day_start)
                print 'initializing start up last_close_px deferred'
        else :
            if x == 0 :
                #last_close_px = bar[0, 2]
                #print 'last close price set as the first bar open px, this should use previous contract', datetime.datetime.fromtimestamp(bar[0,0]), datetime.datetime.fromtimestamp(bar[1,0])
                last_close_px = bar[0, 5]
                print 'lost last close price, set as the first bar close px'
            else :
                last_close_px=bar[x,5]
                print 'last close price set to close px of bar ', datetime.datetime.fromtimestamp(bar[x,0]), ' px: ', last_close_px

        print 'GOT last close px ', last_close_px
    else :
        print 'GIVEN last close price ', last_close_px

    day_end=datetime.datetime.fromtimestamp(bar[-1,0]).strftime('%Y%m%d')
    # deciding on the trading days
    if dt.hour > end_hour or (start_hour == end_hour and dt.hour >= end_hour) :
        # CME 17, ICE 18, 
        # the second rule is for 24 hour trading, note start/end has to be on a whole hour
        ti=l1.TradingDayIterator(day_start,adj_start=False)
        ti.next()
        trd_day_start=ti.yyyymmdd()
    else :
        trd_day_start=day_start
    trd_day_end=day_end
    print 'preparing bar from ', day_start, ' to ', day_end, ' , trading days: ', trd_day_start, trd_day_end

    ti=l1.TradingDayIterator(trd_day_start, adj_start=False) # day maybe a sunday
    day1=ti.yyyymmdd()  # first trading day
    barr=[]
    trade_days=[]
    col_arr=[]
    bad_trade_days=[]
    while day1 <= trd_day_end: 
        utc_e = int(l1.TradingDayIterator.local_ymd_to_utc(day1, end_hour,0,0))
        # get start backwards for starting on a Sunday
        utc_s = utc_e - TRADING_HOURS*3600  # LIMITATION:  start/stop has to be on a whole hour
        day=datetime.datetime.fromtimestamp(utc_s).strftime('%Y%m%d')

        i=np.searchsorted(bar[:, 0], float(utc_s)-1e-6)
        j=np.searchsorted(bar[:, 0], float(utc_e)-1e-6)
        bar0=bar[i:j,:]  # take the bars in between the first occurance of start_hour (or after) and the last occurance of end_hour or before

        print 'getting bar ', day+'-'+str(start_hour)+':00', day1+'-'+str(end_hour)+':00', ' , got ', j-i, 'bars'
        N = (utc_e-utc_s)/bar_sec  # but we still fill in each bar, so N should be fixed for a given symbol/venue pair

        # here N*0.90, is to account for some closing hours during half hour ib retrieval time
        # The problem with using histclient.exe to retrieve IB history data for ES is
        # set end time is 4:30pm, will retreve 3:45 to 4:15.  Because 4:15-4:30pm doesn't
        # have data.  This is only true for ES so far
        # another consideration is that IB Hist client usually won't be off too much, so 90% is 
        # a good threshold for missing/bad day
        bar_good = True
        if j-i<N*0.90 :
            if symbol in ['LE','HE'] or l1.venue_by_symbol(symbol)=='IDX' :
                bar_good = (j-i)>N*0.75
            elif not is_front :
                bar_good = (j-i)>N*0.5
            else :
                bar_good=False

        if not bar_good:
            print 'fewer bars for trading day %s: %d < %d * 0.9'%(day1, j-i,N)
            if day1 not in l1.bad_days and get_missing :
                # recurse with the current last price and get the updated last price
                print 'getting missing day %s'%(day1)
                from ibbar import get_missing_day
                fn = get_missing_day(symbol, [day1], bar_sec=bar_sec, is_front=is_front, reuse_exist_file=True)
                try :
                    _,_,b0=bar_by_file_ib(fn[0],symbol, start_day=day1, end_day=day1)
                except Exception as e :
                    print e
                    b0 = []

                if len(b0) > j-i :
                    print 'Getting more bars %d > %d on %s for %s, take it!'%(len(b0), j-i, day1, symbol)
                    barr0, trade_days0, col_arr0, bad_trade_days0, last_close_px0=write_daily_bar(symbol, b0, bar_sec=bar_sec, is_front=is_front, last_close_px=last_close_px, get_missing=False)
                    # taken as done
                    barr+=barr0
                    trade_days+=trade_days0
                    col_arr+=col_arr0
                    bad_trade_days+=bad_trade_days0
                    last_close_px=last_close_px0
                    ti.next()
                    day1=ti.yyyymmdd()
                    continue
                print 'Got %d bars on %s, had %d bars (%s), use previous!'%(len(b0), day1, j-i, symbol)

        if len(bar0) < 1 :
            print 'Bad Day! Too fewer bars in trading day %s: %d, should have %d '%(day1, j-i,N)
            bad_trade_days.append(day1)
        else :
            ix_utc=((bar0[:,0]-float(utc_s))/bar_sec+1e-9).astype(int) # lr(close_px-open_px) of a bar0 has bar_utc
            bar_utc=np.arange(utc_s+bar_sec, utc_e+bar_sec, bar_sec) # bar time will be time of close price, as if in prod

            if N != j-i :
                print 'fill missing for only ', j-i, ' bars (should be ', N, ')'
                bar1 = np.empty((N,bar0.shape[1]))
                bar1[:,0] = np.arange(utc_s, utc_e, bar_sec)
                # filling all missing for [utc, utc_ltt, open_px, hi_px, lo_px, close_px, vwap, vol, vb, vs]
                # fillforward for utc_ltt, close_px, vwap
                for col in [1, 5, 6] :
                    bar1[:, col] = np.nan
                    bar1[ix_utc, col] = bar0[:, col]
                    df=pd.DataFrame(bar1[:, col])
                    df.fillna(method='ffill',inplace=True)
                    df.fillna(method='bfill',inplace=True)
                # fill zero for vol, vb, bs
                for col in [7,8,9] :
                    bar1[:,col] = 0
                    bar1[ix_utc,col] = bar0[:, col]
                # copy value of close_px for open_px, hi_px, lo_px
                for col in [2,3,4] :
                    bar1[:, col] = bar1[:, 5]
                    bar1[ix_utc,col] = bar0[:, col]

            bar_arr=[]
            bar_arr.append(bar_utc.astype(float))

            # construct the log returns for each bar, fill in zeros for gap
            #lpx_open=np.log(bar0[:,2])
            if last_close_px is None :
                print 'setting last_close_px to ', bar0[0,2]
                last_close_px = bar0[0, 2]

            lpx_open=np.log(np.r_[last_close_px,bar0[:-1,5]])
            lpx_hi=np.log(bar0[:,3])
            lpx_lo=np.log(bar0[:,4])
            lpx_close=np.log(bar0[:,5])
            lpx_vwap=np.log(bar0[:,6])
            lr=lpx_close-lpx_open
            lr_hi=lpx_hi-lpx_open
            lr_lo=lpx_lo-lpx_open
            lr_vw=lpx_vwap-lpx_open

            # remove bars having abnormal return, i.e. circuit break for ES
            # with 9999 prices
            MaxLR=0.5
            if l1.is_holiday(day) or l1.is_fx_future(symbol) or l1.venue_by_symbol(symbol)=='FX':
                MaxLR=5
            ix1=np.nonzero(np.abs(lr)>=MaxLR)[0]
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_hi)>=MaxLR)[0])
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_lo)>=MaxLR)[0])
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_vw)>=MaxLR)[0])
            if len(ix1) > 0 :
                print 'MaxLR (', MaxLR, ') exceeded: ', len(ix1), ' ticks!'
                # removing one-by-one
                for ix1_ in ix1 :
                    dt = datetime.datetime.fromtimestamp(bar_utc[ix1_])
                    if not l1.is_pre_market_hour(symbol, dt) :
                        print 'warning: removing 1 tick lr/lo/hi/vw: ', lr[ix1_],lr_hi[ix1_],lr_lo[ix1_],lr_vw[ix1_]
                        lr[ix1_]=0
                        lr_hi[ix1_]=0
                        lr_lo[ix1_]=0
                        lr_vw[ix1_]=0
                    else :
                        print 'NOT removing 1 tick (pre_market=True: ', symbol, ', ', dt, ') lr/lo/hi/vw: ', lr[ix1_],lr_hi[ix1_],lr_lo[ix1_],lr_vw[ix1_]

            # the trade volumes for each bar, fill in zeros for gap
            vlm=bar0[:,7]
            vb=bar0[:,8]
            vs=np.abs(bar0[:,9])
            vbs=vb-vs

            for v0, vn in zip([lr,lr_hi,lr_lo,lr_vw,vlm,vbs], ['lr','lr_hi','lr_lo','lr_vw','vlm','vbs']) :
                nix=np.nonzero(np.isnan(v0))[0]
                nix=np.union1d(nix, np.nonzero(np.isinf(np.abs(v0)))[0])
                if len(nix) > 0 :
                    print 'warning: removing ', len(nix), ' nan/inf ticks for ', vn
                    v0[nix]=0
                b0=np.zeros(N).astype(float)
                b0[ix_utc]=v0
                bar_arr.append(b0.copy())
         
            # get the last trade time, this is needs to be
            ltt=np.empty(N)*np.nan
            ltt[ix_utc]=bar0[:,1]
            df=pd.DataFrame(ltt)
            df.fillna(method='ffill',inplace=True)
            if not np.isfinite(ltt[0]) :
                ptt=0 #no previous trading detectable
                if i > 0 : #make some effort here
                    ptt=bar[i-1,1]
                    if not np.isfinite(ptt) :
                        ptt=0
                df.fillna(ptt,inplace=True)
            bar_arr.append(ltt)

            # get the last price, as a debugging tool
            # close price
            lpx=np.empty(N)*np.nan
            lpx[ix_utc]=bar0[:,5]
            df=pd.DataFrame(lpx)
            df.fillna(method='ffill',inplace=True)
            if not np.isfinite(lpx[0]) :
                df.fillna(last_close_px,inplace=True)
            bar_arr.append(lpx)



            ba = np.array(bar_arr).T
            bt0=ba[:,0]
            lr0=ba[:,1]
            vl0=ba[:,5]
            vbs0=ba[:,6]
            # add a volatility measure here
            lrhl0=ba[:,2]-ba[:,3]
            vwap0=ba[:,4]
            ltt0=ba[:,7]
            lpx0=ba[:,8]
            barr.append(np.vstack((bt0,lr0,vl0,vbs0,lrhl0,vwap0,ltt0,lpx0)).T)
            last_close_px=lpx[-1]
            trade_days.append(day1)
            col_arr.append(repo.kdb_ib_col)

        ti.next()
        day1=ti.yyyymmdd()

    # filling in missing days if not included in the bad_trade_days
    bad_trade_days = []
    good_trade_days = []
    it = l1.TradingDayIterator(trd_day_start)
    while True :
        day = it.yyyymmdd()
        if day > trd_day_end :
            break
        if day not in trade_days :
            bad_trade_days.append(day)
        else :
            good_trade_days.append(day)
        it.next()

    print 'got bad trade days ', bad_trade_days
    return barr, good_trade_days, col_arr, bad_trade_days, last_close_px


def clip_idx(utc, symbol, start_day, end_day) :
    """
    Find the two indexes for trading day start_day to end_day (inclusive). 
    Return:
    ix0, ix1, so that utc[ix0:ix1] are all the included time instances
    """
    sh, eh = l1.get_start_end_hour(symbol)
    utc0 = l1.TradingDayIterator.local_ymd_to_utc(start_day, eh) - (eh-sh)*3600
    utc1 = l1.TradingDayIterator.local_ymd_to_utc(end_day, eh)
    ix0 = np.searchsorted(utc, utc0)
    ix1 = np.searchsorted(utc, utc1+0.1)
    return ix0, ix1

def get_gzip_filename(fn) :
    """
    get either fn or fn.gz depending on which one
    has non-zero size
    """
    if l1.get_file_size(fn)<10 :
        if fn[-3:] != '.gz' :
            return fn+'.gz'
        return fn[:-3]
    return fn

def get_trd (fntd) :
    """
    need to work with both .csv.gz or .csv format
    All hist file is .gz.  So have to add .gz 
    if not yet
    """
    try :
        fn=get_gzip_filename(fntd)
        print 'reading trd ', fntd, fn
        os.system('chmod u+rw ' + fn)
        bar_trd=np.genfromtxt(fn, delimiter=',',usecols=[0,1,2,3,4,5,6,7]) #,dtype=[('utc','i8'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8'),('vol','i8'),('cnt','i8'),('wap','<f8')])
    except :
        print 'no trade for ', fn
        bar_trd = []
    return bar_trd

def get_qt(fnqt) :
    try :
        fn=get_gzip_filename(fnqt)
        print 'reading quote ', fnqt, fn
        os.system('chmod u+rw ' + fn)
        bar_qt=np.genfromtxt(fn, delimiter=',',usecols=[0,1,2,3,4]) #, dtype=[('utc','i8'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8')])
    except :
        import traceback
        traceback.print_exc()
        print 'no quotes for ', fn
        bar_qt = []
    return bar_qt

def bar_by_file_ib_idx(fn) :
    """
    Read only the trd, mainly for IDX, or other cases 
    _trd.csv expected to exist for the given fn
    return 
    bar_qt and bar_trd, as if returned by future
    where bar_qt is the first 5 columes of bar_trd
    """
    if fn[-3:] == '.gz' :
        fn = fn[:-3]
    if fn[-4:] == '.csv' :
        fn = fn[:-8]
    fntrd=fn+'_trd.csv'
    bar_trd=get_trd(fntrd)
    ix=l1.get_inc_idx(bar_trd[:,0])
    assert len(ix) > 3, 'too few bars found at ' + fn

    bar_trd=bar_trd[ix,:]
    ts = bar_trd[:, 0]
    vwap = bar_trd[:, 4]
    v = np.zeros((3, len(ts)))
    bar=np.vstack((ts,ts,bar_trd[:,1:5].T,vwap,v)).T
    return bar

def bar_by_file_ib_qtonly(fn) :
    """ 
    Read only the quotes, mainly for FX, or other cases 
    (such as some sparse back contract) when there is no trade.
    _qt.csv expected to exist for the given fn
    return same format as bar_by_file_ib, adding vwap as close px
    and v as all 0
    """

    if fn[-3:] == '.gz' :
        fn = fn[:-3]
    if fn[-4:] == '.csv' :
        fn = fn[:-7]
    fnqt=fn+'_qt.csv'
    bar_qt=get_qt(fnqt)
    nqt =  bar_qt.shape[0]
    assert nqt > 3,  'too few bars found at ' + fn

    qix=l1.get_inc_idx(bar_qt[:,0])
    bar_qt = bar_qt[qix,:]
    qts=bar_qt[:,0]
    assert len(np.nonzero(qts[1:]-qts[:-1]<0)[0]) == 0, 'quote time stamp goes back'

    ts = bar_qt[:, 0]
    vwap = bar_qt[:, 4]
    v = np.zeros((3, len(ts)))
    bar=np.vstack((bar_qt[:,0],ts,bar_qt[:,1:5].T,vwap,v)).T
    return bar

def bar_by_file_ib(fn, symbol, start_day='19980101', end_day='20990101', bar_qt=None,bar_trd=None) :
    """ 
    _qt.csv and _trd.csv are expected to exist for the given fn
    return :
    bar_qt[:,0], utc_ltt, bar_qt[:,1:5].T, vwap, vol, vb, vs

    return bar_qt, bar_trd, bar
    where
    bar {utc, utcltt, open, high, low, close,vwap,vol,vb,vs}
    """
    bid_ask_spd = get_future_spread(symbol)
    is_fx = l1.venue_by_symbol(symbol) == 'FX'
    is_idx = l1.venue_by_symbol(symbol) == 'IDX'
    is_etf = l1.venue_by_symbol(symbol) == 'ETF'

    if is_idx :
        print 'Getting IDX quotes!'
        b0 = bar_by_file_ib_idx(fn)
        if len(b0) > 0 :
            ix0, ix1 = clip_idx(b0[:, 0], symbol, start_day, end_day)
            return [], [], b0[ix0:ix1, :]
        return [], [], b0
    else :
        if fn[-3:] == '.gz' :
            fn = fn[:-3]
        if fn[-4:] == '.csv' :
            fn = fn[:-7]
        fnqt=fn+'_qt.csv'
        fntd=fn+'_trd.csv'

    if bar_trd is None or len(bar_trd) == 0 :
        has_trd = l1.get_file_size(fntd)>100 or l1.get_file_size(fntd+'.gz')>100
        if has_trd :
            bar_trd = get_trd(fntd)
        if is_fx or not has_trd or len(bar_trd) < 1:
            print 'Getting Quote Only!'
            b0 = bar_by_file_ib_qtonly(fn)
            if len(b0) > 0 :
                ix0, ix1 = clip_idx(b0[:, 0], symbol, start_day, end_day)
                return [], [], b0[ix0:ix1, :]
            return [], [], b0
    if bar_qt is None or len(bar_qt) == 0 :
        bar_qt = get_qt(fnqt)

    # use quote as ref
    nqt =  bar_qt.shape[0]
    assert nqt > 3,  'too few bars found at ' + fn

    # make sure the time stamps strictly increasing
    qix=l1.get_inc_idx(bar_qt[:,0])
    tix=l1.get_inc_idx(bar_trd[:,0])
    bar_qt = bar_qt[qix,:]
    bar_trd = bar_trd[tix,:]

    qts=bar_qt[:,0]
    tts=bar_trd[:,0]
    assert len(np.nonzero(qts[1:]-qts[:-1]<0)[0]) == 0, 'quote time stamp goes back'
    assert len(np.nonzero(tts[1:]-tts[:-1]<0)[0]) == 0, 'trade time stamp goes back'

    # deal with length difference
    # some times the file content has more days than the file name suggests. 
    # such as ZNH8_20180201_20180302_1S_qt.csv has days from 2/1 to 3/19. 
    # but the _trd.csv only has to 3/2 as file name suggests.  
    # In this case, take the shorter one and ensure the days
    # checked for gaps in between for missing days
    # Only exception is when there is only one day, then 

    while True :
        if len(qts) < 10 :
            return [],[],[]
        #dtq0 = datetime.datetime.fromtimestamp(qts[0])
        #dtt0 = datetime.datetime.fromtimestamp(tts[0])
        #dtq1 = datetime.datetime.fromtimestamp(qts[-1])
        #dtt1 = datetime.datetime.fromtimestamp(tts[-1])

        dtq0 = l1.trd_day(qts[0])
        dtt0 = l1.trd_day(tts[0])
        dtq1 = l1.trd_day(qts[-1])
        dtt1 = l1.trd_day(tts[-1])
        print 'Got Quote: ',  dtq0, ' to ', dtq1, ' Trade: ', dtt0, ' to ', dtt1

        #if (qts[-1] != tts[-1]) :
        if dtq1 != dtt1 :
            # only handles where ending date is different
            print '!!! Quote/Trade ending date mismatch!!!'
            ts = min(qts[-1], tts[-1])
            if qts[-1] > ts :
                ix = np.nonzero(qts>ts)[0]
                qts = qts[:ix[0]]
                bar_qt = bar_qt[:ix[0], :]
            else :
                ix = np.nonzero(tts>ts)[0]
                tts = tts[:ix[0]]
                bar_trd = bar_trd[:ix[0], :]
        #elif (qts[0] != tts[0]) :
        elif dtq0 != dtt0 :
            print '!!! Quote/Trade date starting mismatch!!!'
            ts = max(qts[0], tts[0])
            if qts[0] < ts :
                ix = np.nonzero(qts<ts)[0]
                six = ix[-1]+1
                qts = qts[six:]
                bar_qt = bar_qt[six:, :]
            else :
                ix = np.nonzero(tts<ts)[0]
                six = ix[-1]+1
                tts = tts[six:]
                bar_trd = bar_trd[six:, :]
        else :
            break

    tix=np.clip(np.searchsorted(tts,qts),0,len(tts)-1)
    # they should be the same, otherwise, patch the different ones
    ix0=np.nonzero(tts[tix]-qts!=0)[0]
    if len(ix0) != 0 : 
        print len(ix0), ' bars mismatch!'
    ts=bar_trd[tix,:]

    # This should be tts
    #ts[tix[ix0],5]=0
    #ts[tix[ix0],6]=0
    #ts[tix[ix0],7]=bar_qt[ix0,4].copy()
    ts[ix0,5]=0
    ts[ix0,6]=0
    ts[ix0,7]=bar_qt[ix0,4].copy()

    import pandas as pd
    vwap=ts[:,7].copy()
    vol=ts[:,5].copy()
    vb=vol.copy()
    vs=vol.copy()
    if is_etf :
        print 'adjust ETF size '
        # IB's ETF volume in LOTS, i.e. 250 = 2 LOTS
        vol=vol*100+50
        vb=vb*100+50
        vs=vs*100+50

    utc_ltt=ts[:,0]
    if len(ix0) > 0 : 
        utc_ltt[ix0]=np.nan
        df=pd.DataFrame(utc_ltt)
        df.fillna(method='ffill',inplace=True)

    """ 
    # for those bar without price movements, calculate the volume by avg trade price 
    ixe=np.nonzero(bar_qt[:,1]-bar_qt[:,4]==0)[0]
    #pdb.set_trace()
    vb[ixe]=np.clip((ts[ixe,7]-(bar_qt[ixe,4]-bid_ask_spd/2))/bid_ask_spd*ts[ixe,5],0,1e+10)
    vs[ixe]=ts[ixe,5]-vb[ixe]

    ixg=np.nonzero(bar_qt[:,1]-bar_qt[:,4]<0)[0]
    vs[ixg]=0
    ixl=np.nonzero(bar_qt[:,1]-bar_qt[:,4]>0)[0]
    vb[ixl]=0
    """
    spd=bid_ask_spd*np.clip(np.sqrt((bar_qt[:,2]-bar_qt[:,3])/bid_ask_spd),1,2)
    mid=(bar_qt[:,2]+bar_qt[:,3])/2
    #mid=np.mean(bar_qt[:,1:5], axis=1)

    vb=np.clip((vwap-(mid-spd/2))/spd,0,1)*vol
    vs=vol-vb
    bar=np.vstack((bar_qt[:,0],utc_ltt,bar_qt[:,1:5].T,vwap,vol,vb,vs)).T
    ix0, ix1 = clip_idx(bar[:,0], symbol, start_day, end_day)
    return bar_qt, bar_trd, bar[ix0:ix1, :]

def get_future_spread(symbol) :
    tick, mul = l1.asset_info(symbol)
    return tick


def fn_from_dates(symbol, sday, eday, is_front_future) :
    try :
        is_fx = l1.venue_by_symbol(symbol) == 'FX'
        is_etf = l1.venue_by_symbol(symbol) == 'ETF'
        is_idx = l1.venue_by_symbol(symbol) == 'IDX'
    except :
        print 'Unknow symbol %s'%(symbol)
        raise ValueError('Unknown symbol ' + symbol)

    from ibbar import read_cfg
    hist_path = read_cfg('HistPath')
    sym0 = symbol
    if symbol in l1.RicMap.keys() :
        sym0 = l1.RicMap[symbol]
    if is_etf :
        fqt=glob.glob(hist_path+'/ETF/'+sym0+'_[12]*_qt.csv*')
    elif is_fx :
        fqt=glob.glob(hist_path+'/FX/'+sym0+'_[12]*_qt.csv*')
    elif is_idx :
        fqt=glob.glob(hist_path+'/IDX/'+sym0+'_[12]*_trd.csv*')
    else :
        if is_front_future :
            fqt=glob.glob(hist_path+'/'+symbol+'/'+sym0+'*_[12]*_qt.csv*')
        else :
            fqt=glob.glob(hist_path+'/'+symbol+'/nc/'+sym0+'??_[12]*_qt.csv*')

    ds=[]
    de=[]
    fn=[]
    for f in fqt :
        if os.stat(f).st_size < 500 :
            print '\t\t\t ***** ', f, ' is too small, ignored'
            continue
        ds0=f.split('/')[-1].split('_')[1]
        de0=f.split('/')[-1].split('_')[2].split('.')[0]
        # check for inclusion
        if ds0 > eday or de0 < sday :
            continue
        ds.append(ds0)
        de.append(de0)
        fn.append(f)

    # sort the list in the increasing order of starting dates
    # this will make the merging easier by using append
    # in case of total inclusion, then the rule will be
    # "overwrite", instead of "append"
    # append means add only the new content to the existing daily bar
    # overwrite means add all the content to the existing daily bar, overwirte if overlap
    # merge means to only apply to daily bars of any days that doesn't exists.
    ix=np.argsort(ds)
    dss=np.array(ds)[ix]
    des=np.array(de)[ix]
    fns=np.array(fn)[ix]

    while True :
        if len(fns) == 0 :
            print 'ERROR! Nothing found for %s from %s to %s (front %s), search path %s'%(symbol, sday, eday, str(is_front_future), hist_path)
            break

        # remove the files that are contained
        desi=des.astype(int)
        ix = np.nonzero(desi[1:]-desi[:-1]<=0)[0]
        if len(ix) > 0 :
            print fns[ix+1], ' contained by ', fns[ix], ', removed, if needed, consider load and overwrite repo'
            fns = np.delete(fns, ix+1)
            des = np.delete(des, ix+1)
            dss = np.delete(dss, ix+1)
        else :
            break

    return fns, is_fx, is_etf, is_idx

def get_barsec_from_file(f) :
    fa = f.split('_')
    for f0 in fa :
        if f0[-1] == 'S' :
            try :
                return int(f0[:-1])
            except :
                pass
    raise ValueError('no barsec detected in file %s'%(f))

def get_days_from_file(f) :
    s0 = f.split('_')
    return s0[1], s0[2]

def gen_daily_bar_ib(symbol, sday, eday, default_barsec, dbar_repo, is_front_future=True, get_missing=True, barsec_from_file=True, overwrite_dbar=False, EarliestMissingDay='19980101') :
    """
    generate IB dily bars from sday to eday.
    It is intended to be used to add too the daily bar repo manually
    NOTE 1: bar_sec from file name is used to read/write the day. 
            default_barsec given is taken as a default when getting missing days. 
            When barsec_from_file is not True, the bar_sec from file name
            is checked against the default bar_sec given and raises on mismatch.
    NOTE 2: barsec_from_file being False enforces all day's barsec has to agree with default_barsec
    NOTE 3: The flexibility on barsec from file name is to entertain IB's rule for
            half year history on 1S, 1 year history on 30S bar, etc, enforced 
            differently on asset classes. Inconsistencies on weekly operations
    NOTE 4: if overwrite_dbar is True, then the existing repo content on the day will be deleted before
            ingestion
    """

    fn, is_fx, is_etf, is_idx = fn_from_dates(symbol, sday, eday, is_front_future)
    spread = get_future_spread(symbol)
    print 'Got ', len(fn), ' files: ', fn, ' spread: ', spread

    num_col=8 # adding spd vol, last_trd_time, last_close_pxa
    tda=[]
    tda_bad=[]
    for f in fn :
        bar_sec=get_barsec_from_file(f)
        if bar_sec != default_barsec :
            if not barsec_from_file :
                raise ValueError('Bar second mismatch for file %s with barsec %d'%(f,default_barsec))
            else :
                print 'Set barsec to ', bar_sec, ' from ', default_barsec

        try :
            d0, d1 = get_days_from_file(f)
            _,_,b=bar_by_file_ib(f,symbol, start_day=max(sday,d0), end_day=min(eday,d1))
        except KeyboardInterrupt as e :
            raise e
        except Exception as e :
            print e
            b = []
        if len(b) > 0 :
            ba, td, col, bad_days, last_px = write_daily_bar(symbol, b,bar_sec=bar_sec, is_front=is_front_future, get_missing=get_missing)
            if overwrite_dbar :
                for td0 in td :
                    # assuming days in increasing order: don't delete days
                    # just written 

                    # don't delete if the barsec does not match
                    # Because I don't want the IB_hist (barsec=1) to 
                    # overwrite the KDB days, which has barsec=5
                    if td0 not in tda :
                        dbar_repo.remove_day(td0, match_barsec=bar_sec)
            dbar_repo.update(ba, td, col, bar_sec)
            tda+=td
            tda_bad+=bad_days
        else :
            print '!!! No bars was read from ', f

    tda = list(set(tda))
    tda.sort()
    tda_bad = list(set(tda_bad))
    tda_bad.sort()

    # The following gets the days that are either in tda nor in 
    # tda_bad, i.e. some missing days not found in any history files
    # todo - this shouldn't happen and most probably due to the 
    # half day/holidays, should remove
    if len(tda) == 0 :
        print 'NOTHING found! Not getting any missing days!'
    # in case there are some entirely missed days
    elif get_missing :
        # there could be some duplication in files, so
        # so some files has bad days but otherwise already in other files.
        missday=[]
        d0 = max(sday, EarliestMissingDay)
        print ' checking on the missing days from %s to %s'%(d0, eday)

        diter = l1.TradingDayIterator(d0)
        while d0 <= eday :
            if d0 not in tda and d0 not in tda_bad and  d0 not in l1.bad_days :
                missday.append(d0)
            diter.next()
            d0=diter.yyyymmdd()

        if len(missday) > 0 :
            print 'getting the missing days ', missday
            from ibbar import get_missing_day
            fn = []
            mdays = []
            for md in missday :
                fn0 = get_missing_day(symbol, [md], bar_sec, is_front_future, reuse_exist_file=True)
                if len(fn0) > 0 :
                    fn += fn0
                    mdays.append(md)
                else :
                    print 'nothing on missing day: ', md

            for f, d in zip(fn, mdays) :
                try :
                    _,_,b=bar_by_file_ib(f,symbol, start_day=d, end_day=d)
                    if len(b) > 0 :
                        print 'got ', len(b), ' bars from ', f, ' on missing day', d
                        ba, td, col, bad_days, lastpx0 = write_daily_bar(symbol, b,bar_sec=bar_sec, is_front=is_front_future, get_missing=False)
                        tda+=td
                        tda_bad+=bad_days
                        if len(td) > 0 :
                            if overwrite_dbar :
                                for td0 in td :
                                    dbar_repo.remove_day(td0,match_barsec=bar_sec)
                            dbar_repo.update(ba, td, col, bar_sec)
                        else :
                            print 'no trading day is found from ', f, ' on missing day ', d
                    else :
                        print 'nothing got for missing day: ', d
                except KeyboardInterrupt as e :
                    raise e
                except :
                    traceback.print_exc()
                    print 'problem processing file ', f

    tda.sort()
    tda_bad.sort()
    print 'Done! Bad Days: ', tda_bad
    return tda, tda_bad

def clear_hist_dir(hist_path) :
    """
    gzip all the csv file.
    If both csv and gz file exists, delete csv if gz is larger, otherwise, gzip -f csv
    if csv file is 0 size, then leave it and delete the gz file
    """
    os.system('for f in `find '+hist_path+' -name *.csv -print` ; do \
                  if [ `stat -c %s $f` -eq 0 ] ; then \
                     echo "zero size $f " ; \
                     rm -f "$f.gz" > /dev/null 2>&1 ; \
                  else if [ ! -f "$f.gz" ] || [ `stat -c %s $f` -ge `stat -c %s "$f.gz"` ] ; then \
                       echo "gzip $f" ; \
                       gzip -f $f ; \
                     else \
                       echo "remove csv file $f!" ; \
                       rm -f $f > /dev/null 2>&1 ; \
                     fi ; \
                  fi ; \
               done')

def all_sym(exclude_list) :
    import ibbar
    sym = ibbar.sym_priority_list
    #sym += l1.ven_sym_map['FX'] 
    #sym += ibbar.ib_sym_etf 
    sym += ibbar.ib_sym_idx
    
    for s in exclude_list :
        sym.remove(s)
    return sym

def ingest_all_symb(sday, eday, repo_path=None, get_missing=True, sym_list = None, future_inclusion=['front','back'], sym_list_exclude=[], overwrite_dbar=True, EarliestMissingDay='20180201') :
    """
    This will go to IB historical data, usually in /cygdrive/e/ib/kisco,
    read all the symbols defined by sym_list and update the repo at repo_path,
    which is usually at /cygdrive/e/research/kdb
    if sym_list is None, then it will include all the symbol collected by ibbar.
    future_inclusion defaults to include both front and back.  It is included
        for ibbar's ingestion, when only front contract is in hist, but
        the back contracts haven't been retrieved yet.
    NOTE: ETF and FX symbols are not affected by future_inclusion
    """
    import ibbar
    if repo_path is None :
        repo_path=ibbar.read_cfg('RepoPath')
    fut_sym = ibbar.sym_priority_list
    fx_sym = l1.ven_sym_map['FX'] 
    etf_sym = ibbar.ib_sym_etf 
    fut_sym2 = ibbar.sym_priority_list_l1_next 
    idx_sym = ibbar.ib_sym_idx
    if sym_list is None :
        sym_list = fut_sym + fx_sym + etf_sym + idx_sym
    
    for sym in sym_list :
        if sym in sym_list_exclude :
            continue
        print 'ingesting ', sym
        if sym in fut_sym and 'front' in future_inclusion:
            barsec = 1
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path, create=True)
            gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, is_front_future=True, get_missing = get_missing, overwrite_dbar=overwrite_dbar,EarliestMissingDay=EarliestMissingDay)
        elif sym in fx_sym:
            barsec = 5
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path, create=True)
            gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, get_missing = get_missing, overwrite_dbar=overwrite_dbar, EarliestMissingDay=EarliestMissingDay)
        elif sym in etf_sym or sym in idx_sym:
            barsec = 1
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path, create=True)
            gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, get_missing = get_missing, overwrite_dbar=overwrite_dbar)
        if sym in fut_sym2 and 'back' in future_inclusion:
            barsec = 1
            repo_path_nc = repo.nc_repo_path(repo_path) # repo path of next contract
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path_nc, create=True)
            gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, is_front_future=False, get_missing = get_missing, overwrite_dbar=overwrite_dbar, EarliestMissingDay=EarliestMissingDay)

def weekly_get_ingest(start_end_days=None, repo_path='repo_hist', rsync_dir_list=None) :
    """
    This is supposed to be run on IB machine at EoD Friday.
    It first gets all the history of this week, and then ingest
    into a hist_repo.  The need for ingestion, is to correct
    on any missing data.  After this run, the files in the hist dir
    is copied to data machine
    """
    import ibbar
    if start_end_days is None:
        cdt = datetime.datetime.now()
        if cdt.weekday() != 4 :
            raise ValueError('sday not set while running on non-friday!')
        eday = cdt.strftime('%Y%m%d')
        tdi = l1.TradingDayIterator(eday)
        sday=tdi.prev_n_trade_day(5).yyyymmdd()
    else :
        sday, eday = start_end_days

    print 'Got start/end day: ', sday, eday
    ibbar.weekly_get_hist(sday, eday)

    #No need to do this, unless the previous get failed. But
    #then it should be tried again.
    #ingest_all_symb(sday, eday, repo_path=repo_path)
    hist_path = ibbar.read_cfg('HistPath')
    if rsync_dir_list is not None :
        for rsync_dir in rsync_dir_list :
            if len(rsync_dir) > 0 :
                os.system('rsync -avz ' + hist_path + '/ ' + rsync_dir)

