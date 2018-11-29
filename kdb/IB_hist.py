import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo

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

def write_daily_bar(symbol, bar,bar_sec=5,last_close_px=None, fill_missing=False) :
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

    if fill_missing is set to true, such as a half day, then do zero filling

    Output: 
    array of daily_bar for each day covered in the bar (hist file)
    Each daily_bar have the following format: 
    [obs_utc, lr, lr_hi, lr_low, lr_vwap, volume, vbs, ltt, lpx] 
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
        if x == 0 :
            last_close_px = bar[0, 2]
            print 'last close price set as the first bar open px, this should use previous contract'
        else :
            last_close_px=bar[x,5]
            print 'last close price set to previous close at ', datetime.datetime.fromtimestamp(bar[x,0]), ' px: ', last_close_px
    else :
        print 'last close price set to ', last_close_px

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

    ti=l1.TradingDayIterator(day_start, adj_start=False) # day maybe a sunday
    day=ti.yyyymmdd()  # day is the start_day
    barr=[]
    trade_days=[]
    col_arr=[]
    bad_trade_days=[]
    while day < day_end:  # day is the prevous night of the trading day
        ti.next()
        day1=ti.yyyymmdd()
        utc_e = int(l1.TradingDayIterator.local_ymd_to_utc(day1, end_hour,0,0))

        # get start backwards for starting on a Sunday
        utc_s = utc_e - TRADING_HOURS*3600  # LIMITATION:  start/stop has to be on a whole hour
        day=datetime.datetime.fromtimestamp(utc_s).strftime('%Y%m%d')

        i=np.searchsorted(bar[:, 0], float(utc_s)-1e-6)
        j=np.searchsorted(bar[:, 0], float(utc_e)-1e-6)
        bar0=bar[i:j,:]  # take the bars in between the first occurance of start_hour (or after) and the last occurance of end_hour or before

        print 'getting bar ', day+'-'+str(start_hour)+':00', day1+'-'+str(end_hour)+':00', ' , got ', j-i, 'bars'
        N = (utc_e-utc_s)/bar_sec  # but we still fill in each bar, so N should be fixed for a given symbol/venue pair

        # here N*0.95, is to account for some closing hours during half hour ib retrieval time
        # The problem with using histclient.exe to retrieve IB history data for ES is
        # set end time is 4:30pm, will retreve 3:45 to 4:15.  Because 4:15-4:30pm doesn't
        # have data.  This is only true for ES so far
        # another consideration is that IB Hist client usually won't be off too much, so 95% is 
        # a good threshold for missing/bad day
        if N*0.95 > j-i and not fill_missing and day1 not in l1.bad_days :
            print 'bad day, need to retrieve the hist file again!', N, j-i
            if day1 not in bad_trade_days :
                bad_trade_days.append(day1)
        elif day1 in l1.bad_days and j-i < 3600/bar_sec :
            print 'trade day ', day, ' in holidays ', l1.bad_days, ' and too few updates, ', j-i, ', igored'
        elif day1 in trade_days :
            print 'trade day ', day1, ' already in, skipping'
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
            MaxLR=0.2
            ix1=np.nonzero(np.abs(lr)>=MaxLR)[0]
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_hi)>=MaxLR)[0])
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_lo)>=MaxLR)[0])
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_vw)>=MaxLR)[0])
            if len(ix1) > 0 :
                print 'warning: removing ', len(ix1), 'ticks exceed MaxLR (lr/lo/hi/vw) ', zip(lr[ix1],lr_hi[ix1],lr_lo[ix1],lr_vw[ix1])
                lr[ix1]=0
                lr_hi[ix1]=0
                lr_lo[ix1]=0
                lr_vw[ix1]=0

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

        day=day1

    # filling in missing days if not included in the bad_trade_days
    bad_trade_days = []
    it = l1.TradingDayIterator(trd_day_start)
    while True :
        day = it.yyyymmdd()
        if day > trd_day_end :
            break
        if day not in trade_days :
            bad_trade_days.append(day)
        it.next()
    
    print 'got bad trade days ', bad_trade_days
    return barr, trade_days, col_arr, bad_trade_days


def bar_by_file_ib_qtonly(fn) :
    """ 
    Mainly for FX, there is no trade, quote only in 5 second bar
    _qt.csv expected to exist for the given fn
    """

    if fn[-3:] == '.gz' :
        fn = fn[:-3]
    if fn[-4:] == '.csv' :
        fn = fn[:-7]
    fnqt=fn+'_qt.csv'
    print 'readig ', fn
    # getting gzip if necessary
    gz = False
    f = fnqt
    if l1.get_file_size(f) == 0 :
        if l1.get_file_size(f+'.gz') > 0 :
            print 'got gziped file ', f+'.gz', ' unzip it'
            os.system('gunzip ' + f + '.gz')
            gz = True
        else :
            raise ValueError('file not found: ' + f)

    import pandas as pd
    bar_qt=np.genfromtxt(fnqt, delimiter=',',usecols=[0,1,2,3,4]) #, dtype=[('utc','i8'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8')])
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
    if gz : 
        print 'gzip ' + f
        os.system('gzip ' + f)
    return bar

def bar_by_file_ib(fn,bid_ask_spd,bar_qt=None,bar_trd=None) :
    """ 
    _qt.csv and _trd.csv are expected to exist for the given fn
    """

    if fn[-3:] == '.gz' :
        fn = fn[:-3]
    if fn[-4:] == '.csv' :
        fn = fn[:-7]
    fnqt=fn+'_qt.csv'
    fntd=fn+'_trd.csv'
    print 'readig ', fn
    # getting gzip if necessary
    gz = False
    for b, f in zip([bar_qt, bar_trd],[fnqt, fntd]) :
        if b is not None :
            continue
        if l1.get_file_size(f) == 0 :
            if l1.get_file_size(f+'.gz') > 0 :
                print 'got gziped file ', f+'.gz', ' unzip it'
                os.system('gunzip ' + f + '.gz')
                gz = True
            else :
                raise ValueError('file not found: ' + f)

    import pandas as pd
    if bar_qt is None :
        bar_qt=np.genfromtxt(fnqt, delimiter=',',usecols=[0,1,2,3,4]) #, dtype=[('utc','i8'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8')])
    if bar_trd is None :
        bar_trd=np.genfromtxt(fntd, delimiter=',',usecols=[0,1,2,3,4,5,6,7]) #,dtype=[('utc','i8'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8'),('vol','i8'),('cnt','i8'),('wap','<f8')])

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

    while True :
        if len(qts) < 10 :
            return [],[],[]
        dtq0 = datetime.datetime.fromtimestamp(qts[0])
        dtt0 = datetime.datetime.fromtimestamp(tts[0])
        dtq1 = datetime.datetime.fromtimestamp(qts[-1])
        dtt1 = datetime.datetime.fromtimestamp(tts[-1])
        print 'Got Quote: ',  dtq0, ' to ', dtq1, ' Trade: ', dtt0, ' to ', dtt1

        if (qts[-1] != tts[-1]) :
            print '!!! Quote/Trade ending mismatch!!!'
            ts = min(qts[-1], tts[-1])
            if qts[-1] > ts :
                ix = np.nonzero(qts>ts)[0]
                qts = qts[:ix[0]]
                bar_qt = bar_qt[:ix[0], :]
            else :
                ix = np.nonzero(tts>ts)[0]
                tts = tts[:ix[0]]
                bar_trd = bar_trd[:ix[0], :]
        elif (qts[0] != tts[0]) :
            print '!!! Quote/Trade starting mismatch!!!'
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

    tix=np.searchsorted(tts,qts)
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

    vwap=ts[:,7].copy()
    vol=ts[:,5].copy()
    vb=vol.copy()
    vs=vol.copy()
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
    if gz : 
        for f in [fnqt, fntd] :
            print 'gzip ' + f
            os.system('gzip ' + f)

    return bar_qt, bar_trd, bar 


def get_future_spread(symbol) :
    tick, mul = l1.asset_info(symbol)
    return tick


def fn_from_dates(symbol, sday, eday, is_front_future, is_fx, is_etf) :
    from ibbar import read_cfg
    hist_path = read_cfg('HistPath')
    sym0 = symbol
    if symbol in l1.RicMap.keys() :
        sym0 = l1.RicMap[symbol]
    if is_etf :
        fqt=glob.glob(hist_path+'/ETF/'+sym0+'_[12]*_qt.csv*')
    elif is_fx :
        fqt=glob.glob(hist_path+'/FX/'+sym0+'_[12]*_qt.csv*')
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
    # remove the files that are contained
    desi=des.astype(int)
    ix = np.nonzero(desi[1:]-desi[:-1]<=0)[0]
    if len(ix) > 0 :
        print fns[ix+1], ' contained by ', fns[ix], ', removed, if needed, consider load and overwrite repo'
        fns = np.delete(fns, ix+1)
    return fns

def gen_daily_bar_ib(symbol, sday, eday, bar_sec, check_only=False, dbar_repo=None, is_front_future=True, is_fx = False, is_etf = False, get_missing=True) :
    """
    generate IB dily bars from sday to eday.
    It is intended to be used to add too the daily bar repo manually
    """
    fn = fn_from_dates(symbol, sday, eday, is_front_future, is_fx, is_etf)
    spread = get_future_spread(symbol)
    print 'Got ', len(fn), ' files: ', fn, ' spread: ', spread

    if check_only :
        return
    num_col=8 # adding spd vol, last_trd_time, last_close_pxa
    baa=[]
    tda=[]
    cola=[]
    tda_bad=[]
    for f in fn :
        if is_fx :
            b = bar_by_file_ib_qtonly(f)
        else :
            _,_,b=bar_by_file_ib(f,spread)
        if len(b) > 0 :
            ba, td, col, bad_days = write_daily_bar(symbol, b,bar_sec=bar_sec)
            baa+=ba
            tda+=td
            cola+=col
            tda_bad+=bad_days
        else :
            print '!!! No bars was read from ', f


    if dbar_repo is not None :
        dbar_repo.update(baa, tda, cola, bar_sec)

    tda_bad = list(set(tda_bad))
    tda_bad.sort()

    print 'Done!' 

    if len(tda_bad) > 0 and get_missing and dbar_repo is not None :
        # there could be some duplication in files, so
        # so some files has bad days but otherwise already in other files.
        if len(tda) > 0 :
            print ' checking on the bad days ', tda_bad, ' out of trading days ', tda
            missday=[]
            d0 = min(tda[0], tda_bad[0])
            d1 = max(tda[-1], tda_bad[-1])
            diter = l1.TradingDayIterator(d0)
            d0 = diter.yyyymmdd()
            while d0 <= d1 :
                if d0 not in tda and d0 not in l1.bad_days :
                    missday.append(d0)
                diter.next()
                d0=diter.yyyymmdd()
            tda_bad = missday

        print 'getting the missing days ', tda_bad
        from ibbar import get_missing_day
        fn = get_missing_day(symbol, tda_bad, bar_sec, is_front_future, is_fx)
        for f in fn :
            try :
                _,_,b=bar_by_file_ib(f,spread)
                if len(b) > 0 :
                    ba, td, col, bad_days = write_daily_bar(symbol, b,bar_sec=bar_sec,fill_missing=True)
                    if len(ba) > 0 :
                        dbar_repo.update(ba, td, col, bar_sec)
            except :
                traceback.print_exc()
                print 'problem processing file ', f

    return baa, tda, cola, tda_bad


def ingest_all_symb(sday, eday, repo_path=None, get_missing=False, sym_list = None, future_inclusion=['front','back'], sym_list_exclude=[]) :
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
    if sym_list is None :
        sym_list = fut_sym + fx_sym + etf_sym
    
    for sym in sym_list :
        if sym in sym_list_exclude :
            continue
        print 'ingesting ', sym
        if sym in fut_sym and 'front' in future_inclusion:
            barsec = 1
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path, create=True)
            baa, tda, cola, tda_bad = gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, is_front_future=True, is_fx=False, get_missing = get_missing)
        elif sym in fx_sym:
            barsec = 5
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path, create=True)
            gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, is_fx=True, get_missing = get_missing)

        elif sym in etf_sym :
            barsec = 1
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path, create=True)
            gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, is_etf=True, get_missing = get_missing)

        elif sym in fut_sym2 and 'back' in future_inclusion:
            barsec = 1
            repo_path_nc = repo.nc_repo_path(repo_path) # repo path of next contract
            dbar = repo.RepoDailyBar(sym, repo_path = repo_path_nc, create=True)
            gen_daily_bar_ib(sym, sday, eday, barsec, dbar_repo = dbar, is_front_future=False, get_missing = get_missing)


