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

        if N != j-i and not fill_missing:
            print 'bad day, need to retrieve the hist file again!'
            if day1 not in bad_trade_days :
                bad_trade_days.append(day1)
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

    return barr, trade_days, col_arr, bad_trade_days

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
    return l1.SymbolTicks[symbol]


def fn_from_dates(symbol, sday, eday, is_front_future, is_fx) :
    if is_fx :
        fqt=glob.glob('hist/FX/'+symbol+'_[12]*_qt.csv*')
    else :
        if is_front_future :
            fqt=glob.glob('hist/'+symbol+'/'+symbol+'*_[12]*_qt.csv*')
        else :
            fqt=glob.glob('hist/'+symbol+'/'+symbol+'/nc/*_[12]*_qt.csv*')
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

def gen_daily_bar_ib(symbol, sday, eday, bar_sec, check_only=False, dbar_repo=None, is_front_future=True, is_fx = False, get_missing=True) :
    """
    generate IB dily bars from sday to eday.
    It is intended to be used to add too the daily bar repo manually
    """
    fn = fn_from_dates(symbol, sday, eday, is_front_future, is_fx)
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
        _,_,b=bar_by_file_ib(f,spread)
        ba, td, col, bad_days = write_daily_bar(symbol, b,bar_sec=bar_sec)
        baa+=ba
        tda+=td
        cola+=col
        tda_bad+=bad_days

    if dbar_repo is not None :
        dbar_repo.update(baa, tda, cola, bar_sec)

    tda_bad = list(set(tda_bad))
    tda_bad.sort()

    print 'Done!'

    if get_missing and dbar_repo is not None :
        print 'getting the missing days ', tda_bad
        import ibbar
        reload(ibbar)
        fn = ibbar.get_missing_day(symbol, tda_bad, bar_sec, is_front_future, is_fx)
        for f in fn :
            _,_,b=bar_by_file_ib(f,spread)
            ba, td, col, bad_days = write_daily_bar(symbol, b,bar_sec=bar_sec,fill_missing=True)
            if len(ba) > 0 :
                dbar_repo.update(ba, td, col, bar_sec)

    return baa, tda, cola, tda_bad


def l1_bar(symbol, bar_path) :
    b = np.genfromtxt(bar_path, delimiter=',', use_cols=[0,1,2,3,4,5,6])
    # I need to get the row idx for each day for the columes of vbs and ism
    # which one is better?
    # I could use hist's trade for model, and l1/tick for execution
    pass

"""
def gen_bar(symbol, year_s=1998, year_e=2018, check_only=False, ext_fields=True) :
    ba=[]
    years=np.arange(year_s, year_e+1)
    for y in years :
        try :
            barlr=gen_bar0(symbol,str(y),check_only=check_only,ext_fields=ext_fields)
            if len(barlr) > 0 :
                ba.append(barlr)
        except :
            traceback.print_exc()
            print 'problem getting ', y, ', continue...'

    if check_only :
        return
    fn=symbol+'_bar_'+str(year_s)+'_'+str(year_e)
    if ext_fields :
        fn+='_ext'
    np.savez_compressed(fn,bar=ba,years=years)
"""

