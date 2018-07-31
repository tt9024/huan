import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import glob

import l1
import l1_reader as l1r

def write_daily_bar(bar,bar_sec=5,last_close_px=None) :
    import pandas as pd
    dt=datetime.datetime.fromtimestamp(bar[0,0])

    # get the initial day, last price
    day_start=dt.strftime('%Y%m%d')
    utc_s = int(l1.TradingDayIterator.local_ymd_to_utc(day_start, 18, 0, 0))
    if last_close_px is None :
        x=np.searchsorted(bar[1:,0], float(utc_s-3600+bar_sec))
        last_close_px=bar[x,2]
        print 'last close price set to previous close at ', datetime.datetime.fromtimestamp(bar[x,0]), ' px: ', last_close_px
    else :
        print 'last close price set to ', last_close_px

    day_end=datetime.datetime.fromtimestamp(bar[-1,0]).strftime('%Y%m%d')
    # deciding on the trading days
    if dt.hour > 17 :
        ti=l1.TradingDayIterator(day_start,adj_start=False)
        ti.next()
        trd_day_start=ti.yyyymmdd()
    else :
        trd_day_start=day_start
    trd_day_end=day_end
    print 'preparing bar from ', day_start, ' to ', day_end, ' , trading days: ', trd_day_start, trd_day_end

    ti=l1.TradingDayIterator(day_start, adj_start=False)
    day=ti.yyyymmdd()  # day is the start_day
    barr=[]
    TRADING_HOURS=23
    while day < day_end:
        ti.next()
        day1=ti.yyyymmdd()
        utc_e = int(l1.TradingDayIterator.local_ymd_to_utc(day1, 17,0,0))

        # get start backwards for starting on a Sunday
        utc_s = utc_e - TRADING_HOURS*3600
        day=datetime.datetime.fromtimestamp(utc_s).strftime('%Y%m%d')

        i=np.searchsorted(bar[:, 0], float(utc_s)-1e-6)
        j=np.searchsorted(bar[:, 0], float(utc_e)-1e-6)
        bar0=bar[i:j,:]  # take the bars in between the first occurance of 18:00:00 (or after) and the last occurance of 17:00:00 or before

        N = (utc_e-utc_s)/bar_sec  # but we still fill in each bar
        ix_utc=((bar0[:,0]-float(utc_s))/bar_sec+1e-9).astype(int)
        bar_utc=np.arange(utc_s+bar_sec, utc_e+bar_sec, bar_sec) # bar time will be time of close price, as if in prod

        print 'getting bar ', day+'-18:00', day1+'-17:00', ' , got ', j-i, 'bars'
        # start to construct bar
        if j<=i :
            print ' NO bars found, skipping'
        else :
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
            lpx=np.empty(N)*np.nan
            lpx[ix_utc]=bar0[:,5]
            df=pd.DataFrame(lpx)
            df.fillna(method='ffill',inplace=True)
            if not np.isfinite(lpx[0]) :
                df.fillna(last_close_px,inplace=True)
            bar_arr.append(lpx)
            barr.append(np.array(bar_arr).T.copy())
            last_close_px=lpx[-1]

        day=day1

    return np.vstack(barr), trd_day_start, trd_day_end

def bar_by_file_ib(fn,bid_ask_spd,bar_qt=None,bar_trd=None) :
    """ 
    _qt.csv and _trd.csv are expected to exist for the given fn
    """

    if fn[-3:] == '.gz' :
        fn = fn[:-3]
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


def fn_from_dates(symbol, sday, eday) :
    bfn=glob.glob('hist/'+symbol+'/'+symbol+'*_[12]*_qt.csv*')
    ds=[]
    de=[]
    fn=[]
    for f in bfn :
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
    # remove the files that are containeda
    desi=des.astype(int)
    ix = np.nonzero(desi[1:]-desi[:-1]<=0)[0]
    if len(ix) > 0 :
        print fns[ix+1], ' contained by ', fns[ix], ', removed'
    fns = np.delete(fns, ix+1)
    return fns

def gen_daily_bar_by_file(symbol, sday, eday, bar_sec, check_only=False) :
    """
    generate IB dily bars from sday to eday.
    It is intended to be used to add too the daily bar repo manually
    """
    fn = fn_from_dates(symbol, sday, eday)
    spread = get_future_spread(symbol)
    print 'Got ', len(fn), ' files: ', fn, ' spread: ', spread

    if check_only :
        return
    num_col=8 # adding spd vol, last_trd_time, last_close_pxa
    bar_lr = []
    if len(fn) == 0 :
        return bar_lr
    for f in fn :
        _,_,b=bar_by_file_ib(f,spread)
        ba, sd, ed = write_daily_bar(b,bar_sec=bar_sec)
        bt=ba[:,0]
        lr=ba[:,1]
        vl=ba[:,5]
        vbs=ba[:,6]
        # add a volatility measure here
        lrhl=ba[:,2]-ba[:,3]
        vwap=ba[:,4]
        ltt=ba[:,7]
        lpx=ba[:,8]
        bar_lr.append(np.vstack((bt,lr,vl,vbs,lrhl,vwap,ltt,lpx)).T)

    return bar_lr

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

if __name__ == '__main__' :
    import sys
    symbol_list= sys.argv[1:-2]
    sday=sys.argv[-2]
    eday=sys.argv[-1]
    get_future_bar_fix(symbol_list, sday, eday)

