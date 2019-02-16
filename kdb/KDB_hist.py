import numpy as np
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo
import os

def bar_by_file(fn, symbol) :
    if symbol in kdb_future_symbols :
        return bar_by_file_future(fn, skip_header=5, csv_tz = future_csv_tz[symbol])
    elif symbol in kdb_etf_symbols :
        # all New York Time
        return bar_by_file_etf(fn, skip_header=5)
    elif symbol in kdb_fx_symbols :
        return bar_by_file_fx(fn, skip_header=5, csv_tz = 'Europe/London')

    raise ValueError('unknown symbol ' + symbol)

def bar_by_file_fx(fn, skip_header, csv_tz) :
    """
    date,ric,timeStart,closeBid,closeAsk,avgSpread,cntTick
    2014.02.06,AUD=,00:00:00.000,0.8926,0.8927,2.61415,3

    Return:
    [utc, utc_lt, open, high, low, close, vwap, vol, bvol, svol]

    Note 1 :
    No trade for FX, so vwap, vol, bvol, svol all zeros
    utc_lt is always sod-1
    
    Note 2:
    no open/hi/low prices, all equals to close price

    Note 3:
    the time stamp are taken as London time (Reuters)
    """
    bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[0,2,3,4], skip_header=skip_header,dtype=[('day','|S12'),('bar_start','|S14'),('closebid','<f8'),('closeask', '<f8')])
    bar=[]
    dt=datetime.datetime.strptime(bar_raw[0]['day']+'.'+bar_raw[0]['bar_start'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
    utc_sod=float(l1.TradingDayIterator.dt_to_utc(dt, dt_tz=csv_tz))
    for b in bar_raw :
        dt=datetime.datetime.strptime(b['day']+'.'+b['bar_start'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc=float(l1.TradingDayIterator.dt_to_utc(dt, dt_tz=csv_tz))
        utc_lt = utc_sod-1

        px = (b['closebid'] + b['closeask'])/2
        bar0=[utc, utc_lt, px, px, px, px, px, 0, 0, 0]
        bar.append(bar0)

    bar = np.array(bar)
    open_px_col=2
    ix=np.nonzero(np.isfinite(bar[:,open_px_col]))[0]
    bar=bar[ix, :]
    ix=np.argsort(bar[:, 0])
    return bar[ix, :]

def bar_by_file_future(fn, skip_header, csv_tz) :
    """
    volume can include block/manual trades that are not included in bvol and svol. 
    The side of those trades are lost
    date,ric,timeStart,lastTradeTickTime,open,high,low,close,avgPrice,vwap,volume,buyvol,sellvol
    2017.11.22,CLF8,00:00:00.000,00:00:04.039,57.71,57.72,57.71,57.71,57.7107,57.7104,282,51,-44

    Return:
    [utc, utc_lt, open, high, low, close, vwap, vol, bvol, svol]
    """
    bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[0,2,3,4,5,6,7,9,10,11,12], skip_header=skip_header,dtype=[('day','|S12'),('bar_start','|S14'),('last_trade','|S14'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8'),('vwap','<f8'),('volume','i8'),('bvol','i8'),('svol','i8')])
    bar=[]
    for b in bar_raw :
        dt=datetime.datetime.strptime(b['day']+'.'+b['bar_start'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc=float(l1.TradingDayIterator.dt_to_utc(dt, dt_tz=csv_tz))
        dt_lt=datetime.datetime.strptime(b['day']+'.'+b['last_trade'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc_lt=float(l1.TradingDayIterator.dt_to_utc(dt, dt_tz=csv_tz))+float(b['last_trade'].split('.')[1])/1000.0

        bar0=[utc, utc_lt, b['open'],b['high'],b['low'],b['close'],b['vwap'],b['volume'],b['bvol'],b['svol']]
        bar.append(bar0)

    bar = np.array(bar)
    open_px_col=2
    ix=np.nonzero(np.isfinite(bar[:,open_px_col]))[0]
    bar=bar[ix, :]
    ix=np.argsort(bar[:, 0])
    return bar[ix, :]

def bar_by_file_future_trd_day(symbol, day1, day2, kdb_path, fc=None, nc=False) :
    """
    kdb_path is the path to the kdb files, usually the trd file is kdb_path/symbol/trd/
    The fc is the expected front contract, given by the kdb bar file.  In this case
    the fc doesn't change for the period of day1 to day2, inclusive. 
    None to get from l1.FC(), in this case FC changes with the days.
    nc: if True, then getting the next contract of the fc 

    return: tsarr, fcarr, darr
       tsarr is an array of np.array([utc, trd_px, signed_trd_vol])
       fcarr is an array of front contract in tsarr
       darr is an array of day on the trd file names.  The day 
            is NOT a trading day, it's a calendar day, i.e. from 0 to 23:59:59
            of that day
    """
    ts = []
    fcarr=[]
    darr = []

    it = l1.TradingDayIterator(day1)
    while day1 <= day2 :
        if nc:
            bfn_contract = l1.FC_next(symbol, day1)
        else :
            fc0 = l1.FC(symbol, day1)
            bfn_contract = fc
            if fc is None :
                bfn_contract = fc0
            else :
                if fc0 != bfn_contract :
                    print 'WARNING!  bar file contract is not front contract on ', day1, l1.FC(symbol, day1), bfn_contract

        fn0 = kdb_path + '/'+symbol+'/*'+day1+'*'
        fn = glob.glob(fn0)
        trdfn = None
        if len(fn) == 1 :
            if not nc :
                # just take it
                trdfn = fn[0]
                if bfn_contract not in trdfn :
                    print 'WARNING! got only 1 file (%s) with fc(%s)'%(trdfn, bfn_contract)
            else :
                # nc not found
                if bfn_contract in fn[0] :
                    trdfn = fn[0]
                else :
                    print 'nc not found: ', day1, ' con: ', bfn_contract, ' only file: ', fn[0]

        elif len(fn) > 1:
            szarr=[]
            for f0 in fn :
                if bfn_contract in f0 :
                    trdfn = f0 
                    break
                szarr.append(os.stat(f0).st_size)

            if trdfn is None :
                ix = np.argsort(szarr)
                if not nc :
                    # tka the larger one
                    trdfn = fn[ix[-1]]
                else :
                    trdfn = fn[ix[0]]

        if trdfn is not None :
            ts.append(bar_by_file_future_trd(trdfn))
            fcarr.append(trdfn.split('_')[0])
            darr.append(day1)

        else :
            print 'no trd file found on ', day1, ' symbol ', symbol

        it.next()
        day1=it.yyyymmdd()
    return ts, fcarr, darr

def bar_by_file_future_trd(fn) :
    """
    date,ric,time,gmt_offset,price,volume,tic_dir
    2009.10.28,CLZ9,00:00:00.224,-4,,58,
    2009.10.28,CLZ9,00:00:14.890,-4,79.4,1,^
    2009.10.28,CLZ9,00:00:14.890,-4,79.39,1,v
    
    where gmt_offset is w.r.t ny local time.
    price can be none, a implied trade or block trade
    tic_dir: ^ buy v sell

    Return:
    [utc, px, bsvol]
    """
    bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[0,2,3,4,5,6],skip_header=5,dtype=[('day','|S12'),('time','|S16'),('gmtoff','i8'),('px','<f8'),('sz','i8'),('dir','|S2')])
    ts=[]
    for i, b in enumerate(bar_raw) :
        if b['dir'] == '' :
            continue
        dt = datetime.datetime.strptime(b['day'] + ' ' +b['time'], '%Y.%m.%d %H:%M:%S.%f')
        utc = l1.TradingDayIterator.local_dt_to_utc(dt, micro_fraction=True)
        """
        #dt = datetime.datetime.strftime('%Y.%.%d %H:%M:%S.%f', b['day'] + ' ' +b['time'])
        gmtoff=b['gmtoff']
        dd = datetime.timedelta(0, abs(gmtoff)*3600)
        utc = l1.TradingDayIterator.local_dt_to_utc(dt + np.sign(gmtoff)*dd)
        """
        if b['dir']=='v' :
            bs = -1
        elif b['dir'] == '^' :
            bs = 1
        else :
            raise ValueError('%s line %d got unknown direction in trd file %s'%(fn, i, b['dir']))
        ts.append([utc, b['px'], b['sz']*bs])

    # merge trades with same milli, px and direction
    ts = np.array(ts)

    ux = ts[:,0]*np.sign(ts[:,2])
    sz=np.cumsum(ts[:,2])
    ix = np.r_[np.nonzero(np.abs(ux[1:]-ux[:-1]) > 1e-13)[0], len(ux)-1]
    ts=ts[ix, :2]
    ts=np.vstack((ts.T, sz[ix]-np.r_[0, sz[ix[:-1]]])).T
    return ts

def bar_by_file_etf(fn, skip_header=5) :
    """
    date(0),ric,timeStart(2),exchange_id,country_code,mic,lastTradeTickTime(6),open(7),high,low,close,avgPrice,vwap(12),minSize,maxSize,avgSize,avgLogSize,medianSize,volume(18),dolvol,cntChangePrice,cntTrade,cntUpticks,cntDownticks,sigma,buyvol(24),sellvol(25),buydolvol,selldolvol,cntBuy,cntSell,sideSigma,priceImpr,maxPriceImpr,dolimb,midvol,gmt_offset,lastQuoteTickTime,openBid(37),openAsk,highBid,highAsk,lowBid,lowAsk,closeBid,closeAsk,avgBid,avgAsk,minBidSize,minAskSize,maxBidSize,maxAskSize,avgBidSize,avgAskSize,avgLogBidSize,avgLogAskSize,avgSpread,cntChangeBid,cntChangeAsk,cntTick
    2016.11.10,XLF,04:09:55.000,,,,04:09:57.092,20.99,20.99,20.98,20.98,20.985,20.985,100,100,100,4.60517,100,200,4197,1,2,0,1,0.005,0,-200,0,-4197,0,2,0,-1,200,-4197,0,-5,04:09:57.092,20.98,21.99,20.98,21.99,20.94,21.99,20.94,21.99,20.96,21.99,1,20,21,20,11,20,1.52226,2.99573,479.632,1,0,2

    Return:
    [utc, utc_lt, open, high, low, close, vwap, vol, bvol, svol]

    Note 1:
    All fields before gmt (-5) is empty if there were no trade in this bar period
    k
    NOte 2: 
    volume may be larger than bvol + svol, same as Future

    Note 3:
    The dividant and splits are not taken care of.  For example, XLF has a split on 20160916, the price in reuters kdb 
    has a constant offset before 20160916, but seems to be good after that.  The overnight LR on 20160920 in this case is 15%, 
    which should be removed by outlier. 

    Note 4:
    Eary days, i.e. 1998 - 2010, is quite noisy, especially the open tick.  outlier and cleaning is required.
    """

    bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[0,2,6,12,18,24,25, 38,39,40,41,42,43,44,45], skip_header=skip_header,\
            dtype=[('day','|S12'),('bar_start','|S14'),('last_trade','|S14'),\
                   ('vwap','<f8'),('volume','i8'),('bvol','i8'),('svol','i8'),\
                   ('openbid','<f8'), ('openask','<f8'), ('highbid','<f8'), ('highask','<f8'),\
                   ('lowbid','<f8'), ('lowask','<f8'), ('closebid','<f8'), ('closeask','<f8')])
    bar=[]
    # getting the first time stamp as starting utc_lt
    dt=datetime.datetime.strptime(bar_raw[0]['day']+'.'+bar_raw[0]['bar_start'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
    utc_lt=float(l1.TradingDayIterator.local_dt_to_utc(dt)) - 5
    for b in bar_raw :
        dt=datetime.datetime.strptime(b['day']+'.'+b['bar_start'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc=float(l1.TradingDayIterator.local_dt_to_utc(dt))
        openpx = (b['openbid'] + b['openask'])/2
        highpx = (b['highbid'] + b['highask'])/2
        lowpx  = (b['lowbid']  + b['lowask' ])/2
        closepx= (b['closebid']+ b['closeask'])/2
        if b['vwap'] == np.nan or b['volume'] == np.nan or b['volume'] is None or b['volume'] < 0:
            b['volume'] = 0
            b['vwap'] = closepx
            b['bvol'] = 0
            b['svol'] = 0
        else :
            dt_lt=datetime.datetime.strptime(b['day']+'.'+b['last_trade'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
            utc_lt=float(l1.TradingDayIterator.local_dt_to_utc(dt))+float(b['last_trade'].split('.')[1])/1000.0
        bar0=[utc, utc_lt, openpx, highpx, lowpx, closepx, b['vwap'],b['volume'],b['bvol'],b['svol']]
        bar.append(bar0)

    bar = np.array(bar)
    vol_col=7
    ix=np.nonzero(np.isfinite(bar[:,vol_col]))[0]
    bar=bar[ix, :]
    ix=np.argsort(bar[:, 0])
    return bar[ix, :]

def gen_bar_trd(symbol, sday, eday, dbar, kdb_path='./kdb', bar_sec=1, nc=False) :
    """
    getting from the ts [utc, px, signed_vol]
    output format bt, lr, vl, vbs, lrhl, vwap, ltt, lpx

    if dbar is not None, update dbar, otherwise 
    return :
        bar_arr, days, col_arr

    The problems:
    1. deal with missing days
       still need the bar data to fill in those!
       1 trd file will cause the missing of two days. Only
       way is to use 5S bars to fill, but how do you
       down sample them?
       Just ditch the two days 

       How many missing days do I have by the way?
    2. deal with rolls
       if the contract goes to a different contract
       over, then just apply a constant offset on
       the price. 
    """

    start_hour, end_hour = l1.get_start_end_hour(symbol)
    tds = 1
    it = l1.TradingDayIterator(sday)
    pday = it.yyyymmdd()
    tday = it.yyyymmdd()
    if start_hour < 0 :
        tds = 2
        it.prev()
        pday=it.yyyymmdd()
        it.next()
        start_hour = start_hour % 24

    da = [] ; ta = [] ;  fa = []
    lastpx=0
    prev_con=''
    while tday <= eday :
        # get for trading day
        if len(da) == 2 :
            day0 = tday
            da=[da[-1]] ; ta=[ta[-1]] ; fa=[fa[-1]]
            tds0=1
        else :
            # either initial or previously broken 2-day or 1 day 
            day0 = pday
            tds0=tds
            da=[] ; ta=[] ; fa=[]

        tsarr, fcarr, darr = bar_by_file_future_trd_day(symbol, day0, tday, kdb_path=kdb_path, nc=nc)
        if len(darr) != tds0 :
            print 'error getting trading day ', tday, ' found only ', darr, fcarr
            da=[] ; fa=[] ; fa=[]
            lastpx=0
            prev_con=''
        else :
            # this is the good case
            da+=darr ; fa+=fcarr ; ta+=tsarr
            # figure out the utc
            sutc = it.local_ymd_to_utc(pday,h_ofst=start_hour)
            eutc = it.local_ymd_to_utc(tday,h_ofst=end_hour)
            ix0=np.searchsorted(ta[0][:,0],sutc)
            ix1=np.searchsorted(ta[-1][:,0],eutc+1e-6)
            len0 = len(ta[0][:,0])
            len1 = len(ta[-1][:,0])
            if lastpx == 0 :
                lastpx=ta[0][max(ix0-1,0),1]

            if ix0 >= len0 :
                # first half day?
                print 'starting ix as the last index of first of ', da
                ix0 = len0-1
            if ix1 == 0 :
                # second half day?
                print 'ending ix as the first index of second of ', da

            # need to check rolls and set the lastpx
            if tds == 2 :
                if fa[0] != fa[1] 
                    px_diff = ta[1][0,1]-ta[0][-1,1]
                    ta[0][:,1]+=px_diff
                    lastpx+=px_diff
                bar = np.vstack((ta[0][ix0:,:], ta[1][:ix1,:]))
            else :
                if fa[0] != prev_con :
                    lastpx=ta[0][0,1]
                bar = np.array(ta[0][ix0:ix1,:])

            # have everything, need to get to
            # output format bt, lr, vl, vbs, lrhl, vwap, ltt, lpx
            bt=np.arange(sutc+bar_sec,eutc+bar_sec,bar_sec)

            tts=np.r_[sutc,bar[:,0]]
            pts=np.r_[bar[0,1],bar[:,1]]
            vts=np.r_[0,bar[:,2]]
            pvts=np.abs(vts)*pts

            pxix=np.clip(np.searchsorted(tts[1:],bt+1e-6),0,len(tts)-1)
            lpx=pts[pxix]
            lr = np.log(np.r_[lastpx,lpx])
            lr=lr[1:]-lr[:-1]

            # tricky way to get index right on volumes
            btdc=np.r_[0,np.cumsum(vts)[pxix]]
            vbs=btdc[1:]-btdc[:-1]
            btdc=np.r_[0,np.cumsum(np.abs(vts))[pxix]]
            vol=btdc[1:]-btdc[:-1]

            # even tickier way to get vwap/ltt right
            ixg=np.nonzero(vol)[0]
            btdc=np.r_[0, np.cumsum(pvts)[pxix]]
            vwap=lpx.copy()  #when there is no vol
            vwap[ixg]=(btdc[1:]-btdc[:-1])[ixg]/vol[ixg]
            ltt=np.zeros(len(bt))
            ltt[ixg]=tts[pxix][ixg]
            repo.fwd_bck_fill(ltt, v=0)

            # give up, ignore the lrhl for trd bars
            lrhl=np.zeros(len(bt))

            b=np.vstack((bt,lr,vol,vbs,lrhl,vwap,ltt,lpx)).T
            d=tday
            c=repo.kdb_ib_col
            dbar.remove_day(d)
            dbar.update([b],[d],[c],bar_sec)
            lastpx=lpx[-1]
            prev_con=fa[-1]

        pday=tday
        it.next()
        tday=it.yyyymmdd()


def write_daily_bar(symbol,bar,bar_sec=5,old_cl_repo=None) :
    """
    input format:
         utc, utc_lt, open, high, low, close, vwap, vol, bvol, svol
         Where : 
             utc is the start of the bar

    output format: 
         bt,lr,vl,vbs,lrhl,vwap,ltt,lpx
         Where :
             bt is the end of the bar time, as lr is observed

    """
    import pandas as pd
    start_hour, end_hour = l1.get_start_end_hour(symbol)
    TRADING_HOURS=end_hour-start_hour
    start_hour = start_hour % 24

    dt0=datetime.datetime.fromtimestamp(bar[0,0])
    #assert dt.hour < start_hour , 'start of bar file hour > ' + str(start_hour)
    i=0
    # seek to the first bar greater or equal to 18 on that day
    # as KDB files starts from 0 o'clock
    dt=dt0
    while dt.hour<start_hour :
        i+=1
        dt=datetime.datetime.fromtimestamp(bar[i,0])
        if dt.day != dt0.day :
            #raise ValueError('first day skipped, no bars between 18pm - 24am detected')
            print 'first day skipped, no bars between ', start_hour, ' - 24am detected'
            break

    # get the initial day, last price
    day_start=dt.strftime('%Y%m%d')
    utc_s = int(l1.TradingDayIterator.local_ymd_to_utc(day_start, start_hour, 0, 0))
    x=np.searchsorted(bar[1:,0], float(utc_s-3600+bar_sec))
    last_close_px=bar[x,2]
    print 'last close price set to previous close at ', datetime.datetime.fromtimestamp(bar[x,0]), ' px: ', last_close_px
    day_end=datetime.datetime.fromtimestamp(bar[-1,0]).strftime('%Y%m%d')
    # deciding on the trading days
    if dt.hour > end_hour :
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
    trade_days = []
    col_arr = []
    while day < day_end:
        ti.next()
        day1=ti.yyyymmdd()
        utc_e = int(l1.TradingDayIterator.local_ymd_to_utc(day1, end_hour,0,0))

        # get start backwards for starting on a Sunday
        utc_s = utc_e - TRADING_HOURS*3600
        day=datetime.datetime.fromtimestamp(utc_s).strftime('%Y%m%d')

        i=np.searchsorted(bar[:, 0], float(utc_s)-1e-6)
        j=np.searchsorted(bar[:, 0], float(utc_e)-1e-6)
        bar0=bar[i:j,:]  # take the bars in between the first occurance of 18:00:00 (or after) and the last occurance of 17:00:00 or before

        N = (utc_e-utc_s)/bar_sec  # but we still fill in each bar
        ix_utc=((bar0[:,0]-float(utc_s))/bar_sec+1e-9).astype(int)
        bar_utc=np.arange(utc_s+bar_sec, utc_e+bar_sec, bar_sec) # bar time will be time of close price, as if in prod

        print 'getting bar ', day+'-'+str(start_hour), day1+'-'+str(end_hour),' , got ', j-i, 'bars'
        # start to construct bar
        if j<=i :
            print ' NO bars found ',
            if old_cl_repo is not None :
                # a hack to fix the missing CLZ contracts, somehow lost in the copy
                # just another proof how crappy it could be... The new Repo system
                # will fix it. 

                y = int(day1[:4]) - 1998
                old_cl_bar = old_cl_repo[y]
                i=np.searchsorted(old_cl_bar[:, 0], float(utc_s+bar_sec)-1e-6)
                j=np.searchsorted(old_cl_bar[:, 0], float(utc_e)-1e-6)
                if j - i + 1 == len(bar_utc) :
                    barr.append(old_cl_bar[i:j+1, :])
                    last_close_px=old_cl_bar[j, -1]  #lpx is the last column *hecky*
                    trade_days.append(day1)
                    col_arr.append(repo.kdb_ib_col)
                    print 'get from old_cl_repo '
                else :
                    print 'cannot find ', utc_s, ' to ', utc_e, ' from the old_cl_repo, skipping'
            else :
                print ' skipping'

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
            
            # work out vlm mistakes
            # use the daily trade to correct this
            """
            while True:
                bdix = np.nonzero(vlm-vb-vs)[0]
                vbsix = np.nonzero(vb[bdix]+vs[bdix]==0)[0]
                bdix = np.delete(bdix, vbsix)
                if len(bdix) > 0 :
                    print len(bdix), ' ticks have volume mismatch!'
                    vbix = np.nonzero(vb[bdix]==0)[0]
                    if len(vbix) > 0 :
                        print len(vbix), ' buy zero ticks'
                        vbix0 = bdix[vbix]
                        vb[vbix0] = vlm[vbix0]-vs[vbix0]
                    vsix = np.nonzero(vs[bdix]==0)[0]
                    if len(vsix) > 0 :
                        print len(vsix), ' sell zero ticks'
                        vsix0 = bdix[vsix]
                        vs[vsix0] = vlm[vsix0]-vb[vsix0]

                    if len(vbix) + len(vsix) == 0 :
                        print ' no zero buy/sell ticks to adjust!'
                        break
                else :
                    print ' vbs matches with vlm!'
                    break
            """

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
            ba = np.array(bar_arr).T.copy()
            ## make it into the kdb columns
            ## it's hard coded right now
            bt=ba[:,0]
            lr=ba[:,1]
            vl=ba[:,5]
            vbs=ba[:,6]
            # add a volatility measure here
            lrhl=ba[:,2]-ba[:,3]
            vwap=ba[:,4]
            ltt=ba[:,7]
            lpx=ba[:,8]
            barr.append( np.vstack((bt,lr,vl,vbs,lrhl,vwap,ltt,lpx)).T )
            last_close_px=lpx[-1]
            trade_days.append(day1)
            col_arr.append(repo.kdb_ib_col)

        day=day1

    return barr, trade_days, col_arr


kdb_future_symbols = ['6A',  '6B',  '6C',  '6E',  '6J',  '6M',  '6N',  'CL',  'ES', 'FDX',  'FGBL',  'FGBM',  'FGBS',  'FGBX',  'ZF',  'GC',  'HG',  'HO', 'LCO',  'NG',  'RB',  'SI',  'STXE',  'ZN',  'ZB',  'ZC']
kdb_fx_symbols = ['AUD.USD',  'AUD.JPY',  'AUD.NZD',  'USD.CAD',  'USD.CNH',  'EUR.USD',  'EUR.AUD',  'EUR.GBP',  'EUR.JPY',  'EUR.NOK',  'EUR.SEK',  'GBP.USD',  'USD.JPY',  'USD.MXN',  'NOK.SEK',  'NZD.USD',  'USD.SEK',  'USD.TRY',  'XAU.USD',  'USD.ZAR']
kdb_etf_symbols = l1.ven_sym_map['ETF']

future_csv_tz = {
        # downloaded csv in local time format
        'CL'  :'US/Eastern',\
        'GC'  :'US/Eastern',\
        'HO'  :'US/Eastern',\
        'RB'  :'US/Eastern',\
        'SI'  :'US/Eastern',\
        'HG'  :'US/Eastern',\
        'NG'  :'US/Eastern',\
        #
        # downloaded csv in Eurex time
        # FDX has longer trading hours
        #
        'FDX' :'Europe/Berlin',\
        'STXE':'Europe/Berlin',\
        'FGBX':'Europe/Berlin',\
        'FGBL':'Europe/Berlin',\
        'FGBM':'Europe/Berlin',\
        'FGBS':'Europe/Berlin',\
        #
        # downloaded csv in IPE time
        #
        'LCO' :'Europe/London',\
        #
        # downloaded csv in CME time
        #
        '6A'  :'US/Central',\
        '6B'  :'US/Central',\
        '6C'  :'US/Central',\
        '6E'  :'US/Central',\
        '6J'  :'US/Central',\
        '6M'  :'US/Central',\
        '6N'  :'US/Central',\
        'ES'  :'US/Central',\
        'ZF'  :'US/Central',\
        'ZN'  :'US/Central',\
        'ZB'  :'US/Central',\
        'ZC'  :'US/Central',\
        }

def gen_bar0(symbol,year,check_only=False, spread=None, bar_sec=5, kdb_hist_path='.', old_cl_repo = None) :
    year =  str(year)  # expects a string
    venue_path = ''
    symbol_path = symbol
    venue = l1.venue_by_symbol(symbol)
    sym = symbol
    future_match='??'
    if venue == 'FX' :
        if symbol not in kdb_fx_symbols :
            raise ValueError('FX Symbol '+symbol+' not found in KDB!')
        venue_path = 'FX/'
        future_match=''
        symbol_path = sym.replace('.', '')
        if 'USD' in symbol_path :
            symbol_path = symbol_path.replace('USD','')
            sym = symbol_path+'='
        else :
            sym = symbol_path + '=R'
            symbol_path = symbol_path+'R'
    elif venue == 'ETF' :
        venue_path = 'ETF/'
        future_match=''
    elif venue == 'FXFI' :
        venue_path = 'FXFI/'
        future_match=''
    elif sym in l1.RicMap.keys() :
        symbol_path = symbol
        sym = l1.RicMap[symbol]
    elif symbol in ['ZB', 'ZN', 'ZF'] :
        m0 = {'ZB':'US', 'ZN':'TY', 'ZF':'FV'}
        symbol_path = m0[symbol]
        sym = m0[symbol]

    grep_str = kdb_hist_path + '/' + venue_path + symbol_path+'/'+sym+future_match+'_[12]*.csv*'
    print 'grepping for file ', grep_str
    fn=glob.glob(grep_str)

    ds=[]
    de=[]
    fn0=[]
    for f in fn :
        if os.stat(f).st_size < 500 :
            print '\t\t\t ***** ', f, ' is too small, ignored'
            continue
        ds0=f.split('/')[-1].split('_')[1]
        if ds0[:4]!=year :
            continue
        de0=f.split('/')[-1].split('_')[2].split('.')[0]
        ds.append(ds0)
        de.append(de0)
        fn0.append(f)
    ix=np.argsort(ds)
    dss=np.array(ds)[ix]
    des=np.array(de)[ix]
    fn = np.array(fn0)[ix]
    # check that they don't overlap
    for dss0, des0, f0, f1 in zip(dss[1:], des[:-1], fn[1:], fn[:-1]):
        if des0>dss0 :
            # just apply them!
            #raise ValueError('time overlap! ' + '%s(%s)>%s(%s)'%(des0,f0,dss0,f1))
            print 'time overlap! ' + '%s(%s)>%s(%s)'%(des0,f0,dss0,f1)

    if check_only :
        print year, ': ', len(fn0), ' files'
        return
    num_col=8 # adding spd vol, last_trd_time, last_close_px
    bar_lr=[]
    td_arr = []
    col_arr = []
    if len(fn0) == 0 :
        return [], [], []

    for f in fn :
        if f[-3:]=='.gz' :
            print 'gunzip ', f
            try :
                os.system('gunzip '+f)
                f = f[:-3]
            except :
                print 'problem gunzip ', f, ' skipping'
                continue
        print 'reading bar file ',f
        b=bar_by_file(f, symbol)
        ba, td, col = write_daily_bar(symbol,b,bar_sec=bar_sec, old_cl_repo=old_cl_repo)
        bar_lr += ba  # appending daily bars
        td_arr += td
        col_arr += col
        try :
            os.system('gzip -f ' + f)
        except :
            pass

    return bar_lr, td_arr, col_arr


def gen_bar(symbol, year_s=1998, year_e=2018, check_only=False, repo=None, kdb_hist_path = '/cygdrive/e/kdb', old_cl_repo = None, bar_sec=5) :
    td=[]
    col=[]
    years=np.arange(year_s, year_e+1)
    for y in years :
        try :
            barlr, td_arr, col_arr=gen_bar0(symbol,str(y),check_only=check_only, kdb_hist_path = kdb_hist_path, bar_sec=bar_sec, old_cl_repo=old_cl_repo)
            if len(barlr) > 0 :
                if repo is not None :
                    repo.update(barlr, td_arr, col_arr, bar_sec)
                td+=td_arr
                col+=col_arr
        except :
            traceback.print_exc()
            print 'problem getting ', y, ', continue...'

    if check_only :
        return

    if len(td) == 0 :
        print 'nothing found for ', symbol, ' from ', year_s, ' to ', year_e
        return [], [], [], []

    # generate a bad day list 
    sday = td[0]
    eday = td[-1]
    diter = l1.TradingDayIterator(sday)
    day=diter.yyyymmdd()
    bday = []
    while day <= eday :
        if day not in td :
            bday.append(day)
        diter.next()
        day = diter.yyyymmdd()

    return td, col, bday

def fix_days_from_old_cl_repo(td, sday, eday, old_cl_repo) :
    """
    Some days doesn't have history file, or lost in the backup
    so getting them fro the old 5s_repo
    This only applies to CL
    """
    ti = l1.TradingDayIterator(sday)
    day1 = ti.yyyymmdd()
    barr = []
    tda = []
    col = []

    TRADING_HOURS = 23
    end_hour = 17
    bar_sec = 5

    while day1 <=eday :
        if day1 not in td :
            print "read ", day1, " from olc_cl_repo"
            utc_e = int(l1.TradingDayIterator.local_ymd_to_utc(day1, end_hour,0,0))
            utc_s = utc_e - TRADING_HOURS*3600
            y = int(day1[:4]) - 1998
            old_cl_bar = old_cl_repo[y]
            i=np.searchsorted(old_cl_bar[:, 0], float(utc_s+bar_sec)-1e-6)
            j=np.searchsorted(old_cl_bar[:, 0], float(utc_e)-1e-6)
            N = (utc_e-utc_s)/bar_sec
            if j - i + 1 == N :
                barr.append(old_cl_bar[i:j+1, :])
                tda.append(day1)
                col.append(repo.kdb_ib_col)
                print 'get from old_cl_repo '
            else :
                print 'cannot find ', utc_s, ' to ', utc_e, ' from the old_cl_repo, skipping'

        ti.next()
        day1=ti.yyyymmdd()
    return barr, tda, col

def ingest_all_kdb_repo(kdb_path='/cygdrive/c/zfu/data/kdb', repo_path='/cygdrive/c/zfu/kisco/repo', all_sym=kdb_future_symbols + kdb_fx_symbols + kdb_etf_symbols, year_s=1998, year_e=2018) :
    sym_arr = []
    td_arr = []
    tdbad_arr = []
    for sym in all_sym:
        db = None
        if repo_path is not None :
            try :
                db = repo.RepoDailyBar(sym, repo_path=repo_path)
            except :
                print 'creating repo for ', sym, ' repo_path ', repo_path
                db = repo.RepoDailyBar(sym, repo_path=repo_path, create=True)
        try :
            td, _, bd = gen_bar(sym, year_s, year_e, repo=db, kdb_hist_path=kdb_path)
        except :
            import traceback
            traceback.print_exc()
            print 'problem reading symbol ' + sym
            continue
        if len(td) > 0 :
            sym_arr.append(sym)
            td_arr.append(td)
            tdbad_arr.append(bd)
            np.savez_compressed('kdb_dump.npz', sym_arr=sym_arr, td_arr=td_arr, tdbad_arr=tdbad_arr)


