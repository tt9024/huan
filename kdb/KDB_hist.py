import numpy as np
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo
import os
import copy

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

    it = l1.TradingDayIterator(day1,adj_start=False)
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

        fn0,_,_ = kdb_path_by_symbol(kdb_path, symbol)
        fn0 += '/trd/*'+day1+'*'
        fn = glob.glob(fn0)
        # filter out the spread files from fn
        for f0 in fn :
            if len(f0.split('/')[-1].split('_')[0].split('-'))>1:
                print 'removing ', f0, ' not a recognized trade file!'
                fn.remove(f0)
                break

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
            try :
                ts.append(bar_by_file_future_trd(trdfn))
                fcarr.append(trdfn.split('/')[-1].split('_')[0])
                darr.append(day1)
            except:
                print 'problem getting bar from trade file ', trdfn, ' continue...'

        else :
            print 'no trd file found on ', day1, ' symbol ', symbol, ' glob ', fn0

        it.next()
        day1=it.yyyymmdd()
    return ts, fcarr, darr

def guess_dir_trd(ua0, px0, da0, sa0, do_merge=False, ticksize=1e-8) :
    """
    guess the trade direction given the time,px,sz series
    mainly for EUREX futures trade files, i.e. FDX,
    where no trade directions are given.
    Can be used for other venues, such as LCO,
    where only some trade directions are given.
    (Caution: need to find out if IB have those trade information)

    ua: utc in local time
    px: price at ua
    da: a set of given directions, 'v','^','', where '' is 
        to be guessed.
    sa: size of trade, used to calculate weighted avg px
        for ticks with the same timestammp (merge)
        sa is strictly positive
    return bs
    bs: array of 1 or -1 with same length. 1=B,-1=S

    Logic:
    A simple way: take the price diff with immediate
    previous.  if no diff, then use the existing. 
    The first trade takes from the next. 
    Some trades come in with the same timestamp at milli-seconds
    I have decided to force same direction so they could be merged 
    later. This is done by replacing the px of those ticks with
    a wighted avg of them.

    NOTE: the assumption for merge doesn't apply well. 
    There could be multiple directions happen in 1 millisecond. 
    Also, the weighted avg price requires the ticksize, which
    is an additional dependancy.  SO, should disable it.
    """
    # merge trades with same utc and direction
    # extend the given directions within the given merge group
    # forward and backward to avoid conflicting given directions
    ua=(ua0.copy()*1000.0+0.5).astype(int) ; px=px0.copy(); sa=sa0.copy(); da=da0.copy()
    n=len(px)

    if do_merge :
        sp=np.cumsum(sa*px)
        sz=np.cumsum(sa)
        ix0=np.nonzero(np.abs(ua[1:]-ua[:-1])<1e-13)[0]
        px[ix0]=0
        ix = np.r_[np.nonzero(ua[1:]-ua[:-1])[0], n-1]
        spix=sp[ix]-np.r_[0,sp[ix[:-1]]]
        szix=sz[ix]-np.r_[0,sz[ix[:-1]]]
        px[ix]=spix/szix  # weighted avg price
        # this fills the merge group with the weighted avg price
        # ensure the subsequent logic produces same direction for
        # the merge group
        repo.fwd_bck_fill(px,v=0,fwd_fill=False,bck_fill=True) 
        px = (px/ticksize+0.5).astype(int).astype(float)*ticksize # normalize towards a small tick

        # tricky part -
        # observe the given directions within the merge group
        # this modifies (adds to) the given directions directly
        ixd=np.nonzero(da!='')[0]
        if len(ixd) > 0 :
            ix0=np.union1d(ix0,ix0+1) # all the indexes subject for merge
            eqix=np.intersect1d(ix0,ixd)  # the seeds to be propagated within merge group
            if len(eqix) > 0 :
                # eqix to be propagate towards neighboring indexes
                # forward propagate seed first
                # in case there are multiple seeds (with contradicting directions)
                for sn,sdstr in zip([1, -1 ],['right','left']):
                    eqix0=eqix.copy()
                    cnt=1
                    # ixd maintains to be current idx with direction
                    while True :
                        """
                        # just to show how compliccated and wrong the code could be!
                        # ixd0 is currently undecided directions
                        ixd0=np.intersect1d(np.delete(np.arange(n),ixd), np.clip(eqix+cnt*sn,0,n-1))
                        if len(ixd0)==0 :
                            break

                        ixeq0=np.clip(np.searchsorted(ua[eqix], ua[ixd0],side=sdstr),0,len(eqix)-1)
                        ixeq1=np.nonzero(ua[eqix][ixeq0]-ua[ixd0]==0)[0]
                        if len(ixeq1)==0:
                            break
                        da[ixd0[ixeq1]]=da[eqix][ixeq0][ixeq1]
                        ixd=np.union1d(ixd,ixd0[ixeq1])
                        cnt+=1

                        """
                        ixd0=np.intersect1d(np.delete(np.arange(n),ixd), eqix0+cnt*sn)
                        if len(ixd0)==0 :
                            break
                        t0=ua[ixd0-cnt*sn]
                        t1=ua[ixd0]
                        ixeq=np.nonzero(t0-t1==0)[0]
                        if len(ixeq)==0 :
                            break
                        da[ixd0[ixeq]]=da[ixd0-cnt*sn][ixeq]
                        ixd=np.union1d(ixd,ixd0[ixeq])
                        cnt+=1

    bs = np.zeros(n)
    pd=np.r_[0,px[1:]-px[:-1]]
    bix0=np.nonzero(da=='^')[0]
    six0=np.nonzero(da=='v')[0]
    # take the given direction as priority. 
    # the fillna forward could be improved by
    # also take given first and then fill the rest
    nix0=np.nonzero(da=='')[0]
    bix1=np.intersect1d(np.nonzero(pd>ticksize/2)[0],nix0)
    six1=np.intersect1d(np.nonzero(pd<-ticksize/2)[0],nix0)
    bs[np.r_[bix0,bix1]]=1
    bs[np.r_[six0,six1]]=-1
    # the first one
    ix0=np.nonzero(bs)[0][0]
    bs[0]=-bs[ix0]
    # fill forward
    repo.fwd_bck_fill(bs, v=0)
    return bs

def find_patch_days(symarr,sday,eday,kdb_path='./kdb', check_col='px') :
    """
    just because I had a bug in using
    bar_raw['sz'] instead of sz at the end!
    I lost all zero size ticks. Would like to 
    get them as size 1 trade with lpx/lr updates!
    This is important for earlier days. 
    check_col could be either 'px' or 'sz'
    """
    fza={}
    for symbol in symarr :
        fn0,_,_ = kdb_path_by_symbol(kdb_path, symbol)
        fn0+='/trd/*_trd*'
        fa=glob.glob(fn0)
        fza[symbol]={'d':[],'f':[]}
        for i, fn in enumerate(fa) :
            try :
                tday=fn.split('/')[-1].split('_')[-1].split('.')[0]
                if tday<sday or tday>eday :
                    continue
                print 'checking ', fn
                bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[4,5,6],skip_header=5,dtype=[('px','<f8'),('sz','<f8'),('dir','|S2')])
                n = len(bar_raw)
                ix=[]
                if n> 0 :
                    val = bar_raw['px']
                    gix=np.nonzero(np.isfinite(val))[0]
                    bar_raw=bar_raw[gix]
                    if check_col == 'px':
                        ix=np.nonzero(bar_raw['px']==0)[0]
                        if len(ix) == 0:
                            sz=bar_raw['sz']
                            nix=np.nonzero(sz>0)[0]
                            if len(nix)>0 :
                                bar_raw=bar_raw[nix]
                            ix=ix_bad_px_trd(bar_raw,fn)
                    elif check_col=='sz' :
                        val = bar_raw['sz']
                        ix=np.nonzero(val==0)[0]
                    else :
                        print 'unknown check_col!'
                        return fza
                    if len(ix) > 0 :
                        fza[symbol]['f'].append(fn)
                        fza[symbol]['d'].append(tday)
            except (KeyboardInterrupt) :
                print 'interrupt!'
                return fza
            except :
                traceback.print_exc()
                print 'problem with ', fn
            if i % 100 == 0 :
                print symbol, i, fza[symbol]['d']
    return fza

def ix_bad_px_trd(bar_raw,fn) :
    # such prices in 6E or 6A, typically without a direction,
    # could get stuck to a number and show huge sizes. 
    # this must be bad parser with KDB app
    # The sizes are not in the KDB bar and since price bad, cannot guess.
    # mark as bad to be 10 pip (0.1%) diff with neighboring good ticks (w/ dir)
    bix=np.nonzero(bar_raw['dir']=='')[0]
    if len(bix)>0 :
        gix=np.delete(np.arange(len(bar_raw)),bix)
        if len(gix)==0 :
            # no directional ticks at all
            # i.e. FDXM7_trd_20070412
            print 'no directional trades, take all'
            return []
        px=bar_raw['px'][gix]
        bpx=bar_raw['px'][bix]
        gix1=np.clip(np.searchsorted(gix,bix),0,len(gix)-1)
        gix0=np.clip(gix1-1,0,len(gix)-1)
        gix11=np.clip(gix1+1,0,len(gix)-1)
        gix00=np.clip(gix0-1,0,len(gix)-1)
        mpx=(px[gix1]+px[gix0])/3+(px[gix00]+px[gix11])/6
        dpx=np.abs(mpx-bpx)/mpx
        lmt=0.0005
        sym=fn.split('/')[-1].split('_')[0][:-2]
        if sym not in ['CL','LCO'] : 
            # CL and LCO has big ticks
            # tends to have bigger lmt
            lmt=0.00025
        ix_=np.nonzero(dpx>lmt)[0]
        if len(ix_)>0 :
            if len(ix_) > max(len(bpx)/5,5) :
                print len(ix_), ' stucked non-directional price detected, excluding all non-directional!'
                ix_=np.arange(len(bpx))
            print len(ix_), ' bad prices out of ', len(bix), ' non-directinal ticks'
            # Just to eyeball the numbers
            #for g0, b0 in zip(mpx[ix_], bar_raw['px'][bix[ix_]]) :
            #    print g0, b0, np.abs(g0-b0)/g0
            return bix[ix_]
    return []


def bar_by_file_future_trd(fn,guess_dir=True) :
    """
    date,ric,time,gmt_offset,price,volume,tic_dir
    2009.10.28,CLZ9,00:00:00.224,-4,,58,
    2009.10.28,CLZ9,00:00:14.890,-4,79.4,1,^
    2009.10.28,CLZ9,00:00:14.890,-4,79.39,1,v
    
    where gmt_offset is w.r.t ny local time.
    price can be none, a implied trade or block trade
    tic_dir: ^ buy v sell

    Note: if guess_dir is True, for EURX symbols,
    then the direction is the diff from prev px. 
    diff zero filled by prev direction. First dir is lost

    Return:
    [utc, px, bsvol]
    """
    bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[0,2,3,4,5,6],skip_header=5,dtype=[('day','|S12'),('time','|S16'),('gmtoff','i8'),('px','<f8'),('sz','f8'),('dir','|S2')])
    ts=[]
    gix=np.nonzero(np.isfinite(bar_raw['px']))[0]
    print 'read trd %s size(%d) lines(%d) good(%f)'%(fn, os.stat(fn).st_size, len(bar_raw), float(len(gix))/len(bar_raw))
    bar_raw=bar_raw[gix]

    # zero prices should be removed
    zix=np.nonzero(np.abs(bar_raw['px'])>1e-8)[0]
    if len(zix) < len(bar_raw) :
        print 'removing ', len(bar_raw)-len(zix), ' zero price ticks!'
        bar_raw=bar_raw[zix]

    assert len(bar_raw) >0 ,  'read zero bars!'

    # finally, bad prices should be removed
    # such prices in 6E or 6A, typically without a direction,
    # could get stuck to a number and show huge sizes. 
    # These trades are not included in KDB bar and since price bad, 
    # cannot guess.
    bix=ix_bad_px_trd(bar_raw,fn)
    if len(bix)>0 :
        print len(bix), ' bad non-directional ticks removed due to bad price!'
        bar_raw=np.delete(bar_raw,bix)
    
    px = bar_raw['px']
    sz = bar_raw['sz']
    # some sizes could be missing, replace with 1
    ix1=np.nonzero(np.isnan(sz))[0]
    if len(ix1) > 0 :
        print 'got %d missing size, setting all to 1!'%(len(ix1))
        sz[ix1]=1
    sz=sz.astype(int)

    ixz=np.nonzero(sz==0)[0]
    if len(ixz)>0 :
        print 'setting %d zero sizes to 1'%(len(ixz))
        sz[ixz]=1
    assert len(np.nonzero(sz<0)[0])==0, 'got negative sizes from %s!'%(fn)
    # deal with the time
    ua=[]
    for i, b in enumerate(bar_raw) :
        dt = datetime.datetime.strptime(b['day'] + ' ' +b['time'], '%Y.%m.%d %H:%M:%S.%f')
        utc = l1.TradingDayIterator.dt_to_utc(dt, 'GMT', micro_fraction=True) - b['gmtoff']*3600
        ua.append(utc)
    ua=np.array(ua)

    # getting the trade directions
    if guess_dir :
        bs=guess_dir_trd(ua,px,bar_raw['dir'],sz) 
    else :
        bs=np.zeros(len(ua))
        ixb=np.nonzero(bar_raw['dir']=='^')[0]
        bs[ixb]=1
        ixs=np.nonzero(bar_raw['dir']=='v')[0]
        bs[ixs]=-1
        ixz=np.nonzero(bs==0)[0]
        if len(ixz) > 0 :
            print '%d bars have no directions, try set guess_dir=True'%(len(ixz))

    # merge trades with same utc and direction
    ts = np.array([ua,px,bs*sz]).T
    ux = ts[:,0]*np.sign(ts[:,2])
    sz=np.cumsum(ts[:,2])
    ix = np.r_[np.nonzero(np.abs(ux[1:]-ux[:-1])>1e-13)[0], len(ux)-1]
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

def gen_bar_trd(sym_array, sday, eday, repo_trd_path, repo_bar_path, kdb_path='./kdb', bar_sec=1, nc=False, check_col=None) :
    """
    getting from the ts [utc, px, signed_vol]
    output format bt, lr, vl, vbs, lrhl, vwap, ltt, lpx

    repo_trd_path: repo to store the 1S trd bars
    repo_bar_path: repo to read the 5S kdb bars. Needed for
                   Sunday nights and missing days
    check_col: 'px' or 'sz', only run on those symbol/days
               that the colume has a zero. i.e. a day with zero px or sz
    use_repo_overnight_lr: if True, try to get the first lr from repo
               and set to the new data. This is useful when only
               updating a random sets of days that 
    return : None
        update (remove first) dbar with bar_arr, days, col_arr

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

    3. Biggest Problem: No Sunday market data. Have to 
       use the 5 second to stict it. Distribute all
       trades to the last second of the 5 second interval
       and taking the overnight log-ret

    Bigger than biggest problem:
    1. trades that has no direction: 
       Can be guessed, a simple way is better, a more
       complicated way is tried but no good
    
    Bigger than Bigger than biggest problem:
    1. ZERO SIZE TRADES!
    Due to CME's policy or Reuter's data,
    All CME symbols seem to have mostly zero size trade 
    during 9am to 15:15pm NewYork Time, until 2007-01-01.
    The following assets seems to not having such zero sizes
    ES, FDX, LCO, 6C, STXE, FGB*, ZN is fine.
    There is nothing I can do with it, KDB Bar having the
    exactly same information, worse, sometimes with contradictory
    directions.  There is no sure way to tell which one is
    correct and I decided to leave the trade directions as is.
    Another thing is to add trade count into each second bar.

    2. Zero prices
    Some CME FX, such as 6J, have zero prices, with size, no dir. 
    The KDB bars don't include them, and although the size is big,
    they don't seem to impact the market.  Seems to be spread or
    implied trade.  Without price, it's impossible to "tell" dir.
    And I suspect it won't show in IB. So just ignore those zero prices.

    3. Spikes in price. Some early days have such spike, due to 
       market iliquidity.  Modeul should remove outliers.
    """

    # my hack for figure out the patch run targets
    fza=None
    use_repo_overnight_lr=False
    if check_col is not None :
        fza=find_patch_days(sym_array,sday,eday,kdb_path=kdb_path,check_col=check_col)
        use_repo_overnight_lr=True
    for symbol in sym_array :
        if fza is not None :
            fda=np.unique(fza[symbol]['d'])
            if len(fda)==0 :
                continue
            else :
                print symbol, ' check_col ', check_col, ' running for ', fda
        else :
            fda=None

        try :
            dbar = repo.RepoDailyBar(symbol, repo_path=repo_trd_path)
        except :
            print 'repo_trd_path failed, trying to create'
            dbar = repo.RepoDailyBar(symbol, repo_path=repo_trd_path, create=True)

        try :
            dbar5S = repo.RepoDailyBar(symbol, repo_path=repo_bar_path)
        except :
            print 'repo_bar_path failed, no ammending from 5S bar is possible!'
            dbar5S = None

        start_hour, end_hour = l1.get_start_end_hour(symbol)
        TRADING_HOURS=end_hour-start_hour
        # sday has to be a trading day
        it = l1.TradingDayIterator(sday)
        tday = it.yyyymmdd()
        if tday != sday :
            raise ValueError('sday has to be a trading day! sday: '+sday + ' trd_day: ' + tday)

        tds = 1
        if start_hour < 0 :
            tds = 2
        da = [] ; ta = [] ;  fa = []
        lastpx=0
        prev_con=''
        while tday <= eday :
            if fda is not None and tday not in fda :
                it.next()
                tday=it.yyyymmdd()
                continue

            eutc = it.local_ymd_to_utc(tday,h_ofst=end_hour)
            sutc = eutc - (TRADING_HOURS)*3600
            pday = datetime.datetime.fromtimestamp(sutc).strftime('%Y%m%d')

            # get for trading day
            if len(da) == 2 and da[1] == pday :
                day0 = tday
                da=[da[-1]] ; ta=[ta[-1]] ; fa=[fa[-1]]
            else :
                # either initial or previously broken 2-day or 1 day 
                day0 = pday
                da=[] ; ta=[] ; fa=[]

            try :
                tsarr, fcarr, darr = bar_by_file_future_trd_day(symbol, day0, tday, kdb_path=kdb_path, nc=nc)
            except (KeyboardInterrupt) :
                print 'interrupt!'
                return
            except :
                tsarr=[] ; fcarr=[]; darr=[]
            da+=darr ; fa+=fcarr ; ta+=tsarr

            Filled=False
            # try to patch the missing days from 5S repo
            if len(da) != tds :
                if dbar5S is not None:
                    b_,c_,bs_=dbar5S.load_day(tday)
                    if len(b_) > 0 :
                        if tds == 1 or len(da) == 0:
                            # just put everything in from 5S to 1S
                            print 'missing day', tday, ' filling from ', repo_bar_path, ' bs=',bs_
                            if bs_ != bar_sec :
                                print ' scaling to ', bar_sec
                                b_ = dbar5S._scale(tday,b_,c_,bs_,c_,bar_sec)
                            # simply write everything in and call it done
                            dbar.remove_day(tday)
                            dbar.update([b_],[tday],[c_],bar_sec)
                            # no idea of prev_con
                            prev_con=''
                            lastpx=0
                            Filled=True
                        else : 
                            # take out the bar and ammend half day
                            ts_ = b_[:,repo.ci(c_,repo.utcc)]
                            px_ = b_[:,repo.ci(c_,repo.lpxc)]
                            lr_ = b_[:,repo.ci(c_,repo.lrc)]

                            # assemble a tsarr from the bar       
                            vol_= b_[:,repo.ci(c_,repo.volc)]
                            vbs_= b_[:,repo.ci(c_,repo.vbsc)]
                            bv_=(vol_+vbs_)/2
                            sv_=-vol_+bv_        # negative
                            ts_=np.tile(ts_,(2,1)).T.flatten()
                            px_=np.tile(px_,(2,1)).T.flatten()
                            bs_=np.array([bv_,sv_]).T.flatten()
                            ta_=np.array([ts_,px_,bs_]).T

                            if pday not in da :
                                utc0=ta[0][0,0]
                                ix=np.searchsorted(ts_,utc0)
                                # if ix == 0, first ta[0]
                                # will be zero, to be handled below
                                if ix == 0 :
                                    # all needed is the ta[0]
                                    # divide them into two files and update lastpx
                                    ix0=np.searchsorted(ta[0][:,0],sutc-1e-6)
                                    ix1=ta[0].shape[0]
                                    lastpx=ta[0][ix0,1]*np.exp(-lr_[0])
                                    fa=[fa[0],fa[0]]
                                    da=['',da[0]]
                                    ixm=ix0+(ix1-ix0)/2
                                    ta=[ta[0][ix0:ixm,:], ta[0][ixm:ix1,:]]
                                else :
                                    lastpx=px_[0]*np.exp(-lr_[0])
                                    fa=['',fa[0]]
                                    da=['',da[0]]
                                    ta=[ta_[:ix,:],ta[0]]
                            else :
                                utc0=ta[0][-1,0]
                                ix=np.searchsorted(ts_,utc0)
                                if ix == len(ts_) :
                                    ix0=np.searchsorted(ta[0][:,0],sutc-1e-6)
                                    ix1=np.searchsorted(ta[0][:,0],eutc+1e-6)
                                    ixm=ix0+(ix1-ix0)/2
                                    fa=[fa[0],fa[0]]
                                    da=[da[0],'']
                                    ta=[ta[0][ix0:ixm,:],ta[0][ixm:ix1,:]]
                                else :
                                    ix0_=ix
                                    ix1_=len(ts_)
                                    fa=[fa[0],'']
                                    da=[da[0],'']
                                    ta=[ta[0],ta_[ix0_:ix1_,:]]

            if tds==2 and len(da)==1 and pday not in da :
                # a special case where prev day is Sunday and not in bar5m
                # as the case for LCO 19980907
                # just use whatever given and fill the Sunday night zero
                # all needed is the ta[0]
                # divide them into two files and update lastpx
                print 'missing a previous half day, use the second part anyway!'
                ix0=np.searchsorted(ta[0][:,0],sutc-1e-6)
                ix1=ta[0].shape[0]
                fa=[fa[0],fa[0]]
                da=['',da[0]]
                ixm=ix0+(ix1-ix0)/2
                ta=[ta[0][ix0:ixm,:], ta[0][ixm:ix1,:]]

            if len(da) != tds :
                print 'error getting trading day ', tday, ' found only ', da, fa
                da=[] ; fa=[] ; fa=[]
                lastpx=0
                prev_con=''
            elif not Filled :
                # this is the good case, prepare for the bar
                # 1) get bar with start/stop, 2) contract updated 3) lastpx
                # need to allow for entire content being in one ta, i.e. some
                # days having tds==2 but all contents in one ta, due to gmt_offset
                px_diff=0
                if tds==2:
                    if fa[0] != fa[1] :
                        # adjust price
                        px_diff=ta[1][0,1]-ta[0][-1,1]
                        ta[0][:,1]+=px_diff
                    bar=np.vstack(ta)
                else :
                    if fa[0]!=prev_con:
                        lastpx=ta[0][0,1]
                    bar=ta[0]
                ix0=np.searchsorted(bar[:,0],sutc)
                ix1=np.searchsorted(bar[:,0],eutc+1e-6)
                if lastpx==0 :
                    lastpx=bar[max(ix0-1,0),1]
                else :
                    lastpx+=px_diff
                bar=bar[ix0:ix1,:]

                # here we go!
                if len(bar) > 0 :
                    # have everything, need to get to
                    # output format bt, lr, vl, vbs, lrhl, vwap, ltt, lp

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

                    # honor repo's overnight lr
                    if use_repo_overnight_lr :
                        try :
                            b_,c_,bs_=dbar.load_day(tday)
                            if len(b_)>0 and bs_==bar_sec :
                                lr_ = b_[:,repo.ci(c_,repo.lrc)]
                                print 'Using the repo overnight lr ', lr_[0], '. Replacing lr: ', lr[0]
                                lr[0]=lr_[0]
                        except :
                            print 'problem trying to use repo overnight lr!'
                            traceback.print_exc()

                    b=np.vstack((bt,lr,vol,vbs,lrhl,vwap,ltt,lpx)).T
                    d=tday
                    c=repo.kdb_ib_col
                    dbar.remove_day(d)
                    dbar.update([b],[d],[c],bar_sec)
                    lastpx=lpx[-1]
                    prev_con=fa[-1]

            it.next()
            tday=it.yyyymmdd()


def write_daily_bar(symbol,bar,bar_sec=5,old_cl_repo=None, trade_day_given=None) :
    """
    input format:
         utc, utc_lt, open, high, low, close, vwap, vol, bvol, svol
         Where : 
             utc is the start of the bar

    output format: 
         bt,lr,vl,vbs,lrhl,vwap,ltt,lpx
         Where :
             bt is the end of the bar time, as lr is observed

    NOTE: this only works for CL, and no wonder it sucks so much
          that it fails miserably for other asserts with
          different start_hour and end_hour.  

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
    if trade_day_given is not None:
        trd_day_start=trade_day_given
        trd_day_end=trade_day_given
    else :
        if dt.hour > end_hour :
            ti=l1.TradingDayIterator(day_start,adj_start=False)
            ti.next()
            trd_day_start=ti.yyyymmdd()
        else :
            trd_day_start=day_start
        trd_day_end=day_end
    print 'preparing bar from ', day_start, ' to ', day_end, ' , trading days: ', trd_day_start, trd_day_end

    ti=l1.TradingDayIterator(trd_day_start)
    day1=ti.yyyymmdd()  
    barr=[]
    trade_days = []
    col_arr = []
    while day1 <= trd_day_end:  
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

        ti.next()
        day1=ti.yyyymmdd()

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

def kdb_path_by_symbol(kdb_hist_path, symbol) :
    venue_path=''
    symbol_path=symbol
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
    ret_path = kdb_hist_path + '/' + venue_path + symbol_path
    return ret_path, sym, future_match

def find_kdb_file_by_symbol(symbol, kdb_path='./kdb') :
    ret_path, sym, future_match = kdb_path_by_symbol(kdb_path, symbol)
    grep_str = ret_path +'/'+sym+future_match+'_[12]*.csv*'
    print 'grepping for file ', grep_str
    fn=glob.glob(grep_str)
    return fn

def gen_bar0(symbol,year,check_only=False, spread=None, bar_sec=5, kdb_hist_path='.', old_cl_repo = None) :
    year =  str(year)  # expects a string
    fn = find_kdb_file_by_symbol(symbol, kdb_path=kdb_hist_path)

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
            traceback.print_exc()
            print 'problem reading symbol ' + sym
            continue
        if len(td) > 0 :
            sym_arr.append(sym)
            td_arr.append(td)
            tdbad_arr.append(bd)
            np.savez_compressed('kdb_dump.npz', sym_arr=sym_arr, td_arr=td_arr, tdbad_arr=tdbad_arr)



########################################################################
#
# PATCHING FOR THE FOLLOWING PROBLEMS
#
# 1. OUT contracts are in repo during roll time. 
#    Need to switch to IN contract, or a contracts with more trades
# 2. Missing days from KDB bar, could be filled from CME or trd repos
# 
##########################################################################


def in_out_day_dict(symbol, kdb_path='./kdb') :
    """
    So the last days of a contract has little trades. 
    The first day of a contract couldn't over-write due to 
    over-night lr. 
    This is to get in the first days of a bar file, if that
    day appear as the last day of previous contract
    """
    fn = find_kdb_file_by_symbol(symbol, kdb_path=kdb_path)
    # this is everyfile I have, pop a day dict
    day_dict={}
    try :
        for f in fn :
            f0 = f.split('/')[-1]
            d1=f0.split('_')[1]
            d2=f0.split('_')[2].split('.')[0]
            tdi=l1.TradingDayIterator(d1)
            day=tdi.yyyymmdd()
            con = f0.split('_')[0][-2:]
            con = con[-1]+con[0]
            # there never be a shortage of traps: 0Z > 9Z
            if con[0]=='0' :
                con='Z'+con[1] # The biggest!
            while day <= d2 :
                # populate a day dict on contracts
                try :
                    dd = day_dict[day]
                    if f not in dd['fa'] :
                        dd['fa'].append(f)
                        dd['cnt']+=1
                        if con > dd['con'] :
                            dd['con'] = con
                            dd['fn']=f
                except :
                    dd = {'cnt':1, 'con':con, 'fn':f, 'fa':[f]}
                day_dict[day]= copy.deepcopy(dd)
                tdi.next()
                day=tdi.yyyymmdd()
    except:
        traceback.print_exc()
        raise RuntimeError('runtime error for fixing first contract stuff! See errors before, I am sure there must be plenty.')

    return day_dict

def run_inout_dict(symbol, day_dict, dbar_update, dbar_read=None) :
    # reload all days with a cnt > 1
    for d in day_dict.keys() :
        dd=day_dict[d]
        if dd['cnt'] > 1 :
            print 'getting ', dd
            try :
                run_inout(symbol, d, dd['fn'], dbar_update, dbar_read=dbar_read)
            except KeyboardInterrupt as e :
                raise e
            except Exception as e:
                print 'problem with ', d, ' ' , dd, ' continue!'

def run_inout(symbol, d, fn, dbar, bar_sec=5, dbar_read=None) :
    try :
        if dbar_read is None :
            dbar_read = dbar
        b=bar_by_file(fn, symbol)
        if len(b)==0:
            return
        ba, td, col = write_daily_bar(symbol,b,bar_sec=bar_sec,trade_day_given=d)
        if len(ba)==0 or d != td[0] or len(ba[0])==0:
            print 'got NOTHGING on ', d, ' damn!'
            return
        b_,c_,bs_=dbar_read.load_day(d)
        if len(b_)==0 : 
            print d, ' not found in repo??? use bar!'
            dbar.update(ba, td, col, bar_sec)
            return
        assert bs_ == bar_sec, 'bar_sec mismatch?? What happened? day=%s, fn=%s, bs=%d'%(d,fn,bs_)

        # compare number of trades
        ba = ba[0]
        col=col[0]
        vol= ba[:, repo.ci(col,repo.volc)]
        vol0=b_[:, repo.ci(c_, repo.volc)]
        print 'got ', np.sum(vol) , ' vesus ', np.sum(vol0), ' (trades)',
        if np.sum(vol) <= np.sum(vol0) :
            print ' NOT better, skipping!'
            return
        print ' TAKING it!'

        # first non-zero lr
        lr=ba[:,repo.ci(col,repo.lrc)]
        ix = np.nonzero(lr!=0)[0][0]
        if ix > 10 :
            print ' filling initial ', ix, ' bars with repo '
            lpx0 = b_[ix,repo.ci(c_,repo.lpxc)]
            lpx =  ba[ix,repo.ci(col,repo.lpxc)]
            pd = lpx-lpx0
            b_[:ix,repo.ci(c_,repo.lpxc)]+=pd
            b_[:ix,repo.ci(c_,repo.vwapc)]+=pd
            ba[:ix,:]=b_[:ix,:]

            # redo the lr
            lpx =  ba[:,repo.ci(col,repo.lpxc)]
            lr=np.log(lpx[1:])-np.log(lpx[:-1])
            ba[1:,repo.ci(col,repo.lrc)]=lr
        else :
            # get the first lr
            ba[0,repo.ci(col,repo.lrc)]=b_[0,repo.ci(c_,repo.lrc)]

        dbar.remove_day(d)
        dbar.update([ba], [d], [col], bar_sec)

    except Exception as e :
        print 'problem trying to use repo overnight lr!'
        traceback.print_exc()
        raise RuntimeError(e)

def fill_day_dict(sym_arr, repo_path_arr, sday='19980101', eday='20180214') :
    """
    For each dd, fill in volume and sum of abs(lr) from dbar_read
    """
    sym_dict={}
    for symbol in sym_arr :
        print 'finding ', symbol
        day_dict={}
        dbar_arr=[]
        for rp in repo_path_arr :
            try :
                dbar_arr.append(repo.RepoDailyBar(symbol, repo_path=rp))
            except :
                continue
        if len(dbar_arr) == 0 :
            print ' nothing found for symbol ', symbol, '!!!'
            continue

        tdi=l1.TradingDayIterator(sday)
        d=tdi.yyyymmdd()
        while d <= eday :
            day_dict[d]={}
            for dbar_read in dbar_arr :
                bdict={}
                try :
                    b,c,bs=dbar_read.load_day(d)
                    bdict['totvol']=np.sum(np.abs(b[:,repo.ci(c,repo.volc)]))
                    bdict['totlr']=np.sum(np.abs(b[:,repo.ci(c,repo.lrc)]))
                except KeyboardInterrupt as e :
                    raise e
                except :
                    bdict['totvol']=0
                    bdict['totlr']=0
                day_dict[d][dbar_read.path]=copy.deepcopy(bdict)
            tdi.next()
            d=tdi.yyyymmdd()
        sym_dict[symbol]=copy.deepcopy(day_dict)
    return sym_dict

def fix_kdb_20171016_20171017(sym_arr=kdb_future_symbols) :
    repo_path_write='./repo'
    repo_path_read_arr=['./repo_cme','./repo_trd']
    day_arr=['20171016','20171017']
    bar_sec=5
    repo.UpdateFromRepo(sym_arr, day_arr, repo_path_write, repo_path_read_arr, bar_sec, keep_overnight='onzero')

def update_from_cme(sym_dict) :
    """
    Since KDB bar's trade is front+back, so if it's trade volume is less than cme's front, then
    update with cme on the day
    should_upd_func = repo.trd_cmp_func compares the trade vol
    """
    repo_path_write='./repo'
    repo_path_read='./repo_cme'
    bar_sec=5
    for symbol in sym_dict.keys() :
        dd = sym_dict[symbol]
        days=[]
        for d in dd.keys() :
            for rp in dd[d].keys():
                if repo_path_read not in rp :
                    continue
                if dd[d][rp]['totvol']>0 :
                    days.append(d)
                    break
        if len(days) > 0 :
            repo.UpdateFromRepo([symbol], days, repo_path_write, [repo_path_read], bar_sec, keep_overnight='onzero',\
                                should_upd_func=repo.trd_cmp_func)

def fill_missing_kdb(sym_dict) :
    """
    getting from the 
    sym_dict = fill_day_dict(sym_arr=kdb_future_symbols, repo_path_arr=['./repo'])
    """
    repo_path_write='./repo'
    repo_path_read_arr=['./repo_cme','./repo_trd']
    bar_sec=5
    for symbol in sym_dict.keys() :
        day_arr=[]
        day_dict=sym_dict[symbol]
        for d in day_dict.keys() :
            rp = repo_path_write + '/' + symbol
            if day_dict[d][rp]['totlr'] == 0 :
                day_arr.append(d)
        if len(day_arr) > 0 :
            repo.UpdateFromRepo([symbol], day_arr, repo_path_write, repo_path_read_arr, bar_sec, keep_overnight='no')

def fix_inout(symarr=kdb_future_symbols, repo_update_path='./repo', repo_read_path='./back_repo/repo_kdb') :
    sym_dict={}
    for sym in kdb_future_symbols :
        print 'Fix inout for ', sym
        dd=in_out_day_dict(sym)
        sym_dict[sym]=copy.deepcopy(dd)
        dw=repo.RepoDailyBar(sym, repo_path=repo_update_path)
        dr=repo.RepoDailyBar(sym, repo_path=repo_read_path)
        run_inout_dict(sym, dd, dw, dr)
    return sym_dict

