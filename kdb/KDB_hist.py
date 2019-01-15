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

def bar_by_file_etf(fn, skip_header=5) :
    """
    date(0),ric,timeStart(2),exchange_id,country_code,mic,lastTradeTickTime(6),open(7),high,low,close,avgPrice,vwap(12),minSize,maxSize,avgSize,avgLogSize,medianSize,volume(18),dolvol,cntChangePrice,cntTrade,cntUpticks,cntDownticks,sigma,buyvol(24),sellvol(25),buydolvol,selldolvol,cntBuy,cntSell,sideSigma,priceImpr,maxPriceImpr,dolimb,midvol,gmt_offset,lastQuoteTickTime,openBid(37),openAsk,highBid,highAsk,lowBid,lowAsk,closeBid,closeAsk,avgBid,avgAsk,minBidSize,minAskSize,maxBidSize,maxAskSize,avgBidSize,avgAskSize,avgLogBidSize,avgLogAskSize,avgSpread,cntChangeBid,cntChangeAsk,cntTick
    2016.11.10,XLF,04:09:55.000,,,,04:09:57.092,20.99,20.99,20.98,20.98,20.985,20.985,100,100,100,4.60517,100,200,4197,1,2,0,1,0.005,0,-200,0,-4197,0,2,0,-1,200,-4197,0,-5,04:09:57.092,20.98,21.99,20.98,21.99,20.94,21.99,20.94,21.99,20.96,21.99,1,20,21,20,11,20,1.52226,2.99573,479.632,1,0,2

    Return:
    [utc, utc_lt, open, high, low, close, vwap, vol, bvol, svol]

    Note 1:
    All fields before gmt (-5) is empty if there were no trade in this bar period
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

def write_daily_bar(symbol,bar,bar_sec=5,old_cl_repo=None) :
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


