import numpy as np
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo
import os

def bar_by_file(fn, skip_header=5) :
    bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[0,2,3,4,5,6,7,9,10,11,12], skip_header=skip_header,dtype=[('day','|S12'),('bar_start','|S14'),('last_trade','|S14'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8'),('vwap','<f8'),('volume','i8'),('bvol','i8'),('svol','i8')])
    bar=[]
    for b in bar_raw :
        dt=datetime.datetime.strptime(b['day']+'.'+b['bar_start'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc=float(l1.TradingDayIterator.local_dt_to_utc(dt))
        dt_lt=datetime.datetime.strptime(b['day']+'.'+b['last_trade'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc_lt=float(l1.TradingDayIterator.local_dt_to_utc(dt))+float(b['last_trade'].split('.')[1])/1000.0

        bar0=[utc, utc_lt, b['open'],b['high'],b['low'],b['close'],b['vwap'],b['volume'],b['bvol'],b['svol']]
        bar.append(bar0)

    bar = np.array(bar)
    open_px_col=2
    ix=np.nonzero(np.isfinite(bar[:,open_px_col]))[0]
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

def gen_bar0(symbol,year,check_only=False, spread=None, bar_sec=5, kdb_hist_path='.', old_cl_repo = None) :
    year =  str(year)  # expects a string

    venue_path = ''
    symbol_path = symbol
    venue = l1.venue_by_symbol(symbol)
    if venue == 'FX' :
        venue_path = 'FX/'
        symbol_path = symbol.replace('.', '')
        if 'USD' in symbol_path :
            symbol_path = symbol_path.replace('USD','')
        else :
            symbol_path = symbol_path + 'R'
    elif venue == 'ETF' :
        venue_path = 'ETF/'
    elif venue == 'FXFI' :
        venue_path = 'FXFI/'

    fn=glob.glob(kdb_hist_path + '/' + venue_path + symbol_path+'/'+symbol+'??_[12]*.csv*')

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
            raise ValueError('time overlap! ' + '%s(%s)>%s(%s)'%(des0,f0,dss0,f1))

    if check_only :
        print year, ': ', len(fn0), ' files'
        return
    num_col=8 # adding spd vol, last_trd_time, last_close_px
    bar_lr=[]
    td_arr = []
    col_arr = []
    if len(fn0) == 0 :
        return bar_lr
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
        b=bar_by_file(f)
        ba, td, col = write_daily_bar(symbol,b,bar_sec=bar_sec, old_cl_repo=old_cl_repo)
        bar_lr += ba  # appending daily bars
        td_arr += td
        col_arr += col

    return bar_lr, td_arr, col_arr

def gen_bar(symbol, year_s=1998, year_e=2018, check_only=False, repo=None, kdb_hist_path = '/cygdrive/e/kdb', old_cl_repo = None, bar_sec=5) :
    ba=[]
    td=[]
    col=[]
    years=np.arange(year_s, year_e+1)
    for y in years :
        try :
            barlr, td_arr, col_arr=gen_bar0(symbol,str(y),check_only=check_only, kdb_hist_path = kdb_hist_path, bar_sec=bar_sec, old_cl_repo=old_cl_repo)
            if len(barlr) > 0 :
                ba+=barlr
                td+=td_arr
                col+=col_arr
        except :
            traceback.print_exc()
            print 'problem getting ', y, ', continue...'

    if check_only :
        return

    if repo is not None :
        repo.update(ba, td, col, bar_sec)
    return ba, td, col

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
