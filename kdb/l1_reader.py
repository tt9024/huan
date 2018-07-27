import numpy as np
import os
import datetime
import l1
import pdb
import matplotlib.pylab as pl

### For a detailed description on the these columes, 
### refer to class DailyBar
repo_col={'utc':0, 'lr':1, 'vol':2, 'vbs':3, 'lrhl':4, 'vwap':5, 'ltt':6, 'lpx':7}
utcc=repo_col['utc']
lrc=repo_col['lr']
volc=repo_col['vol']
vbsc=repo_col['vbs']
lrhlc=repo_col['lrhl']
vwapc=repo_col['vwap']
lttc=repo_col['ltt']
lpxc=repo_col['lpx']
ALL_COL_RAW=len(repo_col.keys())

derived_col={'vol1':0, 'vol2':1, 'vol3':2,'vol4':3}
vol1c=derived_col['vol1'] + ALL_COL_RAW
vol2c=derived_col['vol2'] + ALL_COL_RAW
vol3c=derived_col['vol3'] + ALL_COL_RAW
vol4c=derived_col['vol4'] + ALL_COL_RAW
ALL_COL = ALL_COL_RAW + len(derived_col)
weekday=['mon','tue','wed','thu','fri','sat','sun']
def col_name(col) :
    if col<ALL_COL_RAW:
        return repo_col.keys()[np.nonzero(np.array(repo_col.values())==col)[0][0]]
    elif col<ALL_COL:
        return derived_col.keys()[np.nonzero(np.array(derived_col.values())==col-ALL_COL_RAW)[0]]
    raise ValueError('col ' + str(col) + ' not found!')

class DailyBar :
    """
    col of bar: 
    lr:   the log return of this bar observed in period of
          [utc-bar_sec, utc)
    vol:  the total trading volume, including implied 
          buy/sell, also include spread trades without
          matching with legs therefore no direction
    vbs:  buy volume - sell volume (all include implied)
    lrhl: log return of highest px - lowest px
    vwap: log return of vwap px
    ltt:  last trade time observed at utc
    lpx:  last price, as the latest trade price at utc
    derived col of bar:
    vol1: sum of lrhl+abs(lr) for each of the raw bar in bar minus abs(bar_lr)
    vol2: highest - lowest in cumsum of raw bar's lr
    vol3: std of raw lr in bar_sec
    vol4: mse of linear fit (with intercept) of all raw lr 
          within bar
    """
    def __init__(self, repo) :
        self.b=np.load(repo)['bar']
        self.bs=int(b[0][1][utcc]-b[0][0][utcc] + 0.5)
        utc0=[]
        totbar=0
        for b0 in b :
            utc0.append(b0[0][utcc])
            totbar += len(b0)
        self.utc0=np.array(utc0)
        print 'loaded ', totbar, ' bars in ', len(utc0), ' arrays. bar_sec ', self.bs

    def set_repo(self, b) :
        """
        repo format is in [bt,lr,vl,vbs,lrhl,vwap,ltt,lpx]
        bt: the bar time that observe the data
        lr: the log return of the bar just observed at bt as
            last trade price vs last trade price of previous bar
            
        vl: total volume, including buy/sell and spread trade that cannot be implied
        vbs: buy-sell
        lrhl: the logret on the highest trade price minus the logret of the lowest trade price
              within this bar
        vwap: the logret of the vwap price of this bar
        ltt: last trade time
        lpx: last trade price
        """
        self.b=b
        self.bs=int(self.b[0][1][utcc]-self.b[0][0][utcc] + 0.5)
        utc0=[]
        totbar=0
        for b0 in b :
            if len(b0) > 0 :
                utc0.append(b0[0][utcc])
                totbar += len(b0)
        self.utc0=np.array(utc0)
        print 'loaded ', totbar, ' bars in ', len(utc0), ' arrays. bar_sec ', self.bs

    def daily_idx(self,day) :
        """ 
        return a daily bar index of day
        starting from 18:00:00+self.bs(5 sec) of 
        previous day, ending at 17:00:00 of current 
        day each index has two elements indexing into
        self.b[i][j]
        """
        ti=l1.TradingDayIterator(day)
        if ti.yyyymmdd() != day :
            raise ValueError(day + ' not a trading day')
        # starting being 18:00:05
        utc0=float(l1.TradingDayIterator.local_dt_to_utc(ti.dt))
        utc_st=utc0-6*3600+self.bs
        utc_ed=utc0+17*3600
        i=np.searchsorted(self.utc0[1:], utc_st+1)
        i0=np.searchsorted(self.utc0[1:], utc_ed+1)
        assert i==i0, 'repo error, daily bar accross array ' + str(day) + ': '+str( [i,i0] )
        six=np.searchsorted(self.b[i][:,utcc],utc_st)
        eix=np.searchsorted(self.b[i][:,utcc],utc_ed) #eix is included
        assert eix > six, 'nothing found from repo on ' + day
        assert self.b[i][six,utcc]==utc_st, 'repo bar of start not found ' + day + ': ' + str(utc_st)
        assert self.b[i][eix,utcc]==utc_ed, 'repo bar of end not found ' + day + ': ' + str(utc_ed)
        return i, six, eix, day

    @staticmethod
    def fill_daily_bar_col(day, bar_sec, c) :
        """
        first bar starts at the previous day's 18:00+bar_sec, last bar ends at this day's last bar
        before or equal 17:00
        """
        SecPerDay=23*3600
        N=SecPerDay/bar_sec
        if c == utcc :
            u0=int(l1.TradingDayIterator.local_ymd_to_utc(day))-6*3600
            utc0=u0+bar_sec
            utc1=u0+N*bar_sec
            return np.arange(utc0,utc1+1,bar_sec)
        else :
            return np.zeros(N)

    def bar_by_idx(self,day_idx_arr,bar_sec,cols) :
        """
        returns bars from the daily index returned by
        daily_idx. 
        day_idx_arr: [[i, six, eix, day], ... ] 
        Then assemble the cols based on bar_sec
        bar_sec has to be multiple of self.bs from repo
        """
        bar_sec=int(bar_sec)
        assert bar_sec/self.bs*self.bs==bar_sec, 'bar_sec not multiple of ' + str(self.bs)
        bm=bar_sec/self.bs
        ca=[]  # list for all the 
        for i in np.arange(ALL_COL) :
            ca.append([])
        for i, six, eix, day in day_idx_arr :
            if i==-1 :
                # a missing day, filling in ones as place holder
                print 'filling zeros for missing day ', day
                for c in cols :
                    ca[c]=np.r_[ca[c],DailyBar.fill_daily_bar_col(day,bar_sec,c)]
                continue

            # since it's a daily bar, ix should
            # start at six-1 + bm, as six already
            # include a self.bs return
            ix=np.arange(-1,eix-six+1,bm)[1:]
            eix=ix[-1]+six # trim towards the end
            b0=self.b[i][six:eix+1,:]
            for c in cols :
                #pdb.set_trace()
                if c in [lrc, volc, vbsc, lrhlc] :
                    # needs aggregate
                    c0=np.r_[0,np.cumsum(b0[:,c])[ix]]
                    ca[c]=np.r_[ca[c], c0[1:]-c0[:-1]]
                elif c in[vwapc] :
                    # needs aggregate in abs
                    c0=np.r_[0,np.cumsum(np.abs(b0[:,c]))[ix]]
                    ca[c]=np.r_[ca[c], c0[1:]-c0[:-1]]
                elif c in [utcc, lttc] :
                    # needs to get the latest snap
                    ca[c]=np.r_[ca[c], b0[ix, c]]
                elif c in [lpxc] :
                    # needs to get the latest snapshot, but fill
                    # in zero at begining and ending
                    import pandas as pd
                    lpx=b0[ix,c]
                    ix0=np.nonzero(lpx==0)[0]
                    if len(ix0) > 0 :
                        lpx[ix0]=np.nan
                        df=pd.DataFrame(lpx)
                        df.fillna(method='ffill',inplace=True)
                        df.fillna(method='bfill',inplace=True)
                    ca[c]=np.r_[ca[c],lpx]

                elif c >= ALL_COL_RAW and c < ALL_COL :
                    # derived, needs process
                    ca[c]=np.r_[ca[c],self.get_derived(b0, ix, c)]
                else :
                    raise ValueError('unknow col ' + str(c))

        # got everything in ca
        ca0=[]
        for c in cols :
            ca0.append(ca[c])
        return np.vstack(ca0).T

    def get_bar(self, start_day, day_cnt, bar_sec, end_day='', cols=[utcc,lrc]) :
        """
        return bars for specified period, with bar period
        return index of multiday bar starting from
        start_day, running for day_cnt
        """
        if end_day!='' :
            print 'end_day not null, ignoring day_cnt: ', day_cnt
            ti=l1.TradingDayIterator(start_day)
            day_cnt=0
            day=ti.yyyymmdd()
            while day <= end_day :
                day_cnt+=1
                ti.next()
                day=ti.yyyymmdd()
        print 'got ', day_cnt, ' days from ', start_day, ' to ', end_day
        ti=l1.TradingDayIterator(start_day)
        day=ti.yyyymmdd()
        dc=0
        ixa=[]
        while dc < day_cnt :
            try :
                ixa.append(self.daily_idx(day))
            except  Exception as e :
                print str(e)
                print 'problem getting for the day ', day, ', continue'
                ixa.append((-1,0,0,day))
            ti.next()
            day=ti.yyyymmdd()
            dc+=1
        return self.bar_by_idx(ixa,bar_sec,cols)

    #### special colume handling code 
    def get_derived(self, b0, ix, c) :
        if c == vol1c :
            # sum of lrhl-lr for each raw bar minus abs(bar_lr)
            c0=b0[:,lrhlc]+np.abs(b0[:,lrc])
            # aggregate
            c0a=np.r_[0, np.cumsum(c0)[ix]]
            c0b=np.r_[0, np.cumsum(b0[:,lrc])[ix]]
            return c0a[1:]-c0a[:-1]-np.abs(c0b[1:]-c0b[:-1])
        elif c == vol2c :
            # highest - lowest in cumsum of lr
            c0=np.cumsum(np.vstack((np.zeros(len(ix)),b0[:,lrc].reshape((len(ix),ix[0]+1)).T)).T,axis=1)
            return np.max(c0,axis=1)-np.min(c0,axis=1)
        elif c == vol3c :
            # std of all raw lr in bar
            return np.std(b0[:,lrc].reshape((len(ix),ix[0]+1)),axis=1)
        elif c == vol4c :
            # mse of linear fit (with intcpt) of raw lr in bar
            k=ix[0]+1 # number of raw lr in a bar
            print k
            if k<3 :
                print 'no linear fitting for less than 3 points, setting mse to zero'
                return np.zeros(len(ix))
            x0=np.vstack((np.ones(k),np.arange(k))).T
            IH=np.eye(k)-np.dot(np.dot(x0,np.linalg.inv(np.dot(x0.T,x0))),x0.T)
            Y=b0[:,lrc].reshape((len(ix),ix[0]+1))
            return np.sqrt(np.diag(np.dot(np.dot(Y,IH),Y.T))/len(ix))

def get_weekly(dbar, symbol, start_day, end_day, bar_sec, cols=[utcc,lrc,volc,vbsc,vol1c]) :
    """
    dbar is the DailyBar object
    wbdict is a basic object holding weekly data as 3 dimensional array
    wbdict['wbar'] the 3D array of [week, bar_of_week, cols]
    where first 5 columns in cols has to be utcc, lrc, volc, vbsc, vol1c
    """
    ti=l1.TradingDayIterator(start_day)
    while ti.dt.weekday() != 0 :
        ti.next()
    day=ti.yyyymmdd()
    ti2=l1.TradingDayIterator(end_day)
    while ti2.dt.weekday() != 4 :
        ti2.prev()
    end_day=ti2.yyyymmdd()
    print 'getting from ', day , ' to ', end_day
    wbar=[]
    while day <= end_day :
        while ti.dt.weekday() != 4 :
            ti.next()
        eday=ti.yyyymmdd()
        wbar.append(dbar.get_bar(day,0,bar_sec,end_day=eday,cols=cols))
        while ti.dt.weekday() != 0 :
            ti.next()
        day=ti.yyyymmdd()
    wbar=np.array(wbar)
    wbdict={'wbar':wbar,'cols':cols}
    return wbdict

def getwt(tcnt,decay,scl=10) :
    x=np.linspace(-scl,scl,tcnt)
    wt=1.0/(1.0+np.exp(-decay*x))
    return wt

def plot_weekly_bar(wbdict,wt=[], wt_decay=0.1,title_str='') :
    """
    plot 6 subplots from default columns, assuming the first 5 colums always
    the [utcc,lrc,volc,vbsc, vol1c], other columns are ignored
    The bsec should be consistent with wbdict['bar']
    """
    b0=np.array(wbdict['wbar'])
    bsec=b0[0,1,0]-b0[0,0,0]
    tcnt=b0.shape[0]
    wt_str=''
    if len(wt)==0 :
        # use a slow decaying sigmoid curve
        wt=getwt(tcnt,wt_decay)
        wt_str='dcy('+str(wt_decay)+')'
    else :
        wt=wt[:tcnt].astype(float)
    wt/=np.sum(wt)
    mu=[]
    for i in np.arange(4) + 1:
        #mu.append(np.mean(b0[:,:,i],axis=0))
        mu.append(np.dot(b0[:,:,i].T,wt))
    mu=np.array(mu)
    #mu_lra=np.mean(np.abs(b0[:,:,1]),axis=0)
    mu_lra=np.dot(np.abs(b0[:,:,1]).T,wt)
    #sd=np.std(b0[:,:,3],axis=0) # std of vb-vs
    sd=np.sqrt(np.dot(((b0[:,:,3]-mu[2,:]).T)**2,wt))
    sdlr=np.sqrt(np.dot(((b0[:,:,1]-mu[0,:]).T)**2,wt))

    dt=[]
    # create one week of any days
    utc=DailyBar.fill_daily_bar_col('20160905',bsec,utcc)
    utc=np.r_[utc,DailyBar.fill_daily_bar_col('20160906',bsec,utcc)]
    utc=np.r_[utc,DailyBar.fill_daily_bar_col('20160907',bsec,utcc)]
    utc=np.r_[utc,DailyBar.fill_daily_bar_col('20160908',bsec,utcc)]
    utc=np.r_[utc,DailyBar.fill_daily_bar_col('20160909',bsec,utcc)]
    for u0 in utc :
        dt.append(datetime.datetime.fromtimestamp(u0))

    fig=pl.figure(); 
    ax1=fig.add_subplot(6,1,1); ax1.plot(dt,sdlr,'y-.',label='std(lr)');
    ax11=ax1.twinx() ; ax11.plot(dt,mu_lra,label='mu(|lr|)'); 
    ax2=fig.add_subplot(6,1,2,sharex=ax1); ax2.plot(dt,np.cumsum(mu[0,:]), label='cumsum(mu(lr))') ; 
    ax22=ax2.twinx();ax22.plot(dt,mu[0,:],'y.',label='mu(lr)') ; 
    ax3=fig.add_subplot(6,1,3,sharex=ax1) ; ax3.plot(dt,mu[1,:],label='mean(volume)') ; 

    ax4=fig.add_subplot(6,1,4,sharex=ax1) ; ax4.plot(dt, np.cumsum(mu[2,:]), label='cumsum(mean(vb-vs)')
    ax44=ax4.twinx() ; ax44.plot(dt,mu[2,:],'y.',label='mean(vb-vs)');

    ax5=fig.add_subplot(6,1,5,sharex=ax1);ax5.plot(dt,sd,label='std(vb-vs)'); 
    ax6=fig.add_subplot(6,1,6,sharex=ax1) ; ax6.plot(dt,mu[3,:],label='vol1(sum|lr|-|sum|lr||)') ; 
    ax1.set_title(title_str+' weekly stat of '+str(bsec)+' second bars ('+str(b0.shape[0])+' weeks) '+wt_str)

    ax1.legend(loc='lower left');ax11.legend(loc='lower right'); 
    ax2.legend(loc='lower left'); ax22.legend(loc='lower right') ; 
    ax4.legend(loc='lower left'); ax44.legend(loc='lower right') ; 
    ax3.legend();ax4.legend();ax5.legend();ax6.legend();
    ax1.grid(); ax2.grid() ; ax3.grid() ; ax4.grid() ; ax5.grid(); ax6.grid()
    return ax1,ax2,ax22,ax3,ax4,ax5,ax6

def plot_weekly(symbol, sday='19980101', eday='20180210', bsec=5, npz_repo='.', b=None, wt=[],wt_decay=0.1) :
    if b is None :
        b=np.load(npz_repo+'/'+symbol+'_bar_'+sday[:4]+'_'+eday[:4]+'_ext.npz')['bar']
    bar=DailyBar()
    bar.set_repo(b)
    wbdict=get_weekly(bar, symbol, sday, eday, bsec, cols=[utcc,lrc,volc,vbsc,vol1c])
    title_str=symbol+'_'+sday+'_'+eday
    plot_weekly_bar(wbdict, bar,title_str=title_str,wt=wt,wt_decay=wt_decay)
    return wbdict

def get_cols(bar, start_day, day_cnt, bar_sec, end_day='', cols=[utcc,lrc,volc,vbsc,vol1c]) :
    """
    This will get the cols w.r.t. time for the period given. 
    It is similar with get_weekly() but doesn't have to be in weekly and therefore return ca,
    i.e. two dim array indexed by bar number with each row represent cols specified.
    bar: a DailyBar object loaded with repo data, i.e. bar.set_repo(b) has been called
    start_day, end_day is in yyyymmdd string
    day_cnt is integer, including the start_day. 
            if end_day is specified day_cnt is ignored
    b is a loaded object from npz object.  Otherwise 
    return: ca, dt0
            ca: two dim array indexed by bar number with each row represent cols specified. 
            Note, the first cols[0] has to be utcc
            dt0: a one dimensional array of datetime object for each bar's ending time
    Internally, it calls the bar's get_bar function.
    """
    assert cols[0] == utcc, 'first column has to be utcc'
    if len(end_day) == 0 :
        # get the end day
        ti=l1.TradingDayIterator(start_day)
        ti.next_n_trade_day(day_cnt)
        end_day=ti.yyyymmdd()
    ca_arr=[]
    ti=l1.TradingDayIterator(start_day)
    day=ti.yyyymmdd()
    while day <= end_day :
        ca_arr.append(bar.get_bar(day, 1, bar_sec, cols=cols))
        ti.next()
        day=ti.yyyymmdd()
    ca=np.vstack(ca_arr)
    dt0=[]
    for utc0 in ca[:, 0] :
        dt0.append(datetime.datetime.fromtimestamp(utc0))
    return ca, dt0

###############################
# The following functions will plot the bars at the individual time of week, 
# for all the weeks in the bar repo (wbdict)
# For example, it will plot :
# 1. the bar of the week having highest std of volume_buy-volume_sell
# 2. bar at certain time, i.e. 10:30am or 16:30pm, etc
# When plotting it will plot the log ret, total volume and volume imbalance
# of the chosen bar from all the weeks
# it will also try to fit a distribution for the bars from the weeks.
# The goal is to check/valid the data, for correctness, trends, seasonalities, etc
# it is used in impluse response analysis
###############################
def plot_dist_weekly(wbdict, col, rank=0, rank_func='mean', dist=['dweibull','cauchy','dgamma','norm'],if_plot_dist=False,outlier_std_mul=10) :
    """
    wbdict: returned from get_weekly_bar, NOTE: wbdict=bar.get_weekly(), with the given bar_sec parameter.
    col: column name, i.e. lrc or volc
    rank_func: either 'mean' or 'std', or a function takes a 1d array and retuns scalar value for stats
    rank: plot the ith highest number from rank_func, 
          i.e. bar having highest mean lr for all bars in a week
    ret: 
        plot the lr of the chosen bar within a week for all weeks. 
        fitted parameters of the distribution 
    """
    i=np.searchsorted(wbdict['cols'],col)
    assert wbdict['cols'][i]==col, 'columne '+col+' not found in bar: ' + str(wbdict['cols'])
    v=wbdict['wbar'][:,:,i]
    std_th=np.std(v,axis=0)*outlier_std_mul
    ixout=np.nonzero(np.abs(v)-std_th>0)
    if len(ixout) > 0 :
        print 'removing ', len(ixout), ' outliers more than ', std_th, ', i.e. more than ', outlier_std_mul, ' times away from std'
        v=wbdict['wbar'][:,:,i].copy()
        v[ixout[0],ixout[1]]=np.sign(v[ixout])*std_th[ixout[1]]
    if rank_func=='mean' :
        v0=np.mean(v,axis=0)
    elif rank_func=='std' :
        v0=np.std(v,axis=0)
    else :
        v0=rank_func(v)
    ix=np.argsort(v0)[::-1][rank]
    param_str='chosen by col(%s),rank_func(%s),rank(%d)'%(col_name(col),str(rank_func),rank)
    plot_dist_weekly_by_ix(wbdict,ix,dist=dist,if_plot_dist=if_plot_dist,param_str=param_str)
    return ix

def plot_dist_weekly_by_utc(wbdict, weekday, hhmmss, param_str='', if_plot_dist=False) :
    """
    find the ix that corresponding to hhmmss of the weekday. 
    wbdict: returned by bar.get_weekly()
    weekday: integer from datetime.weekday(), mon is 0, sun is 6, etc
    hhmmss: string of 103000, in '%H%M%S' format
    """
    assert weekday < 7, 'weekday has to be less than 7'
    utc0=wbdict['wbar'][0,0,0]
    dt0=datetime.datetime.fromtimestamp(utc0)
    ti=l1.TradingDayIterator(dt0.strftime('%Y%m%d'), adj_start=False)
    dc=0
    while ti.dt.weekday() != weekday and dc<7:
        ti.next()
        dc+=1
    assert ti.dt.weekday() == weekday, 'weekday ' + str(weekday) + ' not found!'
    dtstr=ti.yyyymmdd()+hhmmss
    utc1=int(l1.TradingDayIterator.local_dt_to_utc(datetime.datetime.strptime(dtstr,'%Y%m%d%H%M%S')))
    ix=np.searchsorted(wbdict['wbar'][0,:,0].astype(int),utc1)
    utc2=wbdict['wbar'][0,ix,0]
    dt2=datetime.datetime.fromtimestamp(utc2)
    assert dt2.strftime('%H%M%S')==hhmmss and dt2.weekday()==weekday, hhmmss+' not found on weekday ' + str(weekday)
    plot_dist_weekly_by_ix(wbdict,ix,if_plot_dist=if_plot_dist,param_str=param_str)
    return ix

def plot_dist(lr, dist=['dweibull','cauchy','dgamma','norm'],title_str='') :
    """
    fit the given lr with scipy.stats distributions
    """
    import f0
    n=len(dist)
    fig=pl.figure()
    pl.title(title_str)
    for d in np.arange(n) :
        ax=fig.add_subplot(n,2,2*d+1)
        ax2=fig.add_subplot(n,2,2*d+2)
        f0.f3(lr,dist[d],True,ax=ax2,ax2=ax)

def plot_dist_weekly_by_ix(wbdict, ix, dist=['dweibull','cauchy','dgamma','norm'],if_plot_dist=False, param_str='') :
    lrcol=np.searchsorted (wbdict['cols'],lrc)
    volcol=np.searchsorted(wbdict['cols'],volc)
    vbscol=np.searchsorted(wbdict['cols'],vbsc)
    utccol=np.searchsorted(wbdict['cols'],utcc)
    v=wbdict['wbar'][:,ix,lrcol]
    dt=datetime.datetime.fromtimestamp(wbdict['wbar'][0,ix,0])
    bsec=wbdict['wbar'][0,1,0]-wbdict['wbar'][0,0,0]
    dtstr='%s[%02d:%02d:%02d]'%(weekday[dt.weekday()],dt.hour,dt.minute,dt.second)
    title_str='%dsec bar observed at %s'%(bsec,dtstr)
    if len(param_str) > 0 :
        title_str+='\nparam:[%s]'%(param_str)
    if if_plot_dist:
        plot_dist(v,dist=dist,title_str=title_str)
    # do another plot
    fig=pl.figure()
    dt=[]
    for dt0 in wbdict['wbar'][:,ix,utccol]:
        dt.append(datetime.datetime.fromtimestamp(dt0))
    lr=wbdict['wbar'][:,ix,lrcol]
    lram=np.mean(np.abs(lr))
    lrsd=np.std(lr)
    lbst='lr:mean_abs(%.5f)std(%.5f)'%(lram,lrsd)
    ax1=fig.add_subplot(3,1,1) ; ax1.plot(dt,lr,'.',label=lbst); ax1.legend()
    ax2=fig.add_subplot(3,1,2,sharex=ax1) ; ax2.plot(dt,wbdict['wbar'][:,ix,volcol],label='vol') ; ax2.legend()
    ax3=fig.add_subplot(3,1,3,sharex=ax1) ; ax3.plot(dt,wbdict['wbar'][:,ix,vbscol],label='vol buy-sel'); ax3.legend()
    ax1.grid(); ax2.grid(); ax3.grid()
    ax1.set_title(title_str)
    return ax1, ax2, ax3

def plot_corr_by_ix(wbdict, ix, cols, prev_lag, post_lag, ax_arr=None) :
    """
    plots a correlations w.r.t the weekly bar identified by ix for previous and afterwards
    weekly bars. 
    wbdict: the weekly bar dict returned from get_weekly
    ix:     the index of weekly bar, returned by plot_dist_weekly, or plot_dist_weekly_by_utc()
    cols:   columns to get correlations, such as [lrc, vbsc]
    prev_lag: number of previous bars to include in the plotting
    post_lat: number of afterwards bars to include in the plotting
    ax_arr:   optional array of ax to plot on. Length should be same as cols
    return:
        dt:  a human readable time stamps for trading hours of the days
        corrarr: shape [N,K], where N is len(cols) and K=prev_lag+post_lag+1
                 the correlation coefficients
                 NOTE: to enable plotting, the self correlation is set to be np.nan.
                 i.e. corrarr[:, prev_lag]=np.nan
    """
    corrarr=[]
    utc=wbdict['wbar'][-1,ix,0]
    bs=int(wbdict['wbar'][0,1,0]-wbdict['wbar'][0,0,0])
    dt=make_dt([utc,bs],prev_lag, post_lag)

    if ax_arr is None :
        ax_arr=[]
        fig=pl.figure()
        dt0=dt[prev_lag]
        ax=fig.add_subplot(len(cols),1,1)
        ax.set_title('correlation w.r.t ('+weekday[dt0.weekday()]+')'+dt0.strftime('%H:%M:%S')+' ('+ str(bs) + 'sec) bars')
        ax_arr.append(ax)
        for i in np.arange(len(cols)-1) :
            ax_arr.append(fig.add_subplot(len(cols),1,i+2,sharex=ax_arr[0]))
    else :
        assert len(ax_arr)==len(cols), 'ax_arr should be same length of cols'

    for c,(col,ax) in enumerate(zip(cols,ax_arr)) :
        cix=np.searchsorted(wbdict['cols'],col)
        assert wbdict['cols'][cix]==col, 'column not found from ' + str(wbdict['cols'])
        wb=wbdict['wbar'][:,:,cix]
        #utc=wbdict['wbar'][-1,ix,0]
        #bs=int(wbdict['wbar'][0,1,0]-wbdict['wbar'][0,0,0])
        #if len(dt) == 0 :
        #    dt=make_dt([utc,bs],prev_lag, post_lag)
        corr=[]
        val=wb[:,ix]
        n,k=wb.shape
        ix0=np.arange(n)*k+ix #ravel idx
        wb0=wb.ravel()
        for l in np.r_[-np.arange(prev_lag)[::-1]-1,np.arange(post_lag+1)] :
            if l==0 :
                corr.append(np.nan) # put a np.nan to the self-correlation
            else :
                # fix for over shooting a week
                #corr.append(np.corrcoef(val,wb[:,ix+l])[0,1])
                ix1=ix0+l
                val1=val
                # check overflow
                if ix1[0] < 0 :
                    ix10=np.nonzero(ix1>=0)[0]
                    val1=val1[ix10]
                    ix1=ix1[ix10]
                if ix1[-1] >= n*k :
                    ix10=np.nonzero(ix1<n*k)[0]
                    val1=val1[ix10]
                    ix1=ix1[ix10]
                corr.append(np.corrcoef(val1,wb0[ix1])[0,1])

        corrarr.append(np.array(corr))
        ax.plot(dt, corr, '.-', label=col_name(col))
        ax.plot(dt[prev_lag], 0, 'rx', label='ref(%.5f)'%(np.mean(val)))
        ax.grid() ; ax.legend()
    return dt, np.array(corrarr)

def plot_corr_by_evv(wbdict, ev_v, cols, prev_lag, post_lag, ax_arr=None) :
    """
    Same as plot_corr_by_ix(), except that it plots a correlations 
    w.r.t the specified weekly val, such as surprise, or buy/sell inbalance
    wbdict: the weekly bar dict returned from get_weekly
    ev_v: the event item stored in the npz file. 
          v=np.load('cl_number.npz')['eia_api'].item()
          ev_v=v['eia']['v']
          where ev_v has the shape of [N, 3]
          N is number of events
          column 0: the utc of each event
          column 1: the forecast number of the event
          column 2: the actual number of the event
          so the surprise would be calculated by ev_v[:,2]-ev_v[:,1]
    cols:   columns to get correlations, such as [lrc, vbsc]
    prev_lag: number of previous bars to include in the plotting
    post_lat: number of afterwards bars to include in the plotting
    ax_arr: an optional array of ax to plot cols on. Same length with cols
    return:
        dt:  a human readable time stamps for trading hours of the days
        corrarr: shape [N,K], where N is len(cols) and K=prev_lag+post_lag+1
                 the correlation coefficients
                 NOTE: to enable plotting, the self correlation is set to be np.nan.
                 i.e. corrarr[:, prev_lag]=np.nan
    """
    val=ev_v[:,2]-ev_v[:,1] # the weekly surprise number: actual minus forecast
    utc=ev_v[:,0] # the utc for each element of val
    return plot_corr_by_val(wbdict,val,utc,cols,prev_lag,post_lag,ax_arr=ax_arr)

def plot_corr_by_val(wbdict, val, utc, cols, prev_lag, post_lag, ax_arr=None) :
    # getting the weekly ix for the utc, all in ravel
    wbutc=wbdict['wbar'][:,:,0].ravel()
    ix0 = np.clip(np.searchsorted(wbutc,utc),0,len(wbutc)-1)
    ix00=np.nonzero(wbutc[ix0]==utc)[0]
    ix0=ix0[ix00]  # in ravel of 'wbar'
    utc=utc[ix00]
    val=val[ix00]
    bs=int(wbdict['wbar'][0,1,0]-wbdict['wbar'][0,0,0])

    dt=make_dt([utc[-1],bs],prev_lag, post_lag)

    corrarr=[]
    if ax_arr is None :
        ax_arr=[]
        fig=pl.figure()
        dt0=dt[prev_lag]
        ax=fig.add_subplot(len(cols),1,1)
        ax.set_title('correlation w.r.t surprise at ('+weekday[dt0.weekday()]+')'+dt0.strftime('%H:%M:%S')+' ('+ str(bs) + 'sec) bars')
        ax_arr.append(ax)
        for i in np.arange(len(cols)-1) :
            ax_arr.append(fig.add_subplot(len(cols),1,i+2,sharex=ax_arr[0]))
    else :
        assert len(ax_arr)==len(cols), 'ax_arr should be same length of cols'

    for c, (col,ax) in enumerate(zip(cols,ax_arr)) :
        cix=np.searchsorted(wbdict['cols'],col)
        assert wbdict['cols'][cix]==col, 'column not found from ' + str(wbdict['cols'])
        wb=wbdict['wbar'][:,:,cix].ravel()
        nk=len(wb)
        corr=[]
        for l in np.r_[-np.arange(prev_lag)[::-1]-1,np.arange(post_lag+1)] :
            #if l==0 :
                #corr.append(np.nan) # put a np.nan to the self-correlation
            #    corr.append(np.nan) # put a np.nan to the self-correlation
            #else :
            if True :
                # checking for bound of ix0
                ix1=ix0+l
                val1=val
                # check overflow
                if ix1[0] < 0 :
                    ix10=np.nonzero(ix1>=0)[0]
                    val1=val1[ix10]
                    ix1=ix1[ix10]
                if ix1[-1] >= nk :
                    ix10=np.nonzero(ix1<nk)[0]
                    val1=val1[ix10]
                    ix1=ix1[ix10]
                corr.append(np.corrcoef(val1,wb[ix1])[0,1])
        corrarr.append(np.array(corr))
        ax.plot(dt, corr, '.-', label=col_name(col))
        ax.plot(dt[prev_lag], 0, 'rx', label='ref(%.5f)'%(np.mean(val)))
        ax.grid() ; ax.legend()
    return dt, np.array(corrarr)

############################
# the following are a set of functions to plot impulse response
# for crude number
############################
def ev_lr(ev_v,wbdict,bar_cnt,bar_cnt_prev=0,if_plot2d=True, if_plot3d=False) :
    """
    ev_v: the vector of event number to be comparing to.
          For example, for surprise of eia: 
          v=np.load('cl_number.npz')['eia_api'].item()
          ev_v=v['eia']['v']
          ev_v has format of [utc, forecast_number, actual_number]
    wbdict: weekly bar dict obtain by get_weekly()
            wbbdict=get_weekly(bar,'CL','19980101','20180210',5)
            where bar is the DailyBar object
            Note, assuming ['wbar'] has first column as utc, second as lr
    bar_cnt: number of bars ending at or after the event to return (plot)
    bar_cnt_prev: number of bars ending before the event to return (plot)
    return :
        ev_v: the events with rows that corresponding to the bar
        b_lr0: the bar ending at and after event time.  
               b_lr0[:,0] is the bar log return observed at the event time for each event
                          Note: the event hasn't published yet
               b_lr0[:,1] is the first bar observed after the event time
        b_lr1: the bar ending before the event time.
               b_lr1[:,-1] is the bar log return observed at 1 bar before the event time
        ix:    the index of event time into ev_v, with corresponding b_lr0 and b_lr1
    """
    v_utc=ev_v[:,0]
    v_sup=ev_v[:,2]-ev_v[:,1]

    W,B,C=wbdict['wbar'].shape
    b_utc=wbdict['wbar'][:,:,0].ravel()
    b_lr=wbdict['wbar'][:,:,1].ravel()
    N=len(b_utc)
    bix0=np.clip(np.searchsorted(b_utc,v_utc),0,N-1)
    ix=np.nonzero(b_utc[bix0]==v_utc)[0]
    assert len(ix) >0 ,'event time not found in wbdict'
    bix=bix0[ix]
    b_lr0=b_lr[np.clip(np.outer(bix,np.ones(bar_cnt).astype(int))+np.arange(bar_cnt),0,N-1)]
    b_lr1=None
    if bar_cnt_prev > 0 :
        b_lr1=b_lr[np.clip(np.outer(bix,np.ones(bar_cnt_prev).astype(int))-(np.arange(bar_cnt_prev)[::-1]+1),0,N-1)]
    if if_plot3d:
        plot_ev_lr_3d(v_sup[ix],b_lr0,b_lr1)
    if if_plot2d:
        bsec=b_utc[1]-b_utc[0]
        evtutc_bsec=[ev_v[0,0],bsec]
        plot_ev_lr_2d(v_sup[ix],b_lr0,b_lr1,evtutc_bsec=evtutc_bsec)

    return ev_v[ix,:], b_lr0, b_lr1, ix

def plot_ev_lr_3d(ev_surprise,lr_arr, lr_arr_prev=None, evtutc_bsec=None) :
    """
    ev_surprise: 1d array of length N of "surprise" from event
    lr_arr: shape [N, K], lr series for each event, lr_arr[0] is the lr observed at event time
    lr_arr_prev: shape[N, K1], lr series before event.  lr_arr_prev[:,-1] is the last bar before
                 the event time.  i.e. lr_arr_prev[:,-1] is the lr of [et-2*bar_sec,et-bar_sec]
    evtutc_bsec: 2 element list of [utc, bsec], where 
                 utc is the utc of a typical event time. i.e. ev[:,0][-1]
                 bsec is the bar period in second.  They are used 
                 plot the date time of a week. 
                 if None then x-axis will be in bars
    The plotted lr series will have 0 at the event time. 
    """
    n,k=lr_arr.shape
    assert ev_surprise.shape[-1] == n, 'shape mismatch!'
    # normalize ev_surprise
    nk=np.sqrt(np.dot(ev_surprise, ev_surprise)/n)
    evs0=ev_surprise
    evs0=(evs0-evs0.min()+0.1)/(evs0.max()-evs0.min()+0.2)
    k1=0
    if lr_arr_prev is not None:
        k1=lr_arr_prev.shape[1]
    cs0,dt_arr=mergelr(lr_arr,lr_arr_prev,evtutc_bsec)

    from mpl_toolkits.mplot3d import Axes3D, axes3d
    from matplotlib import cm
    fig=pl.figure()
    ax=fig.gca(projection='3d')
    # NOT SURE why can't I plot as surface or wireframe?
    #X=np.outer(evs0,np.ones(k))
    #Y=np.outer(np.arange(k),np.ones(n)).T
    #Z=np.cumsum(lr_arr,axis=1)
    #ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1)
    #surf=ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0)
    #fig.colorbar(surf,shrink=0.5,aspect=5)
    # draw n 3d lines

    ## plotting the event time and afterwards
    cm=pl.get_cmap()
    for e0,cs in zip(evs0, cs0) :
        #pdb.set_trace()
        ax.plot(np.ones(k+1)*e0-0.5, dt_arr[-k-1:], cs[-k-1:],c=cm(e0))

    ## if previous lr are given, plotting them as well
    if lr_arr_prev is not None: 
        fig=pl.figure()
        ax=fig.gca(projection='3d')
        for e0,cs in zip(evs0, cs0) :
            ax.plot(np.ones(k1)*e0-0.5, dt_arr[:k1], cs[:k1],c=cm(e0))

        # And plotting everything all together
        fig=pl.figure()
        ax=fig.gca(projection='3d')
        for e0,cs in zip(evs0, cs0) :
            ax.plot(np.ones(k+k1+1)*e0-0.5, dt_arr, cs,c=cm(e0))

def plot_ev_lr_2d(ev_surprise, lr_arr,lr_arr_prev=None,evtutc_bsec=None,wt_decay=None,ax=None) :
    """
    refer to plot_ev_lr_3d
    evtutc_bsec: 2 element list of [utc, bsec], where 
          utc is the utc of a typical event time. i.e. ev[:,0][-1]
          bsec is the bar period in second.  They are used 
          plot the date time of a week. 
          if None then x-axis will be in bars

    The plotting will be plot the cumsum of lr for different weighting schemes.
    the cumsum of lr series is normalized at the bar second. refer to ev_lr_3d
    for details

    See also ev_lr() for usage
    """
    wt=[]
    lbl=[]
    n=len(ev_surprise)
    # make 3 weights:
    # 1. flat weight
    wt1=np.ones(n).astype(float)
    wt.append(wt1)
    lbl.append('avg')
    # 2. sign of ev_surprise
    wt2=np.sign(ev_surprise).astype(float)
    wt.append(wt2)
    lbl.append('sign avg')
    # 3. value of ev_surprise
    wt3=ev_surprise.copy()
    wt3/=np.mean(np.abs(wt3))
    wt.append(wt3)
    lbl.append('wt avg')
    cs0, dt_arr=mergelr(lr_arr,lr_arr_prev,evtutc_bsec)

    # get a decay weight if given
    if wt_decay is not None :
        dcy=getwt(n,wt_decay)
        dcy/=np.mean(np.abs(dcy))
    else :
        dcy=np.ones(n)

    if ax is None:
        fig=pl.figure()
        ax=fig.add_subplot(1,1,1)
    for w0, l0 in zip(wt,lbl) :
        cs=np.dot(cs0.T,w0*dcy)/float(n)
        ax.plot(dt_arr,cs,'.-', label=l0)
    ax.legend() ; ax.grid()
    return wt

def mergelr(lr_arr, lr_arr_prev=None, evtutc_bsec=None) :
    """
    the lr cumsum series at event time is exactly 0
    lr_arr[:,0] is observed at event time
    lr_arr[:,1] is the first bar lr responding to the event
    lr_arr_prev[:,-1] is the observed at event time - bar time
    k1 is count of previous bars in lr_arr_prev
    """
    lr0=lr_arr
    n,k=lr_arr.shape
    k1=0
    if lr_arr_prev is not None :
        lr0=np.hstack((lr_arr_prev,lr0))
        k1=lr_arr_prev.shape[1]
    cs=np.cumsum(lr0,axis=1)
    cs0=(np.vstack((np.zeros(n),cs.T))-cs[:,k1]).T
    if evtutc_bsec is not None:
        u0,bsec=evtutc_bsec
        dt_arr=[]
        for kk in np.r_[-np.arange(k1+1)[::-1]-1,np.arange(k)] :
            dt0=datetime.datetime.fromtimestamp(u0+kk*bsec)
            dt_arr.append(dt0)

        # do an adjust for the dt 
        dt_arr=np.array(dt_arr)
        ddt=datetime.timedelta(0)
        for i in np.arange(k1)[::-1] :
            dt_arr[i]-=ddt
            if not l1.tradinghour(dt_arr[i]) :
                dt_arr[i]-=datetime.timedelta(0, 3600)
                ddt+=datetime.timedelta(0, 3600)
        ddt=datetime.timedelta(0)
        for i in np.arange(k)+1+k1 :
            dt_arr[i]+=ddt
            if not l1.tradinghour(dt_arr[i]) :
                dt_arr[i]+=datetime.timedelta(0, 3600)
                ddt+=datetime.timedelta(0, 3600)
    else :
        dt_arr=np.arange(k+k1+1)-(k1+1)
    return cs0, dt_arr

def make_dt(evtutc_bsec, prev_k, post_k) :
    """
    This will make an array of human readable datetime object based on utc
    with period of 17:00 to 18:00 removed
    evtutc_bsec:  [utc0, bar_sec]
    prev_k:  number of bars to include before utc0, not including utc0
    post_k:  number of bars to include after utc0, not including utc0
    returns: datetime object of length prev_k+1+post_k
             with the middle one being the local time of utc0. 
             The local time returned are adjusted to include gap between 17:00 to 18:00
    """
    k1=prev_k
    k=post_k+1
    u0,bsec=evtutc_bsec
    dt_arr=[]
    for kk in np.r_[-np.arange(k1)[::-1]-1,np.arange(k)] :
        dt0=datetime.datetime.fromtimestamp(u0+kk*bsec)
        dt_arr.append(dt0)

    # do an adjust for the dt 
    dt_arr=np.array(dt_arr)
    ddt=datetime.timedelta(0)
    for i in np.arange(k1)[::-1] :
        dt_arr[i]-=ddt
        if not l1.tradinghour(dt_arr[i]) :
            dt_arr[i]-=datetime.timedelta(0, 3600)
            ddt+=datetime.timedelta(0, 3600)
    ddt=datetime.timedelta(0)
    for i in np.arange(k)+k1 :
        dt_arr[i]+=ddt
        if not l1.tradinghour(dt_arr[i]) :
            dt_arr[i]+=datetime.timedelta(0, 3600)
            ddt+=datetime.timedelta(0, 3600)
    return np.array(dt_arr)
