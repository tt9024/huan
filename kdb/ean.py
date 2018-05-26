import numpy as np
import matplotlib.pylab as pl
import l1_reader as l1d
import scipy.stats
import copy
import multiprocessing as mp
import pdb

def get_eia_api(repo='cl_number.npz', if_plot=False):
    v=np.load('cl_number.npz')['eia_api'].item()
    eia_v=v['eia']['v']
    eia_dt=v['eia']['dt']
    api_v=v['api']['v']
    api_dt=v['api']['dt']

    # plotting stock change number
    if if_plot: 
        fig=pl.figure()
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(eia_dt,eia_v[:,2],'rx',label='stock')
        ax1.plot(eia_dt,eia_v[:,1],'bx',label='eia_forecast')
        ax1.plot(api_dt,api_v[:,2],'r.',label='api_survey')
        ax1.plot(api_dt,api_v[:,1],'b.',label='api_forecast')
        ax1.grid() ; ax1.legend(loc='upper left')

    # get a cumsum to reveal actual stock number
    stock=np.cumsum(eia_v[:,2])
    stock_eiaf=np.r_[eia_v[0,1], stock[:-1]+eia_v[1:,1]]
    if if_plot: 
        ax2=fig.add_subplot(2,1,2,sharex=ax1)
        ax2.plot(eia_dt, stock, 'rx-',label='stock')
        ax2.plot(eia_dt, stock_eiaf, 'bx-',label='eia_forecat')

    # api forecast
    ix0=np.searchsorted(eia_dt, api_dt[0])-1
    ix=np.nonzero(api_dt>eia_dt[-1])[0]
    if len(ix) > 0 :
        ix=ix[0]
    else :
        ix=len(api_dt)
    ix0=np.searchsorted(eia_dt[1:], api_dt[:ix])
    stock_apif=stock[ix0]+api_v[:ix,1]
    stock_api=stock[ix0]+api_v[:ix,2]
    if if_plot:
        api_dt=api_dt[:ix]
        ax2.plot(api_dt,stock_api,'r.-',label='api survey')
        ax2.plot(api_dt,stock_apif,'b.-',label='api forecast')
        ax2.grid(); ax2.legend(loc='upper left')

    return eia_dt, stock, stock_eiaf, api_dt, stock_api, stock_apif

def f3(x,dn='dweibull') :
    """
    possible dn could be 'norm', 'cauchy'
    """
    d=getattr(scipy.stats,dn)
    param = d.fit(x)
    if dn=='dweibull' or dn=='dgamma' or dn=='chi2' :
        # hack for returning loc instead of shape
        return param[1]
    return param[0]

def get_mu(x0,wt_decay=None,std_th=3,bootstrap=True,dn='norm',smooth_beta=None) :
    """
    remove outliers and get the median
    To be improved by maximu likelihood
    """
    x=x0.copy()
    n=len(x)
    if wt_decay is not None:
        wt=l1d.getwt(n,wt_decay)
    else :
        wt=np.ones(n).astype(float)
    wt/=np.sum(wt)
    xm=np.dot(x,wt)
    xs=np.sqrt(np.dot((x-xm)**2,wt))
    # outlier
    x=outlier(x,xm,xs,in_th=std_th,out_th=std_th*4)

    #ix=np.nonzero(np.abs(x-xm) > xs*std_th)[0]
    #if len(ix) > 0 :
    #    x[ix]=xm+np.sign(x[ix]-xm)*(xs*std_th)
    if smooth_beta is not None :
        #x=tanh_smooth(x, xm, xs, beta=smooth_beta)
        x=tanh_smooth(x, x.mean(), x.std(), beta=smooth_beta)
    if bootstrap :
        mu=[]
        try_cnt = n*3/2
        for i in np.arange(try_cnt) :
            mu.append(np.mean(x[np.random.choice(n,n*2/3,p=wt)]))
        x=np.array(mu)

    return f3(x,dn=dn), np.std(x)

def auto_corr(x0, period, wt_decay=0.125,roll=None) :
    """
    x: 1d array to find seasonal intra-period auto-correlation
    period: the length of seasonal window, within which auto corr to be calulated
    return: 
        ac: 3d matrix. for each period, the 2-dim outer 
    """
    if roll is None :
        roll = period
    x=x0.copy()
    n=len(x0)
    ac=[]
    cnt=n/int(roll)
    xm=np.mean(x[:period])
    xs=np.std(x[:period])
    for c in np.arange(cnt) :
        if (c+1)*roll < period :
            continue
        # whiten the data according to the running
        # mean/std
        #x0=x[c*period:(c+1)*period]
        x0=x[(c+1)*roll-period:(c+1)*roll]
        x0-=xm
        x0/=xs
        # update for the next year
        xm=(1-wt_decay)*xm+wt_decay*np.mean(x0)
        xs=(1-wt_decay)*xs+wt_decay*np.std(x0)
        # consider a local smoothing at x0

        xc=np.outer(x0, x0)
        ac.append(xc.copy())
    ac=np.array(ac)  # ac is a 3d matrix
    return ac

def est_var(ac, wt_decay=0.125,bootstrap=True,dn='norm',std_th=3,smooth_beta=None) :
    """
    ac: 3-d matrix of raw auto-covariance matrix [period_cnt, elements, elements], i.e. [14, 52, 52], period is 52
    wt_decay: proximity decay favoring more to recent. [0,1], increasing in favoring more recent
    bootstrap: use resampling to estimate the var
    std_th: used to remove the outliers for the same element (week) in period_cnt, max_number is
            std_th*4 times the standard deviantion
    smooth_beta: tanh smooth the same element in period_cnt, beta is a scale of std, max_number is beta*std
    return :
        vmu, the estimated variance for each element in period, i.e. (week in the year)
        vstd, the std of the estimation
        va, the 2-d array with [period_cnt, elements], elements are diagonal of ac
    """
    # variance from ac
    N,P,P0=ac.shape
    assert P==P0, 'ac shape wrong'
    va=[]
    for ac0 in ac :
        va.append(ac0[np.arange(P),np.arange(P)])
    va=np.array(va)

    # rewrite get_mu, too import to ignore
    # for each period find a var
    vmu=[]
    vstd=[]
    for v0 in va.T :
        m,s=get_mu(v0,wt_decay=wt_decay,bootstrap=bootstrap,dn=dn,std_th=std_th,smooth_beta=smooth_beta)
        vmu.append(m)
        vstd.append(s)
    # smooth the vmu at first order, weighted on vstd
    vmu=np.array(vmu)
    vstd=np.array(vstd)
    return vmu, vstd, va

def est_diag_org(ac,period,offset,wt_decay=0.125,bootstrap=True,dn='norm',std_th=3,smooth_beta=None) :
    """
    estimate the off-diagnoal covariance
    ac: 3-d matrix of raw auto-covariance matrix [period_cnt, elements, elements], i.e. [13,104,104]
    period: length of off-diagnal covariance vector to estimate, i.e. 52
    offset: location of the off-diagnoal covariance vector to estimate, i.e. 1, the 
            first off-diagnal elements in ac 
    wt_decay: proximity decay favoring more to recent. [0,1], increasing in favoring more recent
    bootstrap: use resampling to estimate the var
    std_th: used to remove the outliers for the same element (week) in period_cnt, max_number is
            std_th*4 times the standard deviantion
    smooth_beta: tanh smooth the same element in period_cnt, beta is a scale of std, max_number is beta*std
    return :
        cvmu, length period of the estimated covariance at off-diagnoal offset
        cvstd, the std of the estimation
        cva, the 2-d array with [period_cnt, period], raw elements used in estimation
    """
    # covariance from ac
    N,P,P0=ac.shape
    assert P==P0, 'ac shape wrong'
    assert P/period*period==P, 'ac lookback not multiple of period'
    assert offset < P, 'offset not possible, bigger than matrix'
    assert offset > 0, 'non-positive offset'
    cva=[]
    for ac0 in ac :
        l=period
        if offset >= period :
            l=P-offset
        cva.append(ac0[np.arange(l)+offset,np.arange(l)])
    cva=np.array(cva)

    cvmu=[]
    cvstd=[]
    for cv0 in cva.T :
        m,s=get_mu(cv0,wt_decay=wt_decay,bootstrap=bootstrap,dn=dn,std_th=std_th,smooth_beta=smooth_beta)
        cvmu.append(m)
        cvstd.append(s)
    cvmu=np.array(cvmu)
    cvstd=np.array(cvstd)
    return cvmu, cvstd, cva

def est_diag(ac,period,offset,wt_decay=0.125,bootstrap=True,dn='norm',std_th=3,smooth_beta=None) :
    """
    estimate the off-diagnoal covariance
    ac: 3-d matrix of raw auto-covariance matrix [period_cnt, elements, elements], i.e. [13,104,104]
    period: length of off-diagnal covariance vector to estimate, i.e. 52
    offset: location of the off-diagnoal covariance vector to estimate, i.e. 1, the 
            first off-diagnal elements in ac 
    wt_decay: proximity decay favoring more to recent. [0,1], increasing in favoring more recent
    bootstrap: use resampling to estimate the var
    std_th: used to remove the outliers for the same element (week) in period_cnt, max_number is
            std_th*4 times the standard deviantion
    smooth_beta: tanh smooth the same element in period_cnt, beta is a scale of std, max_number is beta*std
    return :
        cvmu, length period of the estimated covariance at off-diagnoal offset
        cvstd, the std of the estimation
        cva, the 2-d array with [period_cnt, period], raw elements used in estimation
    """
    # covariance from ac
    N,P,P0=ac.shape
    assert P==P0, 'ac shape wrong'
    assert P/period*period==P, 'ac lookback not multiple of period'
    assert offset < P, 'offset not possible, bigger than matrix'
    assert offset > 0, 'non-positive offset'
    cva=[]
    for ac0 in ac :
        l=period
        #if offset >= period :
        if offset >= P-period :
            l=P-offset
        #l=P-offset
        cva.append(ac0[np.arange(l)+offset,np.arange(l)])
    cva=np.array(cva)

    cvmu=[]
    cvstd=[]
    for cv0 in cva.T :
        m,s=get_mu(cv0,wt_decay=wt_decay,bootstrap=bootstrap,dn=dn,std_th=std_th,smooth_beta=smooth_beta)
        cvmu.append(m)
        cvstd.append(s)
    cvmu=np.array(cvmu)
    cvstd=np.array(cvstd)
    return cvmu, cvstd, cva

def kernel_smooth(x, hu, wt=[], ss=1.0) :
    """
    x: the data
    hu: gaussian kernel width. typically, set to 2 for monthly,
        or 8 for quartly
    wt:  credibility of each point, the first elements is used against the x[0]
    ss:  the step size. between (0, 1.0], ratio of change to apply
    return:
        the smoothed series
    """
    # get the kernel weights
    n=len(x)
    x0=[]
    if len(wt) == 0 :
        wt=np.ones(n).astype(float)
    else :
        if len(wt) != n :
            # try to tile
            k=len(wt)
            if n/k*k != n :
                print 'wt not same length as x', len(wt)
                raise ValueError('wt not same length as x')
            wt=np.tile(wt,n/k)
    wt/=np.sum(wt)
    #wt_ret=[]
    for i, w0 in enumerate(wt) :
        wt0=scipy.stats.norm.pdf(np.arange(n)-i,0,hu)*w0
        wt0/=np.sum(wt0)
        #wt_ret.append(wt0.copy())
        #x0.append(np.dot(x,wt0))
        xx=np.dot(x,wt0)-x[i]
        x0.append( x[i]+xx*ss )
    return np.array(x0)

def diff_smooth(x, diff_order, smooth_func) :
    """
    x: data to be smoothed
    diff_order: 0 is the data itself, k (k>0) is the
                order of differentiation to be smoothed
    smooth_func: a smooth func that takes 1d array x and
                 returns a smoothed func.
    return :
            smoothed x, at given differentiation order
    """
    n=len(x)
    if n < 2*(diff_order+1) + 1 : 
        return x.copy()

    x0=np.r_[x,x,x]
    xd=[x0]
    while diff_order > 0 :
        xx=xd[-1]
        xxd=np.r_[0, xx[1:]-xx[:-1]]
        xd.append(xxd.copy())
        diff_order-=1
    xs=smooth_func(xd[-1])

    for i in np.arange(len(xd)-1) + 2 :
        xs0=xd[-i][0] + np.cumsum(xs)
        # adjust mean
        X=np.vstack((np.ones(3*n).astype(float),xs0)).T
        beta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),xd[-i])
        xs=np.dot(X,beta)

    xs0=xs[n:-n]
    #adjust mean
    X=np.vstack((np.ones(n).astype(float),xs0)).T
    beta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),x)
    xs=np.dot(X,beta)
    return xs

def smooth_1d(x,sm_param=None) :
    """
    wrapper of the smoothing functions.
    x: 1d array to be smoothed
    sm_param: param dict.  keys:
              'hu': gaussian kernel width, as the std
              'd' : order of difference upon which smoothing is applied
                    0 is no diff
              'ss': step size, a ratio of the change to apply
                    between (0, 1], 1 is to apply all
              'iter': number of iterations to run. i.e. 100
              'wt' : weight, either empty (no effect) or an array 
                     of same size with x credibility of each x. 
                     wt[0] is applied to x[0], etc

    return :
        smoothed 1d series, copied
    """
    x0=x.copy()
    if sm_param is None :
        return x0
    hu=sm_param['hu']
    d=sm_param['d']
    ss=sm_param['ss']
    iter=sm_param['iter']
    if sm_param.has_key('wt') :
        wt=sm_param['wt'].copy()
    else :
        wt=[]
    sm=lambda a: kernel_smooth(a,hu,wt=wt,ss=ss)
    for i in np.arange(sm_param['iter']) :
        #print 'smooth iter', i
        x0=diff_smooth(x0,d,sm)
    return x0

def get_v_w(d0,d1,d2,std_th=3,smooth_beta=3,wt_decay=0.125,period=52) :
    """
    getting the variance estimation given the data d0, d1 and d2
    """
    vmu=[]
    vstd=[]
    for d in [d0, d1, d2] :
        ac0=auto_corr(d,period,wt_decay=wt_decay)
        for p in ['cauchy','dweibull'] :
            mu, sd, va=est_var(ac0,wt_decay=wt_decay,bootstrap=False,dn=p,std_th=std_th,smooth_beta=smooth_beta)
            vmu.append(mu.copy())
            vstd.append(sd.copy())
    v=np.array(vmu)
    vstd=np.array(vstd)
    wt=1.0/(1+vstd) * np.abs(v)
    wt[0,:]*=5; wt[1,:]*=5 #emphasis on d0
    wt0=wt/np.sum(wt,axis=0)
    v0=np.sum(v*wt0, axis=0)

    wt=np.ones(6).astype(float)
    wt[0]=3.0; wt[1]=3.0
    wt0=wt/np.sum(wt)
    sd0=np.dot(vstd.T,wt0)
    return v0, sd0

def mu_std_est(P,period,offset,darr,dnarr,std_th,smooth_beta,wt_decay,sm_param) :
    cmu=[]
    cstd=[]
    for d in darr :
        ac0=auto_corr(d,P,wt_decay=wt_decay,roll=period)
        for p in dnarr :
            mu, sd,va=est_diag(ac0,period,offset,wt_decay=wt_decay,bootstrap=False,dn=p,std_th=std_th,smooth_beta=smooth_beta)
            cmu.append(mu.copy())
            cstd.append(sd.copy())
    v=np.array(cmu)
    cstd=np.array(cstd)
    wt=1.0/(1+cstd) * np.sqrt(np.abs(v))  # tune down sensitivity to std
    wt[0,:]*=3; wt[1,:]*=3 #emphasis on d0
    wt0=wt/np.sum(wt,axis=0)
    v0=np.sum(v*wt0, axis=0)
    sd0=np.sum(cstd*wt0,axis=0)
    if sm_param is not None :
        v0=smooth_1d(v0,sm_param)
        sd0=smooth_1d(sd0,sm_param)
    return v0,sd0,offset

def get_covariance_matrix(d0,d1,std_th=3,smooth_beta=3,wt_decay=0.125,period=52,look_back_multiple=2,sm_param=None,dn=['cauchy','dweibull'],njobs=32) :
    """
    estimating the auto-covariance of 2*period, in the direction of diagnoals 
    of offset from 1 to P, set to 2 times of period
    d0,d1: the two series of data to be averaged in the results.  
           i.e. the eia stock and eia forecast
    std_th, smooth_beta,wt_decay: the threshold used in adjusting outlier 
            and smooth samples in each estimations. see est_var()
    period: the max-length of off-diagonal covariance. i.e. 52
    look_back_multiple: number of periods to look back i.e. 2, 
                        2*period observations are included in the prediction
    sm_param: None (default) or a dict of parameters to further 
              smooth the estimated off-diagonal
              covariance series, refer to smooth_1d()
              NOTE: this kernel smoothing applies to diagonal direction
                    This is not as effective for eia numbers as 
                    horizonal/vertical smoothing which can be applied 
                    at get_corr_matrix()
                    esitmation, see est_corr_matrix().  
              Warning: Without good reason, it should be None.
    njobs: number of maximum processes to start 
    return: 
        a matrix with off-diagonal auto-covariance estimations
    """
    P=look_back_multiple*period
    cvar=np.zeros((P,P)).astype(float)
    cvarstd=np.zeros((P,P)).astype(float)
    # need faster, use more cpu cores
    pool=mp.Pool(processes=njobs)
    results=[]
    for offset in np.arange(P-1) + 1 :
        results.append(pool.apply_async(mu_std_est,args=(P,period,offset,[d0,d1],dn,std_th,smooth_beta,wt_decay,sm_param,)))

    for res in results :
        v0,sd0,offset=res.get()
        print 'getting off-diagonal ', offset
        # store the off-diagnal in
        l=period
        #if offset >= period :
        if offset >= P-period :
            l=P-offset

        #co=offset
        #while co<P :
        cvar[np.arange(l)+offset,np.arange(l)]=v0.copy()
        cvarstd[np.arange(l)+offset,np.arange(l)]=sd0.copy()
        #    cvar[np.arange(l)+co,np.arange(l)+co-offset]=v0.copy()
        #    cvarstd[np.arange(l)+co,np.arange(l)+co-offset]=sd0.copy()
        #    co+=period
    pool.close()

    # assemble the auto-correlation matrix
    #print 'done, assembling'
    #for c0 in [cvar, cvarstd] :
    #    for ofs in np.arange(period-1) + 1:
    #        l=period-ofs
    #        c0[period+np.arange(l)+ofs,period+np.arange(l)]=c0[np.arange(l)+ofs,np.arange(l)].copy()

    # assemble the auto-correlation matrix
    print 'done, assembling'
    for c0 in [cvar, cvarstd] :
        k1=period
        while k1+period <= P :
            # copy
            for k in np.arange(period) :
                c0[k1+k:P, k1+k]=c0[k:P-k1,k].copy()
            k1+=period

    return cvar, cvarstd

def est_corr_matrix(cvar,cvarstd,v0,sm_param,njobs=32) :
    """
    given the covariance and variance estimations, normalize into correlation and apply
    2-d smoothing. 
    cvar, cvarstd: the estimated auto-covariance matrix, each with shape [P,P]
    v0: the estimated variance, each with shape [period]
                P is multiple of period, i.e. 2
    sm_param:   the parameter for smoothing. refer to smooth_1d
    return: 
         the estimated auto correlation matrix
    """
    cv0=cvar.copy()
    cvsd0=cvarstd.copy()
    [P,P0]=cv0.shape
    period=v0.shape[0]
    assert P/period*period==P, 'cvar length not multiple of variance length'

    # apply var
    K=P/period
    vv=np.tile(v0,K)
    #cv0[np.arange(P),np.arange(P)]=vv.copy()
    cv0/= np.sqrt(np.outer(vv,vv))
    cv00=cv0.copy()

    #smooth horizonal and vertically
    iter=sm_param['iter']
    smp=copy.deepcopy(sm_param)
    smp['iter']=1
    pool=mp.Pool(processes=njobs)
    for i0 in np.arange(iter) :
        print 'iteration ', i0, ' of ', iter
        results=[]
        # horizontal
        for p in np.arange(P-3)+3 :
            x0=cv0[p,0:p].copy()
            # wt according to the std
            wt0=cvsd0[p,0:p].copy()
            smp0=copy.deepcopy(smp)
            smp0['wt']=1.0/(1+wt0)
            results.append([p, pool.apply_async(smooth_1d, args=(x0,smp0,))])
        for res in results :
            p=res[0]
            x0=res[1].get()
            cv0[p,0:p]=x0.copy()
        results=[]
        # vertical
        for p in np.arange(P-3) :
            x0=cv0[p+1:,p].copy()
            # wt according to the std
            wt0=cvsd0[p+1:,p].copy()
            smp0=copy.deepcopy(smp)
            smp0['wt']=1.0/(1+wt0)
            results.append([p, pool.apply_async(smooth_1d, args=(x0,smp0,))])
        for res in results :
            p=res[0]
            x0=res[1].get()
            cv0[p+1:,p]=x0.copy()
    pool.close()

    return cv0, cv00

def tanh_smooth(x,mu,sd,beta=1) :
    return mu+np.tanh((x-mu)/(beta*sd))*(sd*beta)

def outlier(x0,mu,sd,in_th=3,out_th=6) :
    x=x0.copy()
    ix0=np.nonzero(np.abs(x)>sd*in_th)[0]
    if len(ix0) > 0 :
        outv=min(np.max(np.abs(x)), sd*out_th)
        if (outv > sd*in_th) :
            scl=outv-sd*in_th
            nv=np.tanh((np.abs(x[ix0])-sd*in_th)/scl)*scl + sd*in_th
            # don't increase the value
            ix=np.nonzero(nv>np.abs(x[ix0]))[0]
            if len(ix) > 0 :
                nv[ix]=np.abs(x[ix0[ix]])
            x[ix0]=nv*np.sign(x[ix0])
    return x

def coef_est (d0, vs, period, roll, clf, fit_sign=False) :
    """
    d0: all training data, i.e. the weekly eia change residual d0
    vs: the variance estimation. Length equals to roll
    period: the number of elements looking back, i.e. 104 (two years)
    roll: the number of elements moving forward for sample, i.e. 52 (1 year)
          period must be multiple of roll
    clf:  the classifier to fit(X,Y) and get clf.coef_
    fit_sign: if True, Y0=np.sign(Y0)
    return: 
        C: coefficient shape [roll, period] 
    """
    P=len(vs)
    assert P==roll, 'vs has different length with roll'
    assert int(period/float(roll)) == period/roll, 'period must be multiple of roll'
    cnt=period/roll
    vs0=np.tile(vs,cnt+1)

    N=len(d0)
    C=[]
    ix=0
    while ix < roll :
        X=[]
        Y=[]
        for k in np.arange(period+ix, N, roll) :
            X.append(d0[k-period:k])
            Y.append(d0[k])
        X0=np.array(X)
        Y0=np.array(Y)

        # scale with variance estimation
        X0/=vs0[ix:ix+period]
        if fit_sign :
            Y0 = np.sign(Y0)
        else :
            Y0/=vs0[ix]
        clf0=copy.deepcopy(clf)
        clf0.fit(X0,Y0)
        C.append(clf0.coef_.copy())
        ix+=1

    return np.array(C)


class EIA_387 :
    def __init__(self, repo='cl_number.npz') :
        """
        Class deal with modeling of the eia actual stock change, 
        published as an event usually on Wednesday 10:30am, 
        identified as Bloomberg US event 387
        Required also the data from API, usually published on
        Tuesday 16:30pm.
        """
        self.v=np.load('cl_number.npz')['eia_api'].item()
        self.eia_v=self.v['eia']['v']
        self.eia_dt=self.v['eia']['dt']
        self.api_v=self.v['api']['v']
        self.api_dt=self.v['api']['dt']
        self.set_api()
        self.set_eia()

    def set_api(lr_5m=None) :
        if lr_5m is not None:
            self.lr_5m=lr_5m.copy()

    def set_eia() :
        pass

def api_baseline(wbdict, repo='cl_number.npz') :
    v=np.load('cl_number.npz')['eia_api'].item()
    eia_v=v['eia']['v']
    api_v=v['api']['v']
    a1=api_v[:,2] # actual (survey)
    af=api_v[:,1] # forecast
    x1=eia_v[:,2] # actual
    xf=eia_v[:,1] # forecast
    ta=api_v[:,0] ; te=eia_v[:,0]
    wb=wbdict['wbar']
    tt=wb[:,:,0].ravel()
    et=min(te[-1],tt[-1])
    if et < ta[-1] :
        ix=np.searchsorted(ta, et)
        ta=ta[:ix]
        af=af[:ix]
        a1=a1[:ix]

    a1=np.r_[0,a1[:-1]] # not yet observed
    af=np.r_[0,af[:-1]]
    ix=np.searchsorted(tt, ta)
    y=wb[:,:,1].ravel()[ix+1]
    ix=np.searchsorted(te, ta)
    x1=x1[ix-1]
    xf=xf[ix]

    x=[]
    # so the series will be
    #a1,    d-1, abs
    x.append(a1.copy())
    x.append(np.r_[0, a1[:-1]])
    x.append(np.abs(a1))
    x.append(a1-x1)
    x.append(np.r_[0, x1[:-1]] - np.r_[0, a1[:-1]])
    x.append(np.abs(a1)-np.abs(x1))
    x.append(xf.copy())
    x.append(xf-af)
    x.append(xf-x1)
    #x.append(np.r_[0, xf[1:]-x1[1:]- (x1[1:]-x1[:-1])])
    x.append(af-a1)
    x.append(np.r_[0, (af-a1)[:-1]])
    x.append(x1-np.r_[0, xf[:-1]]-(a1-af))

    a1d1=np.r_[0, a1[:-1]]
    ad1=a1*a1d1
    ix=np.nonzero(ad1<0)[0]
    ad1[ix]=0
    ad1*=np.sign(a1)
    x.append(ad1.copy())

    a1d2=np.r_[0,0,a1[:-2]]
    ad2=a1*a1d2
    ix=np.nonzero(ad2<0)[0]
    ad2[ix]=0
    ad2*=ad1
    ad2=np.abs(ad2)*np.sign(a1)
    x.append(ad2.copy())
 
    x=np.array(x).T
    x-=np.mean(x,axis=0)
    x/=np.std(x,axis=0)
    return ta, y, a1, af, x1, xf,x

def M_N_from_ac_matrix(cv, period) :
    """
    M,N=M_N_from_ac_matrix(cv,period)
    getting M, N from auto-correlation matrix
    cv: shape [P,P] square matrix
    period: the period of cv, i.e. 52
    return:
        M: 3d matrix of shape [period, P-period-1, P-period-1]
           i.e. for each period (first index), squre matrix M of shape
           [P-period-1, P-period-1]
           the first element in M is for the first target in period
        N: 2d matrix of shape [period, P-period-1] as array of targets
    """
    P,P0=cv.shape
    assert P==P0, 'cv not square'
    assert P/period*period==P, 'cv shape not multiple of period'
    M=[]
    N=[]
    l=P-period-1
    for p in np.arange(period) :
        m0=cv[-p-1-l:-p-1,-p-1-l:-p-1]
        n0=cv[-p-1,-p-1-l:-p-1]
        M.append(m0.copy())
        N.append(n0.copy())
    return np.array(M), np.array(N)

def get_L(d0,period,P) :
    ac0=[]
    L0=[]
    b0=[]
    yh0=[]
    for i in np.arange(26) :
        a0=auto_corr(d0[i:676+i],P,roll=period)
        a0=np.mean(a0, axis=0)
        a0[np.arange(P),np.arange(P)]+=0.00001
        L=np.linalg.cholesky(np.linalg.inv(a0))
        v=L[np.arange(P),np.arange(P)]
        v0=np.sqrt(np.outer(v,v))
        L/=v0
        L0.append(L.copy())
        b=[]
        for i in np.arange(period) :
            b.append(-L[-period+i,i:i+(P-period)])
        b0.append(np.array(b).copy())
        # out-of-sample

def get_C(lr5m,fday) :
    c=[]
    for i in np.arange(1320) :
        c0=[]
        for j in np.arange(1320-i-6)+6+i :
            if j-i > 96 :
                break
            c0.append([i,j,np.corrcoef(fday, np.sum(lr5m[:,i:j],axis=1))[0,1]])
        if len(c0) > 0 :
            c0=np.array(c0)
            ix=np.argsort(np.abs(c0[:,2]))
            c.append(c0[ix[-1],:])
    return np.array(c)


