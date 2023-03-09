import numpy as np
import scipy
import datetime
import multiprocessing as mp
import time
from matplotlib import pyplot as pl

def get_wb(fnpz='../data/wbdict_5m.npz') :
    wb=np.load(fnpz)
    wb.allow_pickle=True
    return wb.items()[0][1].item()['wbar'][2:,:,:]

def get_lr_dt(wb=None, fnpz=None) :
    if wb is None :
        wb=get_wb(fnpz)
    lr=wb[:,:,1]
    dt=[]
    for t0 in wb[0,:,0] :
        dt.append(datetime.datetime.fromtimestamp(t0))
    dt=np.array(dt)
    return lr, dt

def get_lrd(lr_week) :
    n,m=lr_week.shape
    db=m/5
    return lr_week.reshape((n*m/db,db))

dfn = ['norm','beta','gamma','dgamma','dweibull','cauchy','invgamma','invweibull','powerlaw','powerlognorm']
#dfn = ['norm']

def chisquare_test(x0,dn,param,bins=6) :
    """
    make each bin at least about 20 observations.  
    and at least 20 bins.  The more observations the better.
    That's important for CLT to work
    """
    n=len(x0)
    ixr=np.nonzero(np.abs(x0)<np.std(x0)*12)[0]
    x=x0[ixr].copy()
    x.sort()
    c=dn.cdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    cs=np.linspace(c[0],c[-1], np.floor((c[-1]-c[0])*bins))
    ix=np.searchsorted(c, cs[1:]-1e-12)
    cnt=ix-np.r_[0,ix[:-1]]
    E=(c[ix]-c[np.r_[0,ix[:-1]]])*n
    v,p=scipy.stats.chisquare(cnt,E,ddof=len(param)-1)
    return v,p

def chisquare_test_unstable(x, dn, param) :
    n=len(x)
    cnt,bv=np.histogram(x,bins=min(1000, max(n/10, 5)))
    c1=dn.cdf(np.r_[bv[1:-1], 1e+14], *param[:-2], loc=param[-2], scale=param[-1])
    c2=np.r_[0,c1[:-1]]

    #c1=dn.cdf(bv[1:], *param[:-2], loc=param[-2], scale=param[-1])
    #c2=dn.cdf(bv[:-1], *param[:-2], loc=param[-2], scale=param[-1])

    xmid=(bv[1:]+bv[:-1])/2.0
    pc=cnt
    E=((c1-c2)*n).astype(int)

    # remove 0 in E
    zix=np.nonzero(E)[0]+1
    zix0=np.r_[0,zix[:-1]]
    cspc=np.r_[0,np.cumsum(pc)]
    pc=cspc[zix]-cspc[zix0]
    csE =np.r_[0,np.cumsum(E)]
    E=csE[zix]-csE[zix0]

    # remove the tail stuffs
    pc=pc[1:-1].astype(float)
    E=E[1:-1].astype(float)
    v,p=scipy.stats.chisquare(pc,E)
    return v,p

def distfit(x) :
    """
    run with chisquare
    """
    pks=[]
    pchi=[]
    dfs=['dgamma','dweibull','cauchy','norm']
    #dfs=dfn

    for df in dfs :
        try :
            d=getattr(scipy.stats,df)
            param=d.fit(x)
            v,p=scipy.stats.kstest(x,df,args=param)
            pks.append(p)
            v,p=chisquare_test(x,d,param)
            pchi.append(p)
        except KeyboardInterrupt as e:
            return
        except :
            print ('problem fitting {}'.format( df))

    ps=np.array(pks) + np.array(pchi)
    ix=np.argsort(ps)[-1]
    print("{} {} {}".format(dfs[ix], pks[ix], pchi[ix]))

    return pks, pchi
    #for dn, pk, pc in zip(dfn, pks, pchi) :
    #    print dn, pk, pc

def bootstrap_qr0(lr,cnt=None,m0=None,ixa=None,need_var=False) :
    """
    select subset of columns in lr for qr, in case
    n<m.  Note the bootstrap is not strict as
    the order is enforced and no duplicate is allowed
    # consider using a thread pool
    """
    n,m=lr.shape
    if n>m :
        q,r=np.linalg.qr(lr)
        return q,r,None,None
    if m0 is None:
        m0=n-1
    if cnt is None :
        cnt = int(200.0 * (float(m)/float(m0)))
    q=np.zeros((n,m))
    q2=np.zeros((n,m))
    r=np.zeros((m,m))
    r2=np.zeros((m,m))
    for c in np.arange(cnt) :
        if ixa is not None :
            ix=ixa[c]
        else :
            ix=np.random.choice(m,m0,replace=False)
        ix.sort()
        q0,r0=np.linalg.qr(lr[:,ix])
        # need to fix the sign in case the diagonal is negative
        for i,x in enumerate(np.diag(r0)):
            if x<0 :
                r0[i,:]*=-1
        q[:,ix]+=q0
        q2[:,ix]+=q0**2
        r[ np.ix_(ix,ix)]+=r0
        r2[np.ix_(ix,ix)]+=r0**2

    q/=cnt
    r/=cnt
    if not need_var: 
        return q,r
    q2/=cnt
    r2/=cnt
    return q, r, np.sqrt(q2-q**2), np.sqrt(r2-r**2)

def bootstrap_qr(lr,cnt=None,m0=None,njobs=8,ixa0=None,need_var=False) :
    """
    select subset of columns in lr for qr, in case
    n<m.  Note the bootstrap is not strict as
    the order is enforced and no duplicate is allowed
    # consider using a thread pool
    """
    n,m=lr.shape
    if n>m :
        q,r=np.linalg.qr(lr)
        if not need_var :
            return q,r
    if m0 is None:
        m0=int(min(n-1,m*0.8))  # bootstrap size
    if cnt is None :
        cnt = int(80.0 * (float(m)/float(m0)))
    q=np.zeros((n,m))
    q2=np.zeros((n,m))
    r=np.zeros((m,m))
    r2=np.zeros((m,m))
    while True :
        try :
            pool=mp.Pool(processes=njobs)
            break
        except :
            print('problem with resource, sleep a while')
            time.sleep(5)

    results=[]
    ixa=[]
    for c in np.arange(cnt) :
        if ixa0 is not None:
            ix=ixa0[c]  
        else :
            ix=np.random.choice(m,m0,replace=False)
        ix.sort()
        results.append((c, ix.copy(), pool.apply_async(np.linalg.qr,args=(lr[:,ix].copy(),))))

        if (c+1)%njobs == 0 or c+1==cnt:
            for res0 in results :
                c0,ix,res=res0
                q0,r0=res.get()
                #q0,r0=np.linalg.qr(lr[:,ix])
                # need to fix the sign in case the diagonal is negative
                for i,x in enumerate(np.diag(r0)):
                    if x<0 :
                        r0[i,:]*=-1
                q[:,ix]+=q0
                q2[:,ix]+=q0**2
                r[ np.ix_(ix,ix)]+=r0
                r2[np.ix_(ix,ix)]+=(r0**2)
                ixa.append(ix.copy())

            results=[]
            #print('iteration {}'.format(c))
    pool.close()

    q/=cnt
    r/=cnt
    if not need_var:
        return q,r
    q2/=cnt
    r2/=cnt
    #return q,r,q2,r2
    return q, r, np.sqrt(q2-q**2), np.sqrt(r2-r**2)

def mergelr(lr, frac, dt=None, ix0=None) :
    """
    lr0 = mergelr(lr, frac)
    lr shape [n,m], lr0 shape [n,m0]
    m0=frac*m, frac in [0,1]
    reduce the number of bars (sample
    points)
    """
    n,m=lr.shape
    mm = m*frac
    if ix0 is not None:
        ix=np.array(ix0).copy()
    else :
        ix = np.arange(m)
    lrc=np.vstack((np.zeros(n),np.cumsum(lr,axis=1).T)).T
    qr_score=[]
    qr_remove=[]
    while len(ix) > mm :
        #lr0=lrc[:,ix+1]-lrc[:,ix]
        lr0=lrc[:,ix+1]-lrc[:,np.r_[0,ix[:-1]+1]]
        vol=np.std(lr0,axis=0)
        n0=len(vol)

        print('currently {} removing one...'.format(n0))
        lr00=lr0/vol
        q0,r0=bootstrap_qr(lr00)

        #r0=np.abs(r0)
        #wt=(r0[0,0]-r0.diagonal())*vol
        r0=np.abs(r0)
        wt=vol

        print('tot vol {} noise {} '.format(np.sum(vol),np.sum(r0.diagonal()*vol)))
        r0*=wt
        r0=r0+r0.T
        snr=np.sum(r0,axis=1)-r0.diagonal()
        ix0_=np.argsort(snr)  # getting the least useful bar
        print('lowest signal contributors {} {}'.format( ix[ix0_[:10]], snr[ix0_[:10]] ))
        qr_score.append(snr[ix0_[0]])
        if dt is not None:
            print("{}".format(dt[ix[ix0_[:10]]]))

        ix1_list=[]
        for ix0 in ix0_[:10]:
        #ix0=ix0_[0]
            if ix0 == 0 :
                ix1_list.append(ix0)
            elif ix0 == n0-1 :
                ix1_list.append(ix0-1)
            else :
                ix1_list.append(ix0-1)
                ix1_list.append(ix0)

        ix1_list=np.unique(ix1_list)
        rscore=[]
        for ix1 in ix1_list:
            ix_ = np.delete(ix, ix1)
            #lr00 = lrc[:,ix_+1]-lrc[:,ix_]
            lr00 = lrc[:,ix_+1]-lrc[:,np.r_[0,ix_[:-1]+1]]
            q_,r_=bootstrap_qr(lr00,cnt=30)

            # note the total signal (weighted by vol)
            # is the goal

            r_=r_**2
            rd=np.abs(r_.diagonal())
            #rs0=np.sum(rd)*np.sqrt(np.dot(rd,rd))
            rs0=np.sum(np.sqrt(np.sum(r_,axis=0)-r_.diagonal()))
            rs1=np.sqrt(np.mean(r_.diagonal())-np.mean(rd)**2)

            #print('rs numbers: total signal, variance of signal, sharp {} {} {}'.format( rs0, rs1, rs0/rs1))
            rs0/=rs1
            #rs0=np.sum(rd)  # ignoring the variance in rs0 for now
            dtstr=''
            if dt is not None:
                dtstr=dt[ix[ix1]]
            #print('** Score {} {} {} '.format( ix1, dtstr, rs0))
            rscore.append(rs0)

        # find the ix with best score
        rsix=np.argsort(rscore)[::-1]
        print ('gains for removing the bar:')
        for rsix0 in rsix :
            ix1=ix1_list[rsix0]
            dtstr='' if dt is None else dt[ix[ix1]]
            print ('   {} {} {} '.format(dtstr, ix[ix1], rscore[rsix0]))

        # go with the biggest sharp
        ix1=ix1_list[rsix[0]]

        # remove ix1
        print('So, removing {} {} '.format( ix[ix1], '' if dt is None else dt[ix[ix1]]))
        qr_remove.append(ix[ix1])
        ix=np.delete(ix,ix1)
    return ix, qr_score, qr_remove

def dither(x, it_frac = 1, alpha=0.1) : 
    """
    n,m = shape(x) 
    dither along the m direction.  
    it_cnt = it_frac * m, i.e. local values will be
    averged within a neighorhood of it_cnt. 
    the higher it_frac, the larger the smooth neighborhood for
    each local value, therefore looks more smooth. 
    NOTE: it_fact > 0
    alpha doesn't seem to have any effect in the results
    """
    x0 = x.copy()
    reshape=False
    if len(x.shape) == 1 :
        reshape = True
        x0 = x0.reshape((1,len(x)))
    N = x0.shape[1]
    it_cnt = int(N*it_frac+0.5)
    k = min(it_cnt/2, 50)
    k = max(k,1)
    x0=np.vstack((x0[:,1+k:0:-1].T,x0.T,x0[:,-2:-3-k:-1].T)).T
    for i in np.arange(it_cnt) :
        d=(x0[:,:-2]+x0[:,2:])/2-x0[:,1:-1]
        x0[:,1:-1]+=(alpha*d)
    x0=x0[:, k+1:-k-1]
    if reshape :
        x0=x0.flatten()
    return x0

def wtD1(n) :
    pass

def wtD2(lrd, smooth_frac = 1) :
    # just do a dither
    # lrd is the (ndays,nperiod)
    # returns weight that is inverse to the D2 smoothed std

    n,m=lrd.shape
    lrd0 = np.vstack((np.zeros(n), lrd.T, np.zeros(n))).T
    d2s=dither( np.mean(np.abs((lrd0[:,:-2]+lrd0[:,2:])/2-lrd0[:,1:-1]), axis=0), \
               it_frac=smooth_frac)
    w = 1.0/d2s
    return w/np.sum(w)

def _fix_qr_positive(q,r) :
    rn, rm = r.shape
    #assert(rn == rm)
    assert(q.shape[1] == rn)

    rd = np.diag(r)
    ix = np.nonzero(rd <0)[0]
    q[:,ix]*=-1
    r[ix,:]*=-1
    return q,r

def _qr(lr) :
    q,r=np.linalg.qr(lr)
    return _fix_qr_positive(q,r)

def qr1(lr, step_frac = 1.0/32.0) :
    """
    lr shape n, m.  If n>=m, return normal qr,
    otherwise, do the following:
    repeat
    1. get the first n-by-n qr
    2. move forward to next n-by-n by delta
    3. fillin the bew delta coef
    4. estimate the first delta residual from
       delta to n-delta bars
    5. fill in the first delta coef by 
       this residual to the new delta bars

    This is a simple version of getting 
    incomplete qr.  In case n >> delta,
    the newly added delta coef is assumed
    to be less influenced by the bars 
    earlier than n. 
    """

    n, m = lr.shape
    dm = int(max(min(n * step_frac, 20),2))
    m0 = 0
    q = np.zeros((n, m))
    r = np.zeros((m, m))
    m1 = min(m, n-1)
    m1p=0
    while m1 != m1p :
        q0,r0=_qr(lr[:,m0:m1])
        q[:,m1p:m1] = q0[:,m1p-m1:]
        r[max(m1-(n-1),0):max(m1,n-1),m1p:m1]=r0[:,m1p-m1:]
        m1p=m1
        m1=min(m1+dm,m)
        m0=max(m1-(n-1),0)
    return q,r

"""
# testing purpose
randx=0
def _choice(x,k,replace=True) :
    global randx
    randx+=1
    return x[(np.arange(k)+randx)%len(x)]
"""

def gen_incomplete_qr(lr, n1, m1=0) :
    """
    This generates synthetic data based on qr of lr. 
    lr has shape n,m, n<m. 
    It first get q0,r0=np.qr(lr), then uses q0 as
    a distribution and generates n1-n samples, with
    replacement to add to q1.  additional samples
    lr1 = np.dot(q1, r0[:,m1:])

    lr: shape n,m
    n1: total rows. n1-n to be generated
    m1: the last m1 columns to be calculated
    returns lr1
    """
    n,m=lr.shape
    assert (n>m)
    assert (n1>n)
    assert (m1<n)
    q0,r0=_qr(lr[:,:n])
    qn=[]
    for q in q0.T :
        qn.append(np.random.choice(q, n1-n, replace=True))
        #qn.append(_choice(q, n1-n, replace=True))
    qn=np.array(qn).T
    return np.dot(qn,r0[:,m1:])

def qr2(lr) :
    """
    use generative method lr
    synthetically generate more data based on qr and run from there
    """
    n,m=lr.shape
    if m<n : 
        q0, r0 = np.linalg.qr(lr)
        return fix_qr_positive(q0,r0)

    lr1=np.empty((m+1,m))
    lr1[:n,:]=lr.copy()
    lr1[n:,:n-1]=gen_incomplete_qr(lr[:,:n-1],m+1)
    m0 = 1
    m1 = n
    while m1<=m:
        lr1[n:,m1-1:m1]=gen_incomplete_qr(lr[:,m0:m1],m+1,n-2)
        m0+=1
        m1+=1
    return _qr(lr1)

def picklr(lr, k, w=None) :
    n,m=lr.shape
    if w is None:
        w=np.ones(n).astype(float)
    elif w == 'tanh' :
        w=np.tanh(np.arange(n)/float(n)*3.0)
    w/=np.sum(w)
    n0=np.random.choice(np.arange(n),k,replace=False,p=w)
    return n0


def bootstrap_vol(lr,w=None,cnt=1000) :
    """
    calculate a bootstrap version of the volatility (mean of square lr)
    and the deviation of it. 
    """
    n,m=lr.shape
    if w is None:
        w=np.tanh(np.arange(n+1))[1:]
    w/=np.sum(w)

    k=int(n*0.8)
    cnt = np.min(cnt, k)
    sd=[]
    for i in np.arange(k):
        pass

def qr7(lr_, n1=None, n2=None, n3=None, if_plot=False, dt=None,prob=False) :
    lr=lr_.copy()
    n,m=lr.shape
    if m<n : 
        q0, r0 = np.linalg.qr(lr)
        return fix_qr_positive(q0,r0)
    assert(n>m/2+1)
    if n1 is None:
        n1=m*3
    if n2 is None:
        n2=n/5
    if n3 is None:
        n3=max(n2/10,1)
    assert n2 > 10
    assert n2 < m-n3-1
    assert n1 >= n

    lr1=np.zeros((n1,m))
    lr1[:n,:]=lr.copy()
    lr1[n:,:n2]=gen_incomplete_qr(lr[:,:n2],n1)
    m0=n2+1
    m1=min(n2+n3,m)
    lrc=np.cumsum(lr1,axis=1)
    if prob :
        w0=np.sqrt(np.std(lr,axis=0))
        wc=np.cumsum(w0)
    while m0<=m:
        if not prob :
            ix=np.linspace(0,m0-1,n2,dtype=int)
        else :
            ix=np.random.choice(np.arange(m0-1,dtype=int),m0-n2,replace=False,p=w0[:m0-1]/np.sum(w0[:m0-1]))
            ix=np.delete(np.arange(m0),ix)
        lr0=lrc[:,ix[1:]-1]-np.vstack((np.zeros(n1),lrc[:,ix[1:-1]-1].T)).T
        if prob :
            sc=np.arange(m0)+1
            scl=sc[ix[1:]-1]-np.r_[0,sc[ix[1:-1]-1]]
            lr0/=np.sqrt(scl)

        delta=m1-m0+1  # this step size
        lr0=np.vstack((lr0.T,lr1[:,m0-1:m1].T)).T
        q1,r1=_qr(lr0[:,:-delta])
        LR0=lr0[:n,-delta:]
        Q0=q1[:n,:]
        R0=np.dot(Q0.T,LR0)*n1/n
        E=LR0-np.dot(Q0,R0)
        qe,re=_qr(E)
        i0=np.arange(delta)
        q0_=qe[np.random.randint(0,n,(n1-n,delta))[:,i0],i0]
        lrn=np.dot(q1[n:,:],R0)+np.dot(q0_,re)
        lr1[n:,m0-1:m1]=lrn.copy()
        lrn[:,0]+=lrc[n:,m0-2]
        lrc[n:,m0-1:m1]=np.cumsum(lrn,axis=1)

        m0+=delta
        m1=min(m1+delta,m)

    q,r=_qr(lr1)
    if if_plot:
        _plot_qr_eval(lr,q,r,dt)
    return q,r

def qr8(lrp_, lr_, n1=None, n2=None, n3=None, if_plot=False, dt=None,prob=False) :
    lr=lr_.copy()
    n,m=lr.shape
    if m<n : 
        q0, r0 = np.linalg.qr(lr)
        return fix_qr_positive(q0,r0)
    assert(n>m/2+1)
    if n1 is None:
        n1=m*3
    if n2 is None:
        n2=n/5
    if n3 is None:
        n3=max(n2/10,1)
    assert n2 > 10
    assert n2 < m-n3-1
    assert n1 >= n

    lr1=np.zeros((n1,m))
    lr1[:n,:]=lr.copy()
    lr1[n:,:n2]=gen_incomplete_qr(lr[:,:n2],n1)
    m0=n2+1
    m1=min(n2+n3,m)
    lrc=np.cumsum(lr1,axis=1)
    if prob :
        w0=np.sqrt(np.std(lr,axis=0))
        wc=np.cumsum(w0)
    while m0<=m:
        if not prob :
            ix=np.linspace(0,m0-1,n2,dtype=int)
        else :
            ix=np.random.choice(np.arange(m0-1,dtype=int),m0-n2,replace=False,p=w0[:m0-1]/np.sum(w0[:m0-1]))
            ix=np.delete(np.arange(m0),ix)
        lr0=lrc[:,ix[1:]-1]-np.vstack((np.zeros(n1),lrc[:,ix[1:-1]-1].T)).T
        if prob :
            sc=np.arange(m0)+1
            scl=sc[ix[1:]-1]-np.r_[0,sc[ix[1:-1]-1]]
            lr0/=np.sqrt(scl)

        delta=m1-m0+1  # this step size
        lr0=np.vstack((lr0.T,lr1[:,m0-1:m1].T)).T
        q1,r1=_qr(lr0[:,:-delta])
        LR0=lr0[:n,-delta:]
        Q0=q1[:n,:]
        R0=np.dot(Q0.T,LR0)*n1/n
        E=LR0-np.dot(Q0,R0)
        qe,re=_qr(E)
        i0=np.arange(delta)
        q0_=qe[np.random.randint(0,n,(n1-n,delta))[:,i0],i0]
        lrn=np.dot(q1[n:,:],R0)+np.dot(q0_,re)
        lr1[n:,m0-1:m1]=lrn.copy()
        lrn[:,0]+=lrc[n:,m0-2]
        lrc[n:,m0-1:m1]=np.cumsum(lrn,axis=1)

        m0+=delta
        m1=min(m1+delta,m)

    q,r=_qr(lr1)
    if if_plot:
        _plot_qr_eval(lr,q,r,dt)
    return q,r

def _plot_qr_eval(lr, q, r, dt=None) :
    # plot the amount of noise and signal captured
    # lr = Q R
    # var(lr_i) = E(lr_i **2) - E(lr_i)**2
    #           = (Q_i r_i)^T (Q_i r_i) - mu_i**2
    #   Q_i = (q_1, q_2,...,q_i) r_i = R_i
    # Therefore
    #  sum(r_i^2) = n * (var(lr_i) + mu_i**2)
    # 
    sd=np.std(lr,axis=0)
    mu=np.mean(lr,axis=0)
    n,m=q.shape
    if dt is None:
        dt=np.arange(m)

    v=(sd**2+mu**2)*n
    pl.figure()
    for k in np.arange(10) :
        pl.plot(dt[k:], r[np.arange(m-k),np.arange(m-k)+k]**2/v[k:], label='diag'+str(k))
    pl.title('diagonals as ratio of total var')
    pl.grid()
    pl.legend(loc='best')

    pl.figure()
    pl.imshow(np.sqrt(r**2/v))

    pl.figure()
    for k in np.arange(100) :
        pl.plot(dt[k+1:], r[k, k+1:]**2/v[k+1:])
    pl.title('horizontals as ratio of total var')
    pl.grid()


def _eval_qr(lr) :
    rs=[]
    for n1 in [2000,3000,5000,8000,12000] :
        for n2 in [100, 200, 250, 350, 500, 600, 800]:
            #for prob in [False, True] :
            for n3 in [5,20] :
                print ("running ", n1, n2, n3)
                q,r=qr6(lr,n1=n1,n2=n2,n3=n3)
                rs.append((n1,n2,n3,q.copy(),r.copy()))
    return rs

def _get_consistency(rs):
    sig=[]
    r0=0
    for i, rs0 in enumerate(rs):
        n2,n2,n3,q,r=rs0
        r2=r**2
        s0=np.sqrt(np.sum(r2)-np.sum(np.diagonal(r2)))
        sig.append(s0)
        r0=r0+r
    r0/=i
    rd=[]
    rvar=0
    for i, rs0 in enumerate(rs):
        n1,n2,n3,q,r=rs0
        prob=(i%2==1)
        rd0=(r-r0)**2
        rvar+=rd0
        rd.append(np.sum(rd0)-np.sum(np.diagonal(rd0)))

    rvar/=i
    return np.array(sig),rd,r0,rvar

def plot_r3d(r0,diag=False) :
    from mpl_toolkits.mplot3d import Axes3D
    fig=pl.figure()
    ax=fig.add_subplot(111,projection='3d')
    n,m=r0.shape
    assert(n==m)
    r=r0
    if not diag :
        r=r0.copy()
        r[np.arange(n),np.arange(n)]=0
    X=np.tile(np.arange(n),(n,1))
    Y=X.T
    ax.plot_wireframe(X,Y,r)

def qr_pca(lr) :
    """
    So this is to use pca to reduce number of columns before doing 
    qr. 

    X = U D V^T = U D0 V^T + U D1 V^T = U0 D0 V0^T + U1 D1 V1^T
    XV0 = U0 D0 + U1 D1 V1^T V0 , (since D1 is so small, it is discarded)
    XV0 = Q0 R0 + U1 D1 V1^T V0
    X = Q0 R0 V0^T + U1 D1 V1^T

    where Q0 R0 = X' = U0 D0
    """
    pass

##################################
###### ICA related stuffs ########
##################################
def test_ica(ncomp=7,nfeat=20,npca=None,nsamp=2000) :
    # the more complicated, the better
    x1=np.sin(np.arange(nsamp)/nsamp/100.0)
    x2=np.tanh((np.arange(nsamp)-nsamp/2.0)/192.0+3.0)
    x3=((np.arange(nsamp)-nsamp*0.4)/nsamp)**3.0
    #x4=np.cos(np.arange(2000)/412.0/1.2+1.2)*np.sin(np.sqrt((np.arange(2000)/1022.0+1.0)))
    x4=np.cos(np.arange(nsamp)/nsamp/120.0/1.2+1.2)

    x5=(np.arange(nsamp)-nsamp/2.0)/(nsamp/6.0)*(1.0/(1+np.exp(-np.arange(nsamp)/nsamp/3.0)))
    x6=np.random.normal(0,1,nsamp)
    x7=np.random.beta(2,5,nsamp)

    X=np.vstack((x1,x2,x3,x4,x5,x6,x7)).T
    mix=np.random.normal(0,1,(7,nfeat))
    Xm=np.dot(X,mix)
    Xm-=np.mean(Xm,axis=0)
    Xm/=np.std(Xm,axis=0)
    if npca is not None:
        u,s,vh=np.linalg.svd(Xm,full_matrices=False)
        Xm=np.dot(np.dot(u[:,:npca],np.diag(s[:npca])),vh[:npca,:])

    from sklearn.decomposition import FastICA
    ica=FastICA(n_components=ncomp)
    X0=ica.fit_transform(Xm)

    for i in np.arange(7) :
        pl.figure() ; pl.plot(X[:,i]); pl.title('X'+ str(i+1))
    for i in np.arange(ncomp) :
        pl.figure() ; pl.plot(X0[:,i]); pl.title('Xm'+str(i+1)), pl.plot(-X0[:,i])

def get_ica_ncomp_beta_yt(ncomp, X, Xt, Y) :
    from sklearn.decomposition import FastICA
    ica=FastICA(n_components=ncomp)
    x=ica.fit_transform(X)
    xt=ica.transform(Xt)
    beta=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),Y)
    yt=np.dot(xt,beta)
    return ica.components_.copy(), ica.mean_.copy(), beta, yt

def ica_eval0(lrd, ncomp, s0, e0, e1=None, yt_bars=None) :
    n,m=lrd.shape
    if e1 is None:
        e1 = n  # all samples
    if yt_bars is None:
        yt_bars = m # all bars
    X=lrd[s0:e0,:].copy()
    Xt=lrd[e0:e1,:].copy()
    Y=np.sum(lrd[s0+1:e0+1,:yt_bars],axis=1)
    Yt=np.sum(lrd[e0+1:e1+1,:yt_bars],axis=1)
    comp,mu,beta,yt=get_ica_ncomp_beta_yt(ncomp,X,Xt,Y)
    return comp,mu,beta,yt,Yt

def ica_eval(lrd) :
    """
    lrd: daily lr, shape n,m
    ncomp: ica ncomp
    s0, e0, start/end of training
    e1 end of testing, default till end
    yt_bars: number of bars to aggregate for target, default 1 day, i.e. m

    evaluate:
    1. the consistency of ica components
    2. the consistency of beta
    3. predictability of components
    """
    # 1. try consistency of 1 day ica
    for ncomp in [10, 20, 40, 100]:
        for d in [1000, 2000, 4000]:
            for s0 in [] :
                pass

##############
# Main mergelr
##############

def eval(sd,r) :
    """
    sd is the standard deviation of each column of lr
    r is the estimation of qr using normalized lr, i.e. each column has been normalized with sd.
    """
    assert(np.min(np.diagonal(r)) > 0)

def vol_buck(lr,sd0):
    """
    With a reasonable smoothed sd, probably via bootstrap, with tanh weight,
    The procedure works as the following:
    1. obtain estimation of sd for current bar 
    2. find the worst kw bars
    3. pick one to merge that leads to best obj
    4. pick the bewt kb bars
    5. pick one to split that leads to best obj
    6. decide which action to take

    stop if no improvements can be obtained

    objective: possible object function could be:
       1. smoothness of vol
       2. maximum total predictable multiplies the std/vol
    """
    pass


