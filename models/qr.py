import numpy as np
import scipy
import datetime
import multiprocessing as mp
import time
from matplotlib import pyplot as pl

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

def qr6(lr_, n1=None, n2=None, n3=None, if_plot=False, dt=None,prob=False) :
    """
    use generative method lr
    synthetically generate more data based on qr and run from there
    This is merges the previous bars in generating synthetic samples
    for later bars.  It turns out that this is not necessary.  So
    qr3() is preferred. 
    
    There is another concern about the degree of freedome making
    qr unstable.  When n close to m, qr tends to overfit towards
    the latr bars.  So it is benefitial to generate more samples. 
    However, single the real data is only n, each step needs to 
    maintain a smaller n2 in order to avoid overfitting. 
    
    Due to lack of sample, the later bar's r_i are estimated
    with aggregated previous bars. So smaller n2 leads to more
    aggregation, which could reduce the sensitivity at later bars,
    a more smoothed effect in r_i compared with larger n2. 

    Input:
       n1=m*5, i.e. total samples, including synthetically generated.
       n2=n/5, i.e. number of columns used for each step
       n3 is the step size of each esitmation
       5 being a degree of freedom ratio. i.e. expect qr operate on
       matrix with 5 times more samples than features.
    """
    lr=lr_.copy()
    n,m=lr.shape
    if m<n : 
        q0, r0 = np.linalg.qr(lr)
        return fix_qr_positive(q0,r0)

    assert(n>m/2+1)
    # this is the over-sample size. Note
    # further increases this increases the 
    # diagonal of resulting r, esp. at later bars,
    # meaning that the 
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
    # this just generates more samples with existing qr
    lr1[n:,:n2]=gen_incomplete_qr(lr[:,:n2],n1)

    m0=n2+1
    m1=min(n2+n3,m)
    lrc=np.cumsum(lr1,axis=1)
    if prob :
        w0=np.sqrt(np.std(lr,axis=0))
        wc=np.cumsum(w0)
    while m0<=m:
        """
        merging previous bars doesn't make
        big difference as the generated bars will
        be qr'ed at the end
        """
        # merge lr[:,:m0] into [n, n2]
        if not prob :
            ix=np.linspace(0,m0-1,n2,dtype=int)
        else :
            #using volatility as probability to merge bars doesn't seem to help
            #with signal capturing. And it's slow. 
            ix=np.random.choice(np.arange(m0-1,dtype=int),m0-n2,replace=False,p=w0[:m0-1]/np.sum(w0[:m0-1]))
            ix=np.delete(np.arange(m0),ix)
        lr0=lrc[:,ix[1:]-1]-np.vstack((np.zeros(n1),lrc[:,ix[1:-1]-1].T)).T

        # no difference in scaling here
        if prob :
            sc=np.arange(m0)+1
            scl=sc[ix[1:]-1]-np.r_[0,sc[ix[1:-1]-1]]
            lr0/=np.sqrt(scl)

        delta=m1-m0+1  # this step size
        lr0=np.vstack((lr0.T,lr1[:,m0-1:m1].T)).T
        # r0[:,-1] is the in-sample relationship, to be maintained
        q0,r0=_qr(lr0[:n,:])

        assert r0.shape[0]==r0.shape[1] and r0.shape[1]==n2+delta-1
        # q1 is an extention of q0 with previous generated samples
        q1,r1=_qr(lr0[:,:-delta])
        #scale q1 to q0
        q1*=np.sqrt(n1)/np.sqrt(n)
        #pick q0 to generate the column of m1
        i0=np.arange(delta)
        ix=np.random.randint(0,n,(n1-n,delta))

        # testing
        #ix=_choice(np.arange(n),n1-n,replace=True)
        #ix=np.array(ix).reshape((n1-n,delta))

        q0_=q0[ix[:,i0],i0+n2-1]

        #generate
        lrn=np.dot(q1[n:,:],r0[:-delta,-delta:])+np.dot(q0_,r0[-delta:,-delta:])
        lr1[n:,m0-1:m1]=lrn.copy()

        #update cumsum
        lrn[:,0]+=lrc[n:,m0-2]
        lrc[n:,m0-1:m1]=np.cumsum(lrn,axis=1)

        m0+=delta
        m1=min(m1+delta,m)

    q,r=_qr(lr1)
    if if_plot:
        _plot_qr_eval(lr,q,r,dt)
    return q,r


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
                print "running ", n1, n2, n3
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

