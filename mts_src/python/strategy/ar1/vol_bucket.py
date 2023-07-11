import copy
import numpy as np
import Outliers as OT
import time
import dill

"""
Volatility Bucketing

Given a matrix lr, shape of n, m, choose
a sebset in the m columns so that the 
some score is optimized, such as 
aggregated volatility or function of r
from the qr decomposition.
"""

####################################
# Simple bucketing of columns of lr,
# using the volatility std(lr) or 
# trade volumes, with a possible
# regulation (i.e. 1S, 5S, 15S, 30S,
# 60S bucketing). 
# Since the columns are large compared
# with rows (days), it iterates by
# aggregates bars for most equal outcome.
# 
# usually from daily 1S bars 82800 to 
# 82800//60, using vb_vol_ix(), then 
# use q,r based functions to find optimal
# bucketing.
#
####################################
def vector_partition(vector, partition_cnt, min_bars = 1):
    # partition positive vector into ix so that 
    # sum of partitions approximtely equal
    # this is good for numbers that could be added up
    # i.e. trade volume.  The std of lr cannot be
    # added up.

    assert len(vector) >= partition_cnt
    vec0 = np.array(vector).copy()
    vc = np.cumsum(vec0)
    assert np.min(vec0) >= 0

    ix = np.arange(len(vec0)).astype(int)
    ix_exclude = []
    while True:
        while len(vec0) > partition_cnt:
            # find a minimum ix0, ix1 to merge
            v0 = vec0[:-1]+vec0[1:]
            ix0 = np.argsort(v0)[0]
            ix = np.delete(ix, ix0)
            vec0 = vc[ix] - np.r_[0, vc[ix[:-1]]]

        if min_bars > 1:
            ixd = ix[1:]-ix[:-1]
            done = True
            while np.min(ixd) < min_bars:
                ixd0 = np.argsort(ixd, kind='stable')[0]+1 # delete the next one
                # exclude ixd0 from initial vec0
                ix_exclude.append(ix[ixd0])
                ix = np.delete(ix, ixd0)
                ixd = ix[1:]-ix[:-1]
                done = False
            if done:
                break

            print('removing %d close bars: %s'%(len(ix_exclude), str(ix_exclude)))
            ix = np.arange(len(vector)).astype(int)
            ix = np.delete(ix, ix_exclude)
            vec0 = vc[ix] - np.r_[0, vc[ix[:-1]]]
        else:
            break
    return ix, vec0

def vb_vol(lr, m_out, min_bars=2):
    """ evenly distribute vol into buckets
      cl = dill.load(open('/tmp/md_pyres/md_dict_CL_19990101_20220826_30S.dill','rb'))
      For example, if lr is 30S bar in [ndays, nbars, [utc, lr, vol, vbs, lpx]]
      then ix, lrn = vb_vol(lr, 400, 4) gives exactly 276 bars
    """

    # remove outliers in lr
    lrs = OT.soft1(lr, np.std(lr, axis=0), 15, 1)
    lr0 = lrs[:,1:] # remove overnight
    n,m=lr0.shape
    ix = None
    lrc = np.cumsum(lrs[:,1:],axis=1)
    while m > m_out-1:
        m_out0 = max(m_out-1, m-1)
        sd = np.std(lr0,axis=0)

        # smooth sd
        sdm = np.mean(sd)
        sd = OT.soft1(sd-sdm, np.std(sd), 5, 1)+sdm

        sdcs = np.cumsum(sd)
        sd0 = np.r_[np.arange(0,sdcs[-1],sdcs[-1]/(m_out0-1)),sdcs[-1]+0.1]
        ix0 = np.clip(np.searchsorted(sdcs, sd0[1:]),0,m-1)

        # remove the same ix
        ix00 = np.nonzero(ix0[1:]-ix0[:-1]==0)[0]
        ix0 = np.delete(ix0, ix00)

        if ix is not None:
            ix = ix[ix0]
        else:
            ix = ix0
        lr0 = lrc[:,ix]-np.vstack((np.zeros(n),lrc[:,ix[:-1]].T)).T
        n,m=lr0.shape
        print(m)

    # include the overnight
    ix = np.r_[0, ix+1]
    ixz = np.nonzero(ix[1:]-ix[:-1]<min_bars)[0]
    while len(ixz) > 0:
        # enforce a minimum bar time
        ixz1 = np.clip(ixz+1, 0, len(ix)-1)
        ixzn = np.clip(ixz1+1, 0, len(ix)-1)
        ixzp = np.clip(ixz-1, 0, len(ix)-1)
        difp = ix[ixz]-ix[ixzp]
        difn = ix[ixzn]-ix[ixz1]
        ixz_del0 = np.nonzero(difp<=difn)[0]
        ixz_del1 = np.nonzero(difp>difn)[0]

        ixz_del = np.r_[ixz[ixz_del0], ixz1[ixz_del1]]
        ix = np.delete(ix, ixz_del[:1])
        ixz = np.nonzero(ix[1:]-ix[:-1]<min_bars)[0]

    lrc = np.cumsum(lrs, axis=1)
    lr_out = lrc[:,ix]-np.vstack((np.zeros(n),lrc[:,ix[:-1]].T)).T
    return ix, lr_out

def vb_avg(trade_volume, m_out, min_bars=1):
    """same as above, using the avg trade volume to bucket
    trade_volume: shape n,m (ndays, mbars)
    m_out: target mbars to return
    return:
       ix, the ending ix for each bucket, ix[-1] = m-1
    """

    # center and remove outliers
    tvm = np.mean(trade_volume,axis=0)
    tv = trade_volume - tvm #center
    tvs = OT.soft1(tv, np.std(tv, axis=0), 20, 1) + tvm

    # again to remove effect of mean
    tvm = np.mean(tvs,axis=0)
    tv = tvs - tvm
    tvs = OT.soft1(tv, np.std(tv, axis=0), 20, 1) + tvm

    # call optimal partition
    ix = vector_partition(tvs, m_out, min_bars = min_bars)
    return ix

def setup_minute_ixb(m, minute_ix=[1,5,15,30,60]):
    """
    m is number of 1 second bar
    minute_ix is the starting bars for each minute
              each ix is the number of seconds into the start of minute
              NOTE: not same as ixb!  '1' means 1 seconds into the minute
    return: ixb to be used for bucketing
    This is used to regulate the bars can be used for bucketing
    """
    ms=m//60
    assert 60*ms==m
    return (np.tile(np.arange(ms)*60,(len(minute_ix),1)).T+(np.array(minute_ix)-1)).flatten()

def vb_vol_ix(lr, m_out, ixb=None, minute_ix=[1,5,15,30,60]):
    """
    merge lr into m_out bars, one-by-one, merge the minimum std
    """
    n,m=lr.shape
    if ixb is None:
        ixb=setup_minute_ixb(m,minute_ix)
    assert ixb[-1]==m-1, 'last element of ixb has to be m-1'

    # get an lrb
    b=len(ixb)
    lrcs=np.cumsum(lr,axis=1)
    lrb=lrcs[:,ixb]
    lrb[:,1:]-=lrb[:,:-1]
    v=np.std(lrb,axis=0)**2
    while b>m_out:
        k=np.argsort(np.sum(np.vstack((v[:-1],v[1:])),axis=0))[0]

        # update v
        lrb1=lrcs[:,ixb[k+1]]
        lrb0=lrcs[:,ixb[k-1]] if k>0 else np.zeros(n)
        v=np.delete(v,k)
        v[k]=np.std(lrb1-lrb0)**2

        ixb=np.delete(ixb,k)
        b-=1
        if b%100==0:
            print(b)

    lrb=lrcs[:,ixb]
    lrb[:,1:]-=lrb[:,:-1]
    return ixb, lrb

#######################################
# q,r based bucketing, based on explained/
# unexplained coef in r.
# there are several versions
# 1. bucket_base1() takes lr as [ndays,nbars],
#    at each iteration, it finds a bar to merge
#    that resulting in lowest total unexplained
#    variance
# 2. bucket_ind() takes a list of ind of same
#    shape [ndays,nbars], with lr being the first,
#    to find bucketing for the indicator(s).
#    at each iteration, it finds a merge resulting
#    most explained lr's variance from ind's
#    contributions.
# 3. bucket_bootstrap() trys to do the same job
#    as bucket_ind(), but repeating large amound of
#    qr on target shape. It turns out that this
#    is hard to tune and converges slowly (badly)
#######################################

######################
# Utilities: 
# incomplete qr
# handles cases when rows less than cols
# this bootstraps the rows to find locallized
# dependencies to generate additional rows for
# qr
######################
def _fix_qr_positive(q,r):
    rn, rm = r.shape
    assert(q.shape[1] == rn)

    rd = np.diag(r)
    ix = np.nonzero(rd<0)[0]
    q[:,ix]*=-1
    r[ix,:]*=-1
    return q,r

def _qr(lr) :
    q,r=np.linalg.qr(lr)
    return _fix_qr_positive(q,r)

def gen_incomplete_qr(lr, n1, m1=0) :
    """
    This generates synthetic data based on qr of lr
    lr is incomplete, i.e. has the shape of n<m.
    It first get a standard qr decomposition from lr
    as q0, r0, then uses q0 as a distribution and 
    generates n1-n sample, with replacement to add
    to q1.  Additional samples lr1 = np.dot(q1, r0[:,m1:])

    lr: shape n,m
    n1: total rows, n1=-n to be generated
    m1: the last m1 columns to be calculated
    return lr1
    """

    n,m=lr.shape
    assert (n>m)
    assert (n1>n)
    assert (m1<n)
    q0,r0=_qr(lr[:,:n])
    qn=[]
    for q in q0.T :
        qn.append(np.random.choice(q, n1-n, replace=True))
    qn = np.array(qn).T
    return np.dot(qn, r0[:,m1:])

def mts_qr(lr_, n1=None, n2=None, n3=None, if_plot=False, dt=None, prob=True):
    lr=lr_.copy()
    n,m=lr.shape
    if m<n:
        return _qr(lr)
    assert(n>m/2+1) 
    if n1 is None :
        n1 = m*3
    if n2 is None:
        n2=n//5
    if n3 is None:
        n3=max(n2//10,1)

    assert n2>10
    assert n2 < m-n3-1
    assert n1 >= n

    lr1=np.zeros((n1,m))
    lr1[:n,:]=lr.copy()
    lr1[n:,:n2]=gen_incomplete_qr(lr[:,:n2],n1)
    m0=n2+1
    m1=min(n2+n3,m)
    lrc=np.cumsum(lr1,axis=1)
    if prob:
        w0=np.sqrt(np.std(lr,axis=0))
        wc=np.cumsum(w0)
    while m0<=m:
        if not prob:
            ix=np.linspace(0,m0-1,n2,dtype=int)
        else :
            ix=np.random.choice(np.arange(m0-1,dtype=int),m0-n2,replace=False,p=w0[:m0-1]/np.sum(w0[:m0-1]))
            ix=np.delete(np.arange(m0),ix)
        lr0=lrc[:,ix[1:]-1]-np.vstack((np.zeros(n1),lrc[:,ix[1:-1]-1].T)).T
        if prob :
            sc=np.arange(m0)+1
            scl=sc[ix[1:]-1]-np.r_[0,sc[ix[1:-1]-1]]
            lr0/=np.sqrt(scl)

        delta=m1-m0+1
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
        print('iteration: ', m0, m)

    q,r=_qr(lr1)
    if if_plot:
        _plot_qr_eval(lr,q,r,dt)
    return q,r

def _plot_qr_eval(lr, q, r, dt=None):
    """
    plot the amount of noise and signal captured
    lr = Q R
    var(lr_i) = E(lr_i **2) - E(lr_i)**2
              = (Q_i r_i)^T (Q_i r_i) - mu_i**2
    where
        Q_i = (q_1, q_2, ..., q_i) r_i = R_i
    Therefore
       sum(r_i^2) = n * (var(lr_i) + mu_i**2)
    """

    import matplotlib as pl
    sd = np.std(lr,axis=0)
    mu=np.mean(lr,axis=0)
    n,m=q.shape
    if dt is None :
        dt = np.arange(m)

    v=(sd**2+mu**2)*n
    pl.figure()
    for k in np.arange(10) :
        pl.plot(dt[k:], r[np.arange(m-k),np.arange(m-k)+k]**2/v[k:], label='diag'+str(k))
    pl.title('diag as ratio of total var')
    pl.grid()
    pl.legend(loc='best')

    pl.figure()
    for k in np.arange(100) :
        pl.plot(dt[k+1:], r[k, k+1:]**2/v[k+1:])
    pl.title('horizonals as ratio of total var')
    pl.grid()

##########################
# Utilities: 
# 1d/2d gaussian kernel 
# smoothing functions
##########################
def lrg1d(y, width=3, poly=3, dist_fn=None, check_neff=False):
    """
    dist_fn: distance func with wt = dist_fn(np.arange(n)),
             0 being itself
             1 being an immediate neighboring elements
             2 being the next neighboring elements, etc
    check_neff: if true, flatten the weight curve if neff is less than poly+1
                usually off
    """
    n = len(y)
    if dist_fn is None:
        wt = np.exp(-np.arange(n)**2/(2*width**2))/(np.sqrt(2*np.pi)*width)
    else:
        wt = dist_fn(np.arange(n))
    nzix = np.nonzero(wt>0)[0]
    nz = len(nzix)
    assert nz > poly+1, 'not enough weight for the poly'
    # nzix should include 0,1,...,nz-1
    assert 0 in nzix and nz-1 in nzix, 'gap in non-zero weights?'
    wt=wt[nzix]
    wt = np.r_[wt[::-1], wt[1:]]
    nwt = len(wt)
    X = [np.ones(nz)]
    for p in np.arange(poly) + 1 :
        X.append(np.arange(nz)**p)
    X = np.array(X).T
    Xt = np.vstack((X[::-1],X[1:,:]))
    Xt[:nz,np.arange(1,poly+1,2).astype(int)]*=-1
    xs = []
    for i in np.arange(n):
        # all non-zero weights in the neiborhood of i
        i0 = min(i,nz-1)
        i1 = min(n-i,nz)
        nzix0 = np.arange(-i0,i1).astype(int)
        wn0 = wt[nz-1+nzix0]
        nz0 = i0+i1
        if check_neff and nz0 > poly+2:
            # flat the wn curve in case neff is small
            while True:
                neff = np.sum(wn0)**2/np.sum(wn0**2)
                if neff>poly+1:
                    break
                wn0 = wn0**0.9

        X = Xt[nz-1+nzix0]
        wn0/=np.sum(wn0)
        XTwX=np.dot(np.dot(X.T,np.eye(nz0)*wn0),X)
        # try to do invert
        try:
            xi = np.linalg.inv(XTwX)
        except:
            w, v = np.linalg.eig(XTwX)
            # take upto 1e-3th eigan value
            w_max = np.max(w)
            w_min = np.min(w)
            assert w_min > 0, 'non-positive eigval in XTwX'
            w_th = w_max*0.001
            if w_th > w_min:
                wix = np.nonzero(w>w_th)[0]
                print('condition number %f too big, using only %d principle components'%(\
                        w_max/w_min, len(wix)))
                w=w[wix]
                v=v[:,wix]
            xi = np.dot(v,np.dot(np.eye(len(wix))*w**-1,v.T))
        wy0 = wn0*y[i+nzix0]
        xs.append(np.dot(X[i0,:],np.dot(xi,np.dot(X.T,wy0))))
    return np.array(xs)

def lrg2d(y2d, width=(3,3), surface_poly=2, dist_fn=None, min_wt=1e-10, mask=None):
    """ local regression using a gaussian kernel
        y2d:  shape n,m numpy array
        cov:  width[0]**2,     0
                0,         width[1]**2
        surface_poly: 0,1 or 2

        surface fitting with the gaussian kernel 
        Z = A*x**2 + B*x*y + C*x + D*y**2 + E*y + F

        mask: if give, shape[n,m] 0 or 1 to be multiplied to the final probablility,
              used to exclude certain areas in smooth, such as the diagnoal, or the
              lower left triangle in 'r' of qr.

        It is not optimized for speed/memory yet, as a todo item,
        only include those non-zero weights in the 
    """
    assert surface_poly<=2, 'cannot handle more than 2'
    n,m = y2d.shape
    # figure out n0, m0
    if dist_fn is None:  #use width
        nm0 = []
        for ix in [0,1]:
            wt0 = np.exp(-np.arange(n)**2/(2*width[ix]**2))/(np.sqrt(2*np.pi)*width[ix])
            nm0.append(len(np.nonzero(wt0>min_wt)[0]))
        n0,m0=nm0
    else:
        raise 'not implemented yet'

    # nm shape [n*m,2], coordiates (x,y) as (row,col), starting from (0,0), (0,1) to (n-1,m-1)
    nm = np.vstack((np.tile(np.arange(n0),(m0,1)).ravel(order='F'),np.tile(np.arange(m0),(1,n0)))).T
    wt = (np.exp(-np.sum((nm/np.array(width))**2,axis=1)/2)/(2*np.pi*width[0]*width[1])).reshape((n0,m0))
    # wtnm is [2*n-1,2*m-1] for +/-n, +/- m 2d gaussian distribution
    wtnm = np.vstack((np.hstack((wt[::-1,::-1],wt[::-1,1:])),np.hstack((wt[1:,::-1],wt[1:,1:]))))
    nm0 = np.vstack((np.tile(np.arange(2*n0),(2*m0,1)).ravel(order='F'),np.tile(np.arange(2*m0),(1,2*n0)))).T
    X = [np.ones(4*n0*m0)]
    if surface_poly>=1:
        X.append(nm0[:,0])
        X.append(nm0[:,1])
    if surface_poly == 2:
        X.append(nm0[:,0]*nm0[:,1])
        X.append(nm0[:,0]**2)
        X.append(nm0[:,1]**2)
    assert surface_poly<=2, 'surface_poly has to be less than 2, higher order to be implemented'
    pc=len(X)
    X = np.array(X).T.reshape((2*n0,2*m0,pc))
    if mask is None:
        mask = np.ones((n,m))
    assert mask.shape==(n,m) and set(np.unique(mask))<=set([0,1]), \
            'mask have shape same with y2d, can only have 0, or 1'

    xs = []
    for i in np.arange(n*m):
        # r, c the row and col of i
        r=i//m
        c=int(i%m)
        if mask[r,c] == 0:
            xs.append(y2d[r,c])
            continue
        # r0, r1, the above/down non-zero weights centered on wtnm, [n0-1-r0, n0-1+r1], 
        r0 = min(r,n0-1)
        r1 = min(n-r,n0)
        c0 = min(c,m0-1)
        c1 = min(m-c,m0)
        mask0=mask[r-r0:r+r1,c-c0:c+c1] #excluded mask value is 0
        wn=(wtnm[n0-1-r0:n0-1+r1,m0-1-c0:m0-1+c1]*mask0).ravel()
        X0=X[:r0+r1,:c0+c1,:].reshape(((r0+r1)*(c0+c1),pc))
        xtw=X0.T*wn
        wy=wn*(y2d[r-r0:r+r1,c-c0:c+c1].ravel())
        xs.append(np.dot(X[r0,c0,:],np.dot(np.dot(np.linalg.inv(np.dot(xtw,X0)),X0.T),wy)))
        if i%50000 == 0:
            print('%d:%d'%(i,n*m))
    return np.array(xs).reshape((n,m))


####################################
# some manual operations for testing
####################################
def merge_i(lr, i, m):
    # lr shape [n, >2*m]
    # merge with i and i+1
    n,m_=lr.shape
    i=min(m-2,i)
    k=m_//m
    lr1=np.empty((n,k*(m-1)))
    for kk in np.arange(k):
        lr1[:,kk*(m-1):kk*(m-1)+i+1] = lr[:,kk*m:kk*m+i+1]
        lr1[:,kk*(m-1)+i] += lr[:,kk*m+i+1]
        lr1[:,kk*(m-1)+i+1:(kk+1)*(m-1)] = lr[:,kk*m+i+2:(kk+1)*m]
    return lr1

def merge_ix_sum(lr, ixb):
    """
    merge lr, shape [n, m], by sum using ixb, where ixb[i] is the ending ix (including) of bar i
    """
    n,m=lr.shape
    ixb=np.array(ixb).astype(int)
    b=len(ixb)
    lr1=np.cumsum(lr,axis=1)
    lr1=lr1[:,ixb]
    lr1[:,1:]=lr1[:,1:]-lr1[:,:-1]
    return lr1

def merge_ix_sum_k(lr, ixb, m):
    """
    merge lr, shape [n, k*m], by sum using ixb, where ixb[i] is the ending ix (including) of bar i
    in case k>1, lr is a hstack version of k lr, merge all 
    """
    n,m_=lr.shape
    k=m_//m
    assert k*m==m_, 'lr shape not a multiple of m'
    ixb=np.array(ixb).astype(int)
    b=len(ixb)
    assert ixb[-1] == m-1, 'ixb last not m-1'
    lr1=np.cumsum(np.hstack((np.zeros((n,1)),lr[:,:m])),axis=1)
    return np.tile(lr1[:,ixb+1]-lr1[:,np.r_[0,ixb[:-1]+1]],(1,k))

def merge_ix_last_k(lr,ixb,m):
    """
    merge lr, shape [n, k*m], by last value using ixb, where ixb[i] is the ending ix (including) of bar i
    in case k>1, lr is a hstack version of k lr, merge all 
    """
    n,m_=lr.shape
    k=m_//m
    assert k*m==m_, 'lr shape not a multiple of m'
    ixb=np.array(ixb).astype(int)
    b=len(ixb)
    assert ixb[-1] == m-1, 'ixb last not m-1'
    return np.tile(lr[:,ixb],(1,k))

def merge_ix_avg_k(lr,ixb,m):
    """
    merge lr, shape [n, k*m], by sum using ixb, where ixb[i] is the ending ix (including) of bar i
    in case k>1, lr is a hstack version of k lr, merge all 
    """
    n,m_=lr.shape
    k=m_//m
    assert k*m==m_, 'lr shape not a multiple of m'
    ixb=np.array(ixb).astype(int)
    b=len(ixb)
    assert ixb[-1] == m-1, 'ixb last not m-1'
    ixd=ixb-np.r_[-1,ixb[:-1]]
    lr1=np.cumsum(np.hstack((np.zeros((n,1)),lr[:,:m])),axis=1)
    return np.tile((lr1[:,ixb+1]-lr1[:,np.r_[0,ixb[:-1]+1]])/ixd,(1,k))

def score1(lr1, rd_ix):
    # lr1: shape [n, k*m]
    # the explained variance, higher the better, no smooth, 
    n,b2=lr1.shape
    q,r=mts_qr(lr1,prob=False)
    v = np.std(lr1[:,rd_ix],axis=0)**2
    vd=np.diag(r)[rd_ix]**2
    vbf=1-vd/np.sum(r[:,rd_ix]**2,axis=0)
    return np.sum(vbf*v)  # total variance explained

def score_ind_unused(lr1, rd_ix0, rd_ix1):
    """ finding explained vol on target lr at r[:,rd_ix0],
    from contributions of 'j' indicators in r[rd_ix1,:]
    Target (lr) has to be the first in 'b' bar group, so
    no look ahead
    return:
        total explained variance in target (lr)
    """
    n,jbk=lr1.shape
    q,r=mts_qr(lr1,prob=False)
    r[np.arange(jbk),np.arange(jbk)]=0
    return np.sum(r[:,rd_ix0][rd_ix1,:]**2)/n

def score_ind_explained(lr1, rd_ix0, rd_ix1):
    """ finding explained vol on target lr at r[:,rd_ix0],
    from contributions of 'j' indicators in r[rd_ix1,:]
    Target (lr) has to be the first in 'b' bar group, so
    no look ahead
    return:
        total explained variance in target (lr)
    """
    n,jbk=lr1.shape

    lr0=lr1-np.mean(lr1,axis=0)
    lr0/=np.std(lr0,axis=0)

    q,r=mts_qr(lr0,prob=False)
    r[np.arange(jbk),np.arange(jbk)]=0 # just to entertain j=1
    r=r[:,rd_ix0][rd_ix1,:]
    rs=np.std(r)
    r-=np.clip(r,-3*rs,3*rs) # remove noisy r terms
    return np.sum(r**4)/n  # favor bigger r

def score_ind_unexplained(lr1, rd_ix0, rd_ix1):
    """ finding explained vol on target lr at r[:,rd_ix0],
    from contributions of 'j' indicators in r[rd_ix1,:]
    Target (lr) has to be the first in 'b' bar group, so
    no look ahead
    return:
        total explained variance in target (lr)
    """
    n,jbk=lr1.shape
    q,r=mts_qr(lr1,prob=False)
    rs2=  np.sum(r[:,rd_ix0]**2)  #total variance
    r[np.arange(jbk),np.arange(jbk)]=0 # just to entertain j=1
    rind2=np.sum(r[:,rd_ix0][rd_ix1,:]**2)
    return (rs2-rind2)/n

def scoreks(lr1, rd_ix, mask):
    # the explained variance, higher the better, using smoothed r
    # don't confuse with the base1() score function, which is unexplained
    # note this is very SLOW
    n,b2=lr1.shape
    q,r=mts_qr(lr1,prob=False)
    rs = lrg2d(r,width=(1,1),mask=mask)*mask #smoothed r no diag
    vb=np.clip(np.std(lr1[:,rd_ix],axis=0)**2- np.std((lr1[:,rd_ix]-np.dot(q,rs[:,rd_ix])),axis=0)**2,0,1e+10)
    return np.sum(vb)

def scorek0(lr0, rd_ix=None, k=2):
    # lr0: shape [n,m]
    # k:   hstack lr0 to k days and get last m of r
    # diag(r)[-m:], the unexpained variance.
    # lower is better
    n,m=lr0.shape
    lr1=lr0[:n-(k-1),:]
    for d in np.arange(k-1)+1:
        lr1=np.hstack((lr1,lr0[d:n-(k-1)+d,:]))

    if rd_ix is None:
        rd_ix=-(np.arange(m)+1)

    sd = np.std(lr1,axis=0)
    q,r=mts_qr(lr1/sd,prob=False)
    # r from a normalized lr. lower is better
    #return np.dot(np.diag(r)[-m:]**2,sd[-m:]**2)
    return np.dot(np.diag(r)[rd_ix]**2,sd[rd_ix]**2)/n

def scorek1(lr0, rd_ix=None, k=2):
    # explained variance, compared with scorek()
    # NOTE: 
    # scorek1(lr0, rd_ix, k) + scorek0(lr0, rd_ix, k) == np.std(lr0,axis=0)[rd_ix]**2
    # i.e. the explained and unexplained variance should equal to total variance
    n,m=lr0.shape
    lr1=lr0[:n-(k-1),:]
    for d in np.arange(k-1)+1:
        lr1=np.hstack((lr1,lr0[d:n-(k-1)+d,:]))
    if rd_ix is None:
        rd_ix=-(np.arange(m)+1)
    return score1(lr1,rd_ix)

#####################################################
# QR based bucketing:
# bucket_base1() - the baseline 
# first attempt, showed
# 1. the bucketing roughly makes sense and consistent
# 2. the bucketing stable over 1 year lookback
# 3. better understand qr scoring
# It's slow, each merge of 1 bar takes ~50 qr, so 
# all together takes about 50K qr, on often big matrix.
# Trying to speed up by using bootstrap has failed,
# see bucket_bootstrap().
########################################

def bucket_base1(lr, m_out, test_cnt=5):
    """target lr only, no indicator.  
    Refer to bcuket_ind() for getting bucket
    for indicator(s)
    """
    def merge(lr, i, m):
        # lr shape [n, >2*m]
        # merge with i and i+1
        n,m_=lr.shape
        i=min(m-2,i)
        k=m_//m
        lr1=np.empty((n,k*(m-1)))
        for kk in np.arange(k):
            lr1[:,kk*(m-1):kk*(m-1)+i+1] = lr[:,kk*m:kk*m+i+1]
            lr1[:,kk*(m-1)+i] += lr[:,kk*m+i+1]
            lr1[:,kk*(m-1)+i+1:(kk+1)*(m-1)] = lr[:,kk*m+i+2:(kk+1)*m]
        return lr1

    def score(lr1, m):
        # lr1 shape [n,>m], getting a qr
        # diag(r)[-m:], the unexpained variance.
        # lower is better
        n=lr1.shape[0]
        q,r=mts_qr(lr1,prob=False)
        rm=np.diag(r)[-m:]
        return np.dot(rm,rm)/n

    def pick(m, cnt, hist_list=[], hist_best_k=None):
        """ m: current m
            hist_list: a list of history picks, i.e. [-1, 3, 9, 17, m], 
                       meaning previous tried [3,9,17]. initially empty
            hist_best_k: the best one in previous history, i.e. '1', which is '9'
            cnt: number of ix to be tried
        """
        if len(hist_list) == 0:
            hl0=-1
            hl1=m-1
        else:
            hl0=hist_list[-1][hist_best_k]
            hl1=hist_list[-1][hist_best_k+2]

        m=hl1-hl0
        if np.sqrt(m) > cnt:
            kk = int(np.sqrt(m))
            ix = np.random.choice(np.arange(kk-1).astype(int), size=kk, replace=True)
            hl = ix+np.arange(kk).astype(int)*kk
        elif m > cnt:
            hl = np.random.choice(np.arange(m-1).astype(int), size=cnt, replace=False)
        else:
            hl = np.arange(m)
 
        hl.sort()
        hl+=(hl0+1)
        hist_list.append(np.r_[hl0,hl,hl1])
        return hl

    def find_merge(lr, m, test_cnt=5, max_iter=5):
        """ given lr, possibly with shift from original daily lr
            loop for finding a bar to merge with least unexplained variance from qr
        """
        hl=[]
        score_dict={}
        it=0
        best_k=None

        iter_run=[]
        while it<max_iter:
            hl0=pick(m,test_cnt,hist_list=hl,hist_best_k=best_k)
            for k, i in enumerate(hl0):
                if i in score_dict.keys():
                    continue

                # merge i
                lr1=merge(lr,i,m)
                # score
                s=score(lr1,m)
                score_dict[i]={'score':s, 'hl':copy.deepcopy(hl[-1]), 'k':k}
                print('iter %d: %d/%d'%(it,k,len(hl0)))

            # get the best k
            sarr=[]
            for i in score_dict.keys():
                sarr.append(score_dict[i]['score'])
            i=list(score_dict.keys())[np.argsort(sarr)[0]]
            sdi = score_dict[i]
            hl=[sdi['hl']]
            best_k=sdi['k']
            it+=1
            iter_run.append([copy.deepcopy(hl0),i])
        return i, iter_run, score_dict, sdi['score']
    
    n,m=lr.shape
    lr0=np.hstack((lr[:-2,:],lr[1:-1,:],lr[2:,:]))

    ixb = np.arange(m).astype(int)
    #iter_list=[]
    #sd_list=[]
    ixb_list=[]
    while m > m_out:
        m1=np.random.choice(np.arange(m))
        m2=m1+2*m
        lr0_m = lr0[:,m1:m2]
        i,iter_run,score_dict,scr=find_merge(lr0_m,m,test_cnt=test_cnt)
        i=int((i+m1)%m)
        if i==m-1:
            print('!!! cannot merge last bar with next first... merge with previous!')
            i=m-2

        lr0=merge(lr0,i,m)
        m-=1
        print('Merge bar %d with %d! (m:%d/%d)'%(i,i+1,m, m_out))
        ixb_list.append([ixb[i], scr])
        ixb=np.delete(ixb,i)
        #iter_list.append(iter_run)
        #sd_list.append(score_dict)
    return np.vstack((lr0[:,:m_out], lr0[-2:,-m_out:])), np.array(ixb).astype(int), ixb_list

######################################
# bucket_ind() 
# bucket for indciator's contribution
# onto lr's explained variance
# similar logic with bucket_base1()
#######################################
def bucket_ind(ind_list, merge_func_list, m_out, test_cnt=5, state_in=None, persist_fn=None, weekly=False):
    def pick(m, cnt, hist_list=[], hist_best_k=None):
        """ m: current m
            hist_list: a list of history picks, i.e. [-1, 3, 9, 17, m], 
                       meaning previous tried [3,9,17]. initially empty
            hist_best_k: the best one in previous history, i.e. '1', which is '9'
            cnt: number of ix to be tried
        """
        if len(hist_list) == 0:
            hl0=-1
            hl1=m-1
        else:
            hl0=hist_list[-1][hist_best_k]
            hl1=hist_list[-1][hist_best_k+2]

        m=hl1-hl0
        if np.sqrt(m) > cnt:
            kk = int(np.sqrt(m))
            ix = np.random.choice(np.arange(kk-1).astype(int), size=kk, replace=True)
            hl = ix+np.arange(kk).astype(int)*kk
        elif m > cnt:
            hl = np.random.choice(np.arange(m-1).astype(int), size=cnt, replace=False)
        else:
            hl = np.arange(m)
 
        hl.sort()
        hl+=(hl0+1)
        hist_list.append(np.r_[hl0,hl,hl1])
        return hl

    def find_merge_ind(ind_list, merge_func_list, ixb, weekly, test_cnt=5, max_iter=5):
        """ given X, shape[n,K*b*j], possibly with shift from original daily lr
            loop for finding a bar to merge with least unexplained variance from qr
        """
        j=len(merge_func_list)
        b=len(ixb)
        n=ind_list[0].shape[0]

        hl=[]
        it=0
        best_k=None
        score_dict={}
        b-=1 # target b
        rd_ix0 = -(np.arange(b)+1)[::-1]*j
        if weekly:
            K=1
            ix0=0
            # only cares about the last day's r
            rd_ix0=rd_ix0[-int(len(rd_ix0)//5):]
        else:
            K=2
            ix0=np.random.choice(b) # starting ix

        if j==1:
            rd_ix1=np.arange(K*b)
        else:
            rd_ix1 = np.delete(np.arange(K*b*j),np.arange(0,K*b*j,j))

        while it<max_iter:
            hl0=pick(b+1,test_cnt,hist_list=hl,hist_best_k=best_k)
            for k, i in enumerate(hl0):
                # i should be less than b
                if i in score_dict.keys():
                    continue
                i0 = int((i+ix0)%(b+1))
                ixb0=np.delete(ixb,i0)

                # this maybe slow, but necessary (?) to allow
                # multiple indicators having different merge_func_list
                X=merge_ix(ind_list,ixb0,merge_func_list=merge_func_list)

                # make K+1 hstack of X
                lr1=X[:n-K,:]
                for d in np.arange(K)+1:
                    lr1=np.hstack((lr1,X[d:n-K+d,:]))
                # take 2K from ix0
                lr1=lr1[:,ix0*j:(ix0+K*b)*j]

                # score: explained var, higher is better
                #s=-score_ind_unexplained(lr1, rd_ix0, rd_ix1)
                s=score_ind_explained(lr1, rd_ix0, rd_ix1)
                score_dict[i]={'score':s, 'hl':copy.deepcopy(hl[-1]), 'k':k}
                print('iter %d: %d/%d, i:%d, i0:%d, ixb:%d, score:%lf'%(it,k,len(hl0),i,i0,ixb[i0],s))

            # get the best k
            sarr=[]
            for i in score_dict.keys():
                sarr.append(score_dict[i]['score'])
            i=list(score_dict.keys())[np.argsort(sarr)[-1]]
            sdi = score_dict[i]
            hl=[sdi['hl']]
            best_k=sdi['k']
            it+=1
        score_dict['ixb']=ixb.copy()
        score_dict['ix0']=ix0
        return int((i+ix0)%(b+1)), sdi['score'], score_dict

    #####################
    # the main function 
    #####################
    # setup state
    j=len(ind_list)
    n,m=ind_list[0].shape
    if state_in is None:
        ixb = np.arange(m).astype(int)
        state={'ixb':ixb, 'ixb_list':[], 'weekly':weekly}
    else:
        state=copy.deepcopy(state_in)
    ixb=state['ixb']
    ixb_list=state['ixb_list']
    b=len(ixb)
    while b > m_out:
        i,scr,score_dict=find_merge_ind(ind_list, merge_func_list, ixb, weekly, test_cnt=test_cnt)
        print('Merge bar %d with %d! (m:%d/%d)'%(i,i+1, b-1, m_out))
        ixb_list.append([ixb[i], scr])
        ixb=np.delete(ixb,i)
        b=len(ixb)
        if persist_fn is not None:
            state['score_dict']=score_dict
            state['ixb']=ixb
            fp=open(persist_fn, 'wb')
            dill.dump(state, fp)
            fp.close()
    b=len(ixb)
    X=merge_ix(ind_list,ixb,merge_func_list=merge_func_list)
    return ixb, X, ixb_list


######################################################
# NOT USED, Converges TOO SLOW!!
# this is an indicator specific merge function, 
# to be provided by the main function

# The idea is to repeatedly perform qr score on target shape, searching
# for optimal bucketing points.  Points are weighted on previous score.
#
# use a weighted bootstap to find optimal
# bucket point, thus to allow to do qr onto target shape (much small)
# and would allow much direct search.  It also could entertain
######################################################
def merge_ix(ind_list, ixb, ixn=None, merge_func_list=None):
    """
    ind_list: length j list of shape [nday, mbars] array
        note in case j>1, the first indicator should be lr
    ixb: length k array of chosen ix, returned by pick
    ixn: index of 'n', days, to be included, default to all
    merge_func_list: if not None, length j list of function to merge,
        can be sum/last/avg, default to be all sum

    return:
       shape [nday, k*j] array for merged bars
    """
    j = len(ind_list)
    n,m=ind_list[0].shape
    if merge_func_list is None:
        merge_func_list=[merge_ix_sum]*j
    k=len(ixb)
    lr1=[]
    if ixn is None:
        ixn=np.arange(n)
    for ind, fn in zip(ind_list, merge_func_list):
        lr1.append(fn(ind[ixn,:], ixb).flatten())
    lr1=np.array(lr1).T.reshape((len(ixn),k*j))
    return lr1

def pick_w(wt, b, neff_min=0.5):
    """wt: length m array as p in choice, wt[-1] is ignored, as m-1 always picked
       b:  number of bars to be picked
       neff_min: minimum neff in terms of ixd/m
       return:
       x: length b array, x[i] as the ending ix of bar i, note that x[-1]=m-1
    """
    m=len(wt)
    assert m>=b
    v=np.arange(m-1).astype(int)
    while True:
        x=np.r_[np.random.choice(v,b-1,replace=False,p=wt[:-1]/np.sum(wt[:-1])),m-1]
        x.sort()
        xw=x-np.r_[-1,x[:-1]]
        neff=(m**2.0)/np.sum(xw**2.0)
        #print(neff)
        if neff>b*neff_min:
            return x

def update(ind_list, state, iterations, wt_count, wt_exp=1):
    """
    perform and qr and update the states
    state: {'rd_ix','mask',wt_sum','wt_cnt','score','ix_list','merge_func_list','neff_min','n_wt','n_cnt'}
    """
    #shift_ix: a start ix used for getting [n,(k-1)*m*j] for performing qr and score
    j=len(ind_list)
    n,m=ind_list[0].shape
    st=[]
    for k in ['rd_ix','mask','wt_sum','wt_cnt','score','ix_list','merge_func_list','neff_min','n_wt','n_cnt','K']:
        st.append(state[k])
    rd_ix,mask,wt_sum,wt_cnt,score,ix_list,merge_func_list,neff_min,n_wt,n_cnt,K=st
    b=len(rd_ix)

    #t0=time.time()
    #st0=0
    #mt=0
    score_updated=False
    for i in np.arange(iterations):
        # gen wt so that it's uniformed in [0,1] 
        # with wt_count non-zero elements
        wt=wt_sum/wt_cnt
        wt[np.isnan(wt)]=1  #higher priority for uninitiated ones
        wix=np.argsort(wt)
        wt/=wt[wix[-1]] #normalize to be of max=1
        if wt_count < m:
            wmin=wt[wix[-wt_count-1]]
        else:
            wmin=1e-6
        wt=np.clip(wt-wmin,1e-10,1e+10)**wt_exp
        ixb=pick_w(wt, b, neff_min=neff_min)
        ix0=np.random.choice(b) # starting ix
        ixn=np.random.choice(np.arange(n),n_cnt,replace='False',p=n_wt)

        #t01=time.time()
        X=merge_ix(ind_list,ixb,ixn,merge_func_list=merge_func_list)
        #mt+=(time.time()-t01)
        lr1=np.tile(X,(1,(K+1)))[:,ix0*j:(ix0+K*b)*j]
        #scr=score0(lr1, rd_ix, mask)

        #t00=time.time()
        scr=score1(lr1, rd_ix)
        #st0+= (time.time()-t00)

        # update state
        wt_sum[ixb]+=scr
        wt_cnt[ixb]+=1

        # record if its good
        if scr>score[-1]:
            ixs=np.searchsorted(-score,-scr)
            score=np.r_[score[:ixs],scr,score[ixs:-1]]
            ix_list=ix_list[:ixs]+[ixb]+ix_list[ixs:-1]
            score_updated=True

    #t1=time.time()
    #print('total: %f, score: %f(%f), merge: %f(%f)'%(t1-t0,st0,st0/(t1-t0),mt,mt/(t1-t0)))

    for k,v in zip(['wt_sum','wt_cnt','score','ix_list'], \
                   [wt_sum, wt_cnt, score, ix_list]):
        state[k]=v

    return score_updated

def bucket_bootstrap(ind_list, m_out, state_in=None, persist_fn=None):
    # setup state
    j=len(ind_list)
    n,m=ind_list[0].shape
    b=m_out
    if state_in is None:
        merge_func_list=[merge_ix_sum]*j  #all agg
        neff_min=0.6                    # set to close 1 for equal bucketing initially
        n_cnt=int(min(max(m_out*10,n*0.5),n*0.8))
        n_wt =1+(np.arange(n)+1)/n
        n_wt/=np.sum(n_wt)
        K = 2
        rd_ix = -(np.arange(b)+1)[::-1]*j
        mask = np.triu(np.outer(np.ones(K*b),np.ones(K*b)),k=1)
        wt_sum=np.zeros(m)
        wt_cnt=np.zeros(m).astype(int)
        nhist=100
        score=np.zeros(nhist)
        ix_list=[[]]*nhist
        state={}
        for k,v in zip(['rd_ix','mask','wt_sum','wt_cnt','score','ix_list','merge_func_list','neff_min','n_wt','n_cnt','K','nhist'],\
                       [rd_ix,mask,wt_sum,wt_cnt,score,ix_list,merge_func_list,neff_min,n_wt,n_cnt,K,nhist]):
            state[k]=v

        # initiate all the wt/scores
        state['neff_min']=0.6
        run_cnt = 100
        while run_cnt > 0:
            update(ind_list, state, 100, m)
            run_cnt-=1
            print('init remaining run: %d/%d'%(run_cnt, 100))

        if persist_fn is not None:
            fp=open(persist_fn,'wb')
            dill.dump(state,fp)
            fp.close()

    else:
        state=copy.deepcopy(state_in)
 
    # run through
    state['neff_min']=0.4
    nhist=state['nhist']
    while True:
        # start a search
        run_cnt = 200
        iter_cnt= 100
        frac=1-1.0/run_cnt

        # reset the wt_cnt to reflect the
        wt=state['wt_sum']/state['wt_cnt']
        state['wt_cnt']=(state['wt_cnt']/(np.mean(state['wt_cnt'])/(run_cnt*iter_cnt))).astype(int)
        state['wt_sum']=state['wt_cnt']*wt

        # tract the wt_count
        wt_count = m
        score_updated=False
        run_cnt =0
        while wt_count >= b:
            score_updated|=update(ind_list, state, iter_cnt, int(np.ceil(wt_count)))
            run_cnt+=1
            wt_count*=frac
            print('run_left: %d, wt_count: %d, best score: %f'%(run_cnt, wt_count, state['score'][0]))
        if persist_fn is not None:
            fp=open(persist_fn,'wb')
            dill.dump(state,fp)
            fp.close()
        if not score_updated:
            print('no better points found, exit')
            break

    return state

