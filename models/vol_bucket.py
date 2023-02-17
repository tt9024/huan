import numpy as np
import Outliers as OT

"""
Volatility Bucketing

Given a matrix lr, shape of n, m, choose
a sebset in the m columns so that the det(R)
is minimized, where R is the qr decomposition of 
the given lr.

This is is used to pick the trigger time of 
strategy in MTS, especially when the look back
is long relative to history data available.  
"""

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
    """
    this is taken from qr.py
    """
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

def vb(lr, m_out, prob=False, merge_frac = 0.1):
    """
    cumsum lr into n_out columns for approximately equal vol
    pick a column to merge so that diag(r)**2 reduces minimum
    """
    n,m=lr.shape
    lr0 = lr.copy()

    idx = np.arange(m).astype(int)
    while m  > m_out:
        q,r = mts_qr(lr0, prob=prob)
        rd = np.diag(r)
        # pick a smallest r_i to merge
        ix = np.argsort(rd)

        # merge 
        merge_cnt = min(max(int(np.round(m*merge_frac)),1), m-m_out)

        ixk = ix[:merge_cnt]
        ixx = np.argsort(ixk)
        for d, k in enumerate(ixx):
            i = ixk[k]-d
            if i == 0:
                j = 1
            elif i == m-1:
                j = m-2
            else:
                if rd[ixk[k]-1] > rd[ixk[k]+1]:
                    j = i+1
                else:
                    j = i-1
            # merge i and j
            i = min(i,j)
            idx = np.delete(idx,i+1)
            lr1 = np.zeros((n,m-1))
            lr1[:,:i] = lr0[:,:i]
            lr1[:,i] = lr0[:,i]+lr0[:,i+1]
            lr1[:,i:] = lr0[:,i+1:]
            lr0 = lr1
            n,m=lr0.shape
        print('iterate: ', m)
    return idx, lr0

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

# haven't tested yet
def clean_md(lr,vol,vbs,lpx,utc,aspd=None, bdif=None, adif=None):
    """lr = smooth_md(lr,vol,vbs,lpx,utc[-1,:])
    removes the inf/nan and 
    checks and warns the potential outliers, 
    lr - OT 30std
    vol - warning on 30std volume - mean
    lpx - warning on 30std simple return

    input:
        lr,vol,vbs,lpx,utc: shape [m,n] mdays, nbar
    return:
        lr: shape[m,n] log returns cleaned up
            make sure no warning for others
    
    """
    import Outliers as OT

    m,n=lr.shape
    lr0 = lr.copy()
    vol0 = vol.copy()
    vbs0 = vbs.copy()
    lpx0 = lpx.copy()

    aspd0 = aspd.copy() if aspd is not None else np.zeros((m,n))
    bdif0 = bdif.copy() if bdif is not None else np.zeros((m,n))
    adif0 = adif.copy() if adif is not None else np.zeros((m,n))
    val = [lr0, vbs0, vol0, lpx0, aspd0, bdif0, adif0]

    # remove nan/inf with 0
    for i, v in enumerate (val):
        ix = np.nonzero(np.isinf(v))
        if len(ix[0])>0:
            print('replacing %d inf for (%d)'%( len(ix[0]), i))
            v = 0
        ix = np.nonzero(np.isnan(v))
        if len(ix[0])>0:
            print('replacing %d nan for (%d)'%( len(ix[0]), i))
            v = 0
        val[i] = v

    # remove 30*sd as "wrong" value
    cnt = 2
    while cnt > 0:
        for i, v in enumerate(val[:1]):
            vm = np.mean(v,axis=0)
            vs = np.clip(np.std(v,axis=0),1e-10,1e+10)
            val[i] = OT.soft1(v-vm, vs, 30, 1)+vm
        cnt -= 1

    # need to regulate the overnight
    lr0 = val[0]
    lr00 = lr0[:,0]
    cnt = 2
    while cnt > 0:
        lr00m = np.mean(lr00)
        lr00 = OT.soft1(lr00-lr00m, np.std(lr00), 15, 1)+lr00m
        cnt -= 1
    val[0][:,0] = lr00

    # detect 100*sd vol
    vol0 = val[2]
    vm = np.mean(vol0,axis=0)
    vs = np.clip(np.std(vol0,axis=0),1,1e+10)
    ix = np.nonzero(np.abs(vol0-vm)-100*vs>0)
    if len(ix[0])>0:
        print('(%d) vol outliers (please check vbs also) at '%(len(ix[0])) + str(ix) + '\n' + str(vol0[ix]))
    else:
        print('vol/vbs good')

    # vbs check is included in the vol, so skipped here
    lpx0 = val[3]
    lpx0_ = lpx0.flatten()
    ixz = np.nonzero(np.abs(lpx0_) < 1e-10)[0]
    assert len(ixz) == 0, "zero price detected!"

    # simple return
    rtx0_ = np.r_[0, (lpx0_[1:]-lpx0_[:-1])/lpx0_[:-1]].reshape((m,n))
    vm = np.mean(rtx0_,axis=0)
    vs = np.std(rtx0_,axis=0)
    ix = np.nonzero(np.abs(rtx0_-vm)-100*vs>0)
    if len(ix[0])>0:
        print('(%d) lpx outliers at '%(len(ix[0])) + str(ix) + '\n' + str(lpx0[ix]))
        return ix
    else:
        print('lpx good')

    # check on the avg_spread and bid/ask diff
    for vx,name in zip(val[4:7], ['avg_spd','bid_diff','ask_diff']):
        vm = np.mean(vx,axis=0)
        vs = np.clip(np.std(vx,axis=0),1e-10,1e+10)
        ix = np.nonzero(np.abs(vx-vm)-30*vs>0)
        if len(ix[0])>0:
            print('(%d) %s outliers at '%(len(ix[0]),name) + str(ix) + '\n' + str(vol0[ix]))
        else:
            print('spd/bid_ask_diff good')

    return val[0]

def md_agg_ix(md_dict, ix) :
    """
    md_dict[symbol][mdays,nbars,[utc,lr,vol,vbs,lpx,aspd,bdif,adif]
    ix: the index at which the each aggegated bar should end at
        as returned by i.e. vol_lr(), note ix[i] = b means
        the partition ends at b (including b)
        0 <= b <= nbars-1, ix[-1] = nbars-1 
    """
    ix = np.array(ix).astype(int)
    assert np.min(ix[1:]-ix[:-1])>=1
    md_dict_out = {}
    for sym in md_dict.keys():
        md = md_dict[sym]
        m,n,cols=md.shape
        assert cols == 8 or cols == 5
        utc = md[:,:,0]

        # aggregate: lr, vol, vbs
        arr1 = []
        for k in [1,2,3,6,7]:
            if k >=cols:
                continue

            v = md[:,:,k]
            vc = np.vstack((np.zeros(m),np.cumsum(v,axis=1)[:,ix].T)).T
            arr1.append( vc[:,1:]-vc[:,:-1] )

        # just pick the latest
        arr2 = []
        for k in [0, 4]:
            arr2.append(md[:,ix,k])

        # just get an average
        arr3 = []
        ixd = np.r_[ix[0]+1, ix[1:]-ix[:-1]]
        assert (np.min(ixd)>0)
        for k in [5]:
            if k >= cols:
                continue
            v = md[:,:,k]
            vc = np.vstack((np.zeros(m),np.cumsum(v,axis=1)[:,ix].T)).T
            arr3.append( (vc[:,1:]-vc[:,:-1])/ixd )

        # output the same format [utc,lr,vol,vbs,lpx,aspd,bdif,adif]
        if cols == 8:
            md_dict_out[sym] = np.vstack((arr2[0].flatten(),arr1[0].flatten(),arr1[1].flatten(), arr1[2].flatten(), arr2[1].flatten(), arr3[0].flatten(), arr1[3].flatten(), arr1[4].flatten())).T.reshape((m,len(ix),8))
        else:
            md_dict_out[sym] = np.vstack((arr2[0].flatten(),arr1[0].flatten(),arr1[1].flatten(), arr1[2].flatten(), arr2[1].flatten())).T.reshape((m,len(ix),5))

    return md_dict_out

##############################################################
# pyres md_dict procedure
# 
# get raw md_dict from ar1_md, say 30 second bar
# 1. clean the data
# 2. adjust the columns to be [utc, lr, vol, vbs, lpx]
# 3. save to the md_pyres as a cleaned format of a single md
# 
# with the cleaned up 30 second bar, do a bucket using a chosen
# indicator, could be the primary lr, or the vbs of other symbol.
# 1. run:    ix, lr = def vb_vol(lr, m_out, min_bars=2)
# 2. repeat 1, so that number of bars is total_sec/barsec 
#    i.e. 276 for barsec = 300, 184 for barsec=450, 
#    138 for barsec=600, etc
# 3. aggregate all symbols in md_dict using the given bucket ix
#
# Finally call xobj_vd with md_dict as md_dict_in
###############################################################

def get_raw_md_dict(mts_symbol, start_day, end_day, barsec=30, extended_cols=True):
    """
    copied from ar1_md.md_dict_from_mts() with slight change:
    1. return cols including spd+bdif+adif
    2. default parameters
    return:
        bar: shape(ndays,n,8)) for [utc, lr, vol, vbs, lpx, aspd, bdif, adif]
        Note both the lr and lpx are roll adjusted
    """
    import ar1_md
    import mts_repo

    #sym = ar1_md.mts_repo_mapping[symbol]
    sym = mts_symbol
    if extended_cols:
        cols=['open', 'close', 'vol','vbs','lpx','utc','aspd','bdif','adif']
    else:
        cols=['open', 'close', 'vol','vbs','lpx','utc']
    open_col = np.nonzero(np.array(cols) == 'open')[0][0]
    close_col = np.nonzero(np.array(cols) == 'close')[0][0]
    vol_col = np.nonzero(np.array(cols) == 'vol')[0][0]
    vbs_col = np.nonzero(np.array(cols) == 'vbs')[0][0]
    lpx_col = np.nonzero(np.array(cols) == 'lpx')[0][0]
    utc_col = np.nonzero(np.array(cols) == 'utc')[0][0]
    if extended_cols:
        aspd_col = np.nonzero(np.array(cols) == 'aspd')[0][0]
        bdif_col = np.nonzero(np.array(cols) == 'bdif')[0][0]
        adif_col = np.nonzero(np.array(cols) == 'adif')[0][0]

    bars, roll_adj_dict = ar1_md.md_dict_from_mts_col(sym, start_day, end_day, barsec, cols=cols, use_live_repo=False, get_roll_adj=True)
    bars=mts_repo.MTS_REPO.roll_adj(bars, utc_col, [lpx_col], roll_adj_dict)

    # construct md_days
    if extended_cols:
        cols_ret = ['utc','lr','vol','vbs','lpx','aspd','bdif','adif']
    else:
        cols_ret = ['utc','lr','vol','vbs','lpx']

    ndays, n, cnt = bars.shape
    bars=bars.reshape((ndays*n,cnt))

    # use the close/open as lr
    # for over-night lr, repo adjusts first bar's open as previous day's close
    lr = np.log(bars[:,close_col]/bars[:,open_col])

    if extended_cols:
        bars = np.vstack((bars[:,utc_col], lr, \
            bars[:,vol_col], bars[:,vbs_col], bars[:,lpx_col],
            bars[:,aspd_col], bars[:,bdif_col], bars[:,adif_col])).T.reshape((ndays,n,8))
    else:
        bars = np.vstack((bars[:,utc_col], lr, \
            bars[:,vol_col], bars[:,vbs_col], bars[:,lpx_col])).T.reshape((ndays,n,8))

    return bars, cols_ret

############
# this may take some time
# check the data carefully to remove missing/wrong/abnormal data
#############
def clean_bars(bars):
    m,n,col_cnt = bars.shape
    utc = bars[:,:,0]
    lr = bars[:,:,1]
    vol = bars[:,:,2]
    vbs = bars[:,:,3]
    lpx = bars[:,:,4]
    if col_cnt > 5:
        aspd = bars[:,:,5]
        bdif = bars[:,:,6]
        adif = bars[:,:,7]
    else:
        aspd = None
        bdif = None
        adif = None

    # check for the warning message
    lr0 = clean_md(lr,vol,vbs,lpx,utc,aspd=aspd, bdif=bdif, adif=adif)

    return lr0

###################
# play with the vb_vol using the lr 
# lr0 = clean_bars(cl)
# ix, lrn = vb.vb_vol(lr, 400, 3) # i.e. could get 276 bars
# 
# md_dict = def md_agg_ix(md_dict, ix)
#
# build it into xobj
####################

def local_regression_gaussian(y, width=3, poly=3):
    """ local regression using a gaussian kernel
    """
    n = len(y)
    wt = np.exp(-np.arange(n)**2/(2*width**2))/(np.sqrt(2*np.pi)*width)
    wt = np.r_[wt[::-1], wt[1:]]
    nwt = len(wt)
    X = [np.ones(n)]
    for p in np.arange(poly) + 1 :
        X.append(np.arange(n)**p)
    X = np.array(X).T
    xs = []
    for i in np.arange(n):
        wn = wt[n-1-i:2*n-1-i]
        xtw =  X.T * wn
        wy = wn*y
        xs.append(np.dot(X[i,:], np.dot(np.dot(np.linalg.inv(np.dot(xtw,X)),X.T),wy)))
    return np.array(xs)

def local_regression_gaussian_2d(y2d, width=(3,3), poly=3):
    """
        y2d: shape n,m numpy array
        width: variances for n and m. [0] for row, [1] for col
              
        
        gassian kernel is mean 0, cov as 
           [ width[0]**2,  0 
             0,           width[1]**2 ]
    """
    n,m=y2d.shape
    nm=np.vstack((np.tile(np.arange(n),(m,1)).ravel(order='F'),np.tile(np.arange(m),(1,n)))).T
    wt=(np.exp(-((nm[:,0]/width[0])**2+(nm[:,1]/width[1])**2)/2)/(np.sqrt((2*np.pi)**2)*width[0]*width[1])).reshape((n,m))
    wtnm=np.vstack((np.hstack((wt[::-1,::-1],wt[::-1,1:])),np.hstack((wt[1:,::-1],wt[1:,1:]))))
    X = [np.ones(n*m)]
    for p in np.arange(poly)+1:
        X.append(np.arange(n*m)**p)
    X=np.array(X).T

    xs=[]
    y2dr = y2d.ravel()
    for i in np.arange(n*m):
        n0=i//m
        m0=int(i%m)
        wn=wtnm[n-1-n0:2*n-1-n0,m-1-m0:2*m-1-m0].ravel()
        xtw=X.T*wn
        wy=wn*y2dr
        xs.append(np.dot(X[i,:],np.dot(np.dot(np.linalg.inv(np.dot(xtw,X)),X.T),wy)))
    return np.array(xs).reshape((n,m))

