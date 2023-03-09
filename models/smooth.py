import numpy as np

def lrg1d(y, width=3, poly=3, dist_fn=None, check_neff=False):
    """
    y: len n vector to be smoothed
    width: the std of gaussian pdf as weight of polynomial fitting
    poly:  degree, 0 flat fitting, 1 linear, etc
    dist_fn: distance func with wt=dist_fn(np.arange(n).astype(int)),
             input to the dist_fn is integer, 0 being weight of self,
             1 being weight of immediate neighbor, etc. symetric.
    check_neff: if true, flatten the weight curve if neff is less than
                poly+1.  Usually set it off
    """
    n=len(y)
    if dist_fn is None:
        wt = np.exp(-np.arange(n)**2/(2*width**2))/(np.sqrt(2*np.pi)*width)
    else:
        wt = dist_fn(np.arange(n))
    nzix = np.nonzero(wt>0)[0]
    nz = len(nzix)
    assert nz>poly+1, 'not enough weight for the poly'
    # nzix should include 0,1,...,nz-1
    assert 0 in nzix and nz-1 in nzix, 'gap in non-zero weights?'
    wt=wt[nzix]
    wt=np.r_[wt[::-1],wt[1:]]
    nwt=len(wt)
    X=[np.ones(nz)]
    for p in np.arange(poly)+1:
        X.append(np.arange(nz)**p)
    X=np.array(X).T
    Xt=np.vstack((X[::-1],X[1:,:]))
    Xt[:nz,np.arange(1,poly+1,2).astype(int)]*=-1
    xs=[]
    for i in np.arange(n):
        # all non-zero weights in the neighborhood of i
        i0=min(i,nz-1)
        i1=min(n-i,nz)
        nzix0=np.arange(-i0,i1).astype(int)
        wn0=wt[nz-1+nzix0]
        nz0=i0+i1
        if check_neff and nz0>poly+2:
            # flatt the wn curve in case neff is small
            while True:
                neff = np.sum(wn0)**2/np.sum(wn0**2)
                if neff>poly+1:
                    break
                wn0=wn0**0.9

        X=Xt[nz-1+nzix0]
        wn0/=np.sum(wn0)
        XTwX=np.dot(np.dot(X.T,np.eye(nz0)*wn0),X)
        # try to do invert
        try:
            xi=np.linalg.inv(XTwX)
        except:
            print('problem inverting, check eig')
            w,v=np.linalg.eig(XTwX)
            # take upto 1e-3th eigan value
            w_max=np.max(w)
            w_min=np.min(w)
            assert w_min>0, 'non-positive eigval in XTwX'
            w_th = w_max*0.001
            if w_th > w_min:
                wix=np.nonzero(w>w_th)[0]
                print('condition number %f. take principle components from %f to %f, (%d out of %d)'\
                        %(w_max/w_min, w_max, w_th, len(wix),len(w)))
                w=w[wix]
                v=v[:,wix]
            xi=np.dot(v,np.dot(np.eye(len(wix))*w**-1,v.T))
        wy0=wn0*y[i+nzix0]
        xs.append(np.dot(X[i0,:],np.dot(xi,np.dot(X.T,wy0))))
    return np.array(xs)

def lrg2d(y2d, width=(3,3), poly=2, dist_fn=None, min_wt=1e-10):
    """
    y2d: shape n,m numpy array
    cov: width[0]**2, 0
         0,           width[1]
    poly: 0(flat): Z=F
          1(2-linear): Z=F+E*x+D*y, or
          2(2-poly): Z=A*x**2+B*x*y+C*x+D**y**2+E*y+F
    dist_fn: not yet implemented
    min_wt: weight less than this is ignored
    """
    assert poly<=2, 'cannot handle more than 2'
    n,m=y2d.shape
    # figure out n0, m0
    if dist_fn is None:  #use width
        nm0=[]
        for ix in [0,1]:
            wt0=np.exp(-np.arange(n)**2/(2*width[ix]**2))/(np.sqrt(2*np.pi)*width[ix])
            nm0.append(len(np.nonzero(wt0>min_wt)[0]))
        n0,m0=nm0
    else:
        raise 'not implemented yet' #but why bother to put it there?

    # nm shape[n*m,2], coordinates (x,y) as (row,col), starting from (0,0), (0,1) to (n-1,m-1)
    nm=np.vstack((np.tile(np.arange(n0),(m0,1)).ravel(order='F'),np.tile(np.arange(m0),(1,n0)))).T
    wt=(np.exp(-np.sum((nm/np.array(width))*2,axis=1)/2)/(2*np.pi*width[0]*width[1])).reshape((n0,m0))
    wtnm=np.vstack((np.hstack((wt[::-1,::-1],wt[::-1,1:])),np.hstack((wt[1:,::-1],wt[1:,1:]))))
    nm0=np.vstack((np.tile(np.arange(2*n0),(2*m0,1)).ravel(order='F'),np.tile(np.arange(2*m0),(1,2*n0)))).T
    X=[np.ones(4*n0*m0)]
    if poly>=1:
        X.append(nm0[:,0])
        X.append(nm0[:,1])
    if poly==2:
        X.append(nm0[:,0]*nm0[:,1])
        X.append(nm0[:,0]**2)
        X.append(nm0[:,1]**2)
    pc=len(X)
    X=np.array(X).T.reshape((2*n0,2*m0,pc))

    xs=[]
    for i in np.arange(n*m):
        r=i//m
        c=int(i%m)
        r0=min(r,n0-1)
        r1=min(n-r,n0)
        c0=min(c,m0-1)
        c1=min(m-c,m0)
        wn=wtnm[n0-1-r0:n0-1+r1,m0-1-c0:m0-1+c1].ravel()
        X0=X[:r0+r1,:c0+c1,:].reshape(((r0+r1)*(c0+c1),pc))
        xtw=X0.T*wn
        wy=wn*(y2d[r-r0:r+r1,c-c0:c+c1].ravel())
        xs.append(np.dot(X[r0,c0,:],np.dot(np.dot(np.linalg.inv(np.dot(xtw,X0)),X0.T),wy)))
        if i%50000==0:
            print('%d:%d'%(i,n*m))
    return np.array(xs).reshape((n,m))
