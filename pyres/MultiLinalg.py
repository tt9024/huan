"Contains function sfor efficiently performing Linear algebra on symetric block diagonal matrices \n\
 and lower triangular block diagonal matrices \n\
 The 'n' diagonal 'ni' x 'ni' blocks are represented by an 'n' x 'nv' numpy array, where \n\
 'nv = (ni*(ni+1))/2'.  A single row corresponds to the lower triangle of a single \n\
 diagonal block represented as the vector formed from row by row \n\
 vectors of the lower triangle \n\
"
import numpy as np
import scipy.linalg as lg

def indexvectors(ni) :
    """
    ix0, ix1=indexvectors(ni)
    generates index vectors for passing between the lower triangle row by row vector
    representation. The lower triangular entries of an 'ni x 'ni' matrix
    and all the entries of a symmetric 'ni x ni' matrix
    'M' = symmetric 'ni x ni' matrix
    'L' = lower triangular 'ni x ni' matrix
    'v' = length 'nv = (ni*(ni+1))/2' vector, the row by row vectors of the lower triangle
    The indices 'ix0'. 'ix1' relate the above by :
    M.flat=v[ix0]
    L.flat=[ix1]=v or v=M.flat[ix1] or v=L.flat[ix1]
    """
    Ix=np.zeros((ni,ni),dtype=int)
    ix1=np.zeros((ni*(ni+1))/2,dtype=int)
    count=0
    for p in np.arange(ni) :
        for q in np.arange(p+1) :
            Ix[p,q]=count
            Ix[q,p]=count
            ix1[count]=p*ni+q
            count+=1
    ix0=Ix.flat[:]
    
    return ix0, ix1

def chol(v,ix0,ni) :
    """
    cv=chol(v,ix0,ni)
    cholesky decomposition, i.e. 'v=cv*cv.T' with cv lower
    triangular
    """
    nv=(ni*(ni+1))/2
    ix=np.reshape(ix0,(ni,ni))
    cv=np.zeros(v.shape)
    d=np.zeros(v.shape)
    k=0
    cv[:,ix[k,k]]=np.sqrt(v[:,ix[k,k]])
    cv[:,ix[k+1:,k]]=v[:,ix[k+1:,k]]/np.tile(cv[:,ix[k,k]],(ni-1-k,1)).T
    for k in np.arange(1,ni-1) :
        cv[:,ix[k,k]]=np.sqrt(v[:,ix[k,k]]-np.sum(cv[:,ix[k,:k]]**2,axis=1))
        d[:,:]=0
        for l in np.arange(k) :
            d[:,ix[k+1:,k]]=d[:,ix[k+1:,k]]+cv[:,ix[k+1:,l]]*np.tile(cv[:,ix[k,l]],(ni-k-1,1)).T
        cv[:,ix[k+1:,k]]=(v[:,ix[k+1:,k]]-d[:,ix[k+1:,k]])/np.tile(cv[:,ix[k,k]],(ni-1-k,1)).T
    k=ni-1
    cv[:,ix[k,k]]=np.sqrt(v[:,ix[k,k]]-np.sum(cv[:,ix[k,:k]]**2,axis=1))
    return cv

def cholU(v,ix0,ni) :
    """
    cv=cholU(v,ix0,ni)
    upper cholesky decomposition, i.e., 'v=cv.T*cv' 
    with cv lower triangular
    """
    nv=(ni*(ni+1))/2
    ix=np.reshape(ix0,(ni,ni))
    ix=ix[-np.arange(1,ni+1),:][:,-np.arange(1,ni+1)]
    cv=np.zeros(v.shape)
    d=np.zeros(v.shape)
    k=0
    cv[:,ix[k,k]]=np.sqrt(v[:,ix[k,k]])
    cv[:,ix[k+1:,k]]=v[:,ix[k+1:,k]]/np.tile(cv[:,ix[k,k]],(ni-1-k,1)).T
    for k in np.arange(1,ni-1) :
        cv[:,ix[k,k]]=np.sqrt(v[:,ix[k,k]]-np.sum(cv[:,ix[k,:k]]**2,axis=1))
        d[:,:]=0
        for l in np.arange(k):
            d[:,ix[k+1:,k]]=d[:,ix[k+1:,k]]+cv[:,ix[k+1:,l]]*np.tile(cv[:,ix[k,l]],(ni-k-1,1)).T
        cv[:,ix[k+1:,k]]=(v[:,ix[k+1:,k]]-d[:,ix[k+1:,k]])/np.tile(cv[:,ix[k,k]],(ni-1-k,1)).T
    k=ni-1
    cv[:,ix[k,k]]=np.sqrt(v[:,ix[k,k]]-np.sum(cv[:,ix[k,:k]]**2,axis=1))
    return cv

def invL(L,ix0,ni):
    """
    iL=invL(L,ix0,ni)
    inverse for lower triangular matrices
    """
    nv=(ni*(ni+1))/2
    ix=[]
    iL=np.zeros(L.shape)
    d=np.zeros((L.shape[0],ni))
    for k in np.arange(ni):
        ix.append(ix0[np.arange(ni-k)*(ni+1)+k*ni])
    iL[:,ix[0]]=1/L[:,ix[0]]
    for k in np.arange(1,ni):
        d[:,:]=0
        for l in np.arange(k):
            d[:,k:]=d[:,k:]+iL[:,ix[l][k-l:ni-l]]*L[:,ix[k-l][0:ni-k]]
        iL[:,ix[k]]=-d[:,k:]/L[:,ix[0][:ni-k]]
    return iL

def prod_LLT(L,ix0,ni):
    """
    LLT=prod_LLT(L,ix0,ni)
    product of lower triangular matrix and its transpose (L*L.T)
    """
    nv=(ni*(ni+1))/2
    ix=np.reshape(ix0,(ni,ni))
    LLT=np.zeros((L.shape[0],nv))
    for k in np.arange(ni):
        for l in np.arange(k+1):
            LLT[:,ix[k:,k]]=LLT[:,ix[k:,k]]+L[:,ix[k:,l]]*np.tile(L[:,ix[k,l:l+1]],(1,ni-k))
    return LLT

def prod_LTL(L,ix0,ni):
    nv=(ni*(ni+1))/2
    ix=np.reshape(ix0,(ni,ni))
    LTL=np.zeros((L.shape[0],nv))
    for k in np.arange(ni):
        for l in np.arange(k+1):
            LTL[:,ix[ni-1-k,:ni-k]]=LTL[:,ix[ni-1-k,:ni-k]]+L[:,ix[ni-1-l,:ni-k]]*np.tile(L[:,ix[ni-1-l:ni-l,ni-1-k]],(1,ni-k))
    return LTL

def M_sigval_bound(v,ix0,ni):
    """
    mxsig-M_sigval_bound(v,ix0,ni)
    calculates an upper bound for the maximum singular value of the symmetric matrix specified by 'v'
    """
    ix=np.reshape(ix0,(ni,ni))
    r=np.zeros((v.shape[0],ni))
    c=np.zeros((v.shape[0],ni))
    mxsig=np.zeros(v.shape[0])
    for k in np.arange(ni) :
        r[:,k]=np.sum(np.abs(v[:,ix[k,:]]),axis=1)
        c[:,k]=np.sum(np.abs(v[:,ix[k,:]]),axis=1)
    mxsig=np.sqrt(np.max(r,axis=1)*np.max(c,axis=1))
    return mxsig

