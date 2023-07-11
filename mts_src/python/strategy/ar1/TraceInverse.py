import os
import sys

import numpy as np
import scipy.linalg as lg
from scipy.sparse.linalg import aslinearoperator as LO
#import _TrInv as TI  # from C_modules_dir

def sym_tri(d0, d1) :
    "Ti=sym_tri(d0, d1)\n\
    python wrapped C code for calculating the trace of the inverse of a symmetric tridiagonal matrix \n\
    Input: \n\
    'd0' = main diagonal with d0[0] being the [0,0] entry\n\
    'd1' = sub and super diagonal with d1[0] being the [0,1] and [1,0] entry \n\
    Outputs: \n\
    'Ti' = trace of inverse of trigiagonal matrix\n"
    if not d0.flags.contiguous:
        d0=array(d0)
    if not d1.flags.contiguous:
        d1=array(d1)
    return py_sym_tri(d0, d1)

def sym_penta(d0, d1, d2) :
    "Same but with 5 strips"
    if not d0.flags.contiguous:
        d0=array(d0)
    if not d1.flags.contiguous:
        d1=array(d1)
    if not d2.flags.contiguous:
        d2=array(d2)
    return py_sym_penta(d0, d1, d2)

def py_sym_tri(d0, d1) :
    """
    See the 
    """
    n=len(d0)
    assert (n>4)
    Q = np.eye(n)

    # create a real B
    B = np.zeros((n,n))
    B[np.arange(n),np.arange(n)]=d0
    B[np.arange(n-1)+1,np.arange(n-1)]=d1[:n-1]
    B[np.arange(n-1),np.arange(n-1)+1]=d1[:n-1]

    for i in np.arange(n-1) :
        a0=B[i,i]
        for X in [Q, B] :
            qi=X[i,:i+2]/a0
            X[i+1,:i+2]-=qi*B[i+1,i]

    Q2=LTinv(B.T,1)
    s=0
    for i in np.arange(n) :
        s+=np.dot(Q2[i,:i+1],Q[i,:i+1])
    return s

def LTinv(B, m) :
    """
    inverse of lower triang via guass elim, 
    B: n by n lower triangular matrix
    m: number of non-zero sub-diag, 
       m=0 is diagonal, m=n-1 fully populated
    return:
       inverse of B
    """
    n=B.shape[0]
    assert (m<=n)
    Q = np.eye(n+m)
    for i in np.arange(n) :
        Q[i,:i+1]/=B[i,i]
        qi=Q[i,:i+1]
        j=min(i+m+1,n)
        for m0 in np.arange(j-i-1)+1+i :
            Q[m0,:i+1]-=qi*B[m0,i]

    return Q[:n,:n]

def py_sym_penta(d0, d1, d2) :
    """
    why bother? just put it in for now
    Simple proceedure, just do a gaussian elimination
    Let Z being the matrix
    Q1*Z=B, where B is upper triag
    Q2*Bt=Zd, where Zd is the diagonal of B
    therefore
    Q2*Z_t*Q1_t=Zd or Q1*Z*Q2_t=Zd, 
    Z_{-1}=Q2_t*Zd_{-1}*Q1
    tr(Z{-1})=tr(Q2_t*Zd_{-1}*Q)

    By the way, C will run it very fast
    Maybe I could use the previous version to
    opotimize this further, keeping trace of Zd, Z1, Z2,
    But Z2 will always be the same, so I don't need to
    keep that big B.
    Also is there a way to parallel it?

    Other possible improvements:
    1. avoid operating a N-by-N matrix by not maintaining
       entire B, instead, just maintain 3-by-3 state Qi and
       simultaneously do row and col elimination.
       (See py_sym_penta_save() at the end)
    2. possible split it into blocks and run it at once
    """
    n=len(d0)
    assert (n>4)
    Q = np.eye(n)

    # create a real B
    B = np.zeros((n,n))
    B[np.arange(n),np.arange(n)]=d0
    B[np.arange(n-1)+1,np.arange(n-1)]=d1[:n-1]
    B[np.arange(n-1),np.arange(n-1)+1]=d1[:n-1]
    B[np.arange(n-2)+2,np.arange(n-2)]=d2[:n-2]
    B[np.arange(n-2),np.arange(n-2)+2]=d2[:n-2]

    for i in np.arange(n-2) :
        a0=B[i,i]
        for X in [Q, B] :
            qi=X[i,:i+3]/a0
            X[i+1,:i+3]-=qi*B[i+1,i] 
            X[i+2,:i+3]-=qi*B[i+2,i]

    # do the last line
    i=n-2
    a0=B[i,i]
    for X in [Q, B] :
        qi=X[i,:i+2]/a0
        X[i+1,:i+2]-=qi*B[i+1,i]

    #tr(Z{-1})=tr(Q2_t*Zd_{-1}*Q), where Zd=I
    # just sum(Q2*Q1)
    Q2=LTinv(B.T,2)
    s=0
    for i in np.arange(n) :
        s+=np.dot(Q2[i,:i+1],Q[i,:i+1])
    return s

def py_sym_block_tri(d0, d1):
    "Ti=py_sym_block_tri(d0,d1)\n\
    python code for calculating the trace of the inverse of a symmetric block trigiagonal matrix \n\
    Inputs:\n\
    'd0' = 3d numpy array for the main diagonal with d0[0,:,:] being the [0,0] block\n\
    'd1' = 3d numpy array for the sub and super diagonal with d1[0,:,:] being the [1,0] and [0,1]' blocks\n\
        the d1[-1,:,:]  block is ignored \n\
    Outputs: \n\
    'Ti' = trace of inverse of block tridiagonal matrix\n"

    n=d0.shape[0]
    D=np.zeros(d0.shape)
    D[0,:,:]=lg.inv(d0[n-1,:,:])
    for k in np.arange(1,n):
        D[k,:,:]=lg.inv(d0[n-1-k,:,:]-np.dot(d1[n-1-k,:,:].T,np.dot(D[k-1,:,:],d1[n-1-k,:,:])))
    for k in np.arange(n-2,-1,-1):
        D[k,:,:]=D[k,:,:]+np.dot(D[k,:,:],np.dot(d1[n-2-k,:,:],np.dot(D[k+1,:,:],np.dot(d1[n-2-k,:,:].T,D[k,:,:]))))

    return np.sum(np.trace(D,axis1=1,axis2=2))

def py_sym_block_triI(d0, d1):
    "Ti=py_sym_block_triI(d0,d1)\n\
    python code for calculating the trace of the inverse of a symmetric block trigiagonal matrix \n\
    Inputs:\n\
    'd0' = 3d numpy array for the main diagonal with d0[0,:,:] being the [0,0] block\n\
    'd1' = 1d numpy array for the sub and super diagonal with d1[0]*np.identity(m)  \n\
           being the [1,0] and [0,1]' blocks\n\
           the d1[-1,:,:]  block is ignored \n\
    Outputs: \n\
    'Ti' = trace of inverse of block tridiagonal matrix\n"

    n,m,k = d0.shape
    assert m==k, "d0 element not square"
    assert n==len(d1), "d0 and d1 length mismatch"

    d1h = np.zeros((n,m,m))
    d1h[:,np.arange(m),np.arange(m)]=np.tile(d1,(m,1)).T
    return py_sym_block_tri(d0, d1h)

def py_sym_block_triwc(d0,d1):
    "Ti=py_sym_block_triwc(d0,d1)\n\
    python code for calculating the trace of the inverse of a symmetric block tridiagonal matrix\n\
    with corners\n\
    Inputs:\n\
    'd0'=3d numpy array for the main diagonal with d0[0,:,:] being the [0,0] block\n\
    'd1'=3d numpy array for the sub and super diagonal with d1[0,:,:] being the [1,0] and [0,1]' blocks\n\
         and d1[-1,:,:] being the [0,-1] and [-1,0]' blocks\n\
    Outputs: \n\
    'Ti'=trace of inverse of block tridiagonal matrix with corners\n"

    n=d0.shape[0]
    D=np.zeros(d0.shape)
    D[0,:,:]=lg.inv(d0[n-1,:,:])
    D[n-1,:,:]=d0[0,:,:]
    A=np.zeros(d0.shape)
    A[0,:,:]=d1[n-1,:,:].T
    for k in np.arange(1,n-1):
        D[k,:,:]=lg.inv(d0[n-1-k,:,:]-np.dot(d1[n-1-k,:,:].T,np.dot(D[k-1,:,:],d1[n-1-k,:,:])))
        D[n-1,:,:]=D[n-1,:,:]-np.dot(A[k-1,:,:].T,np.dot(D[k-1,:,:],A[k-1,:,:]))
        A[k,:,:]=-np.dot(d1[n-1-k,:,:].T,np.dot(D[k-1,:,:],A[k-1,:,:]))
    A[n-2,:,:]=A[n-2,:,:]+d1[0,:,:]
    D[n-1,:,:]=lg.inv(D[n-1,:,:]-np.dot(A[n-2,:,:].T,np.dot(D[n-2,:,:],A[n-2,:,:])))
    dD=np.dot(D[n-2,:,:],np.dot(A[n-2,:,:],np.dot(D[n-1,:,:],np.dot(A[n-2,:,:].T,D[n-2,:,:]))))
    A[n-2,:,:]=-np.dot(D[n-2,:,:],np.dot(A[n-2,:,:],D[n-1,:,:]))
    D[n-2,:,:]=D[n-2,:,:]+dD
    for k in np.arange(n-3,-1,-1):
        dD=np.dot(A[k,:,:],np.dot(D[n-1,:,:],A[k,:,:].T)+np.dot(A[k+1,:,:].T,d1[n-2-k,:,:].T))
        dD=dD+np.dot(d1[n-2-k,:,:],np.dot(A[k+1,:,:],A[k,:,:].T)+np.dot(D[k+1,:,:],d1[n-2-k,:,:].T))
        dD=np.dot(D[k,:,:],np.dot(dD,D[k,:,:]))
        A[k,:,:]=-np.dot(D[k,:,:],np.dot(A[k,:,:],D[n-1,:,:])+np.dot(d1[n-2-k,:,:],A[k+1,:,:]))
        D[k,:,:]=D[k,:,:]+dD

    return np.sum(np.trace(D,axis1=1,axis2=2))

####
#= Backup for future possible optimization
#  See py_sym_penta() on improvements using state Qi
#####
def py_sym_penta_save(d0, d1, d2) :
    n=len(d0)
    assert (n>4)
    Q = np.eye(n)
    Zd = np.empty(n)

    # Qi is 3 by 3 matrix keeping track of Zd
    Qi=np.array([[d0[0],d1[0],d2[0]],[d1[0],d0[1],d1[1]],[d2[0],d1[1],d0[2]]])
    d00=np.r_[d0,0]
    d10=np.r_[d1,0]
    d20=np.r_[d2,0]
    for i in np.arange(n-2) :
        a0=Qi[0,0]
        Zd[i]=a0
        qi=Q[i,:i+1]/a0
        Q[i+1,:i+1]-=qi*Qi[1,0] 
        Q[i+2,:i+1]-=qi*Qi[2,0]

        # also include Qi update
        Qi[1,1:]-=Qi[0,1:]/a0*Qi[1,0]
        Qi[2,1:]-=Qi[0,1:]/a0*Qi[2,0]
        Qi[:2,:2]=Qi[1:,1:]
        Qi[2,:]=np.array([d20[i+1],d10[i+2],d00[i+3]])
        Qi[:-1,2]=np.array([d20[i+1],d10[i+2]])

    # do the last line
    i=n-2
    a0=Qi[0,0]
    Zd[i]=a0
    qi=Q[i,:i+1]/a0
    Q[i+1,:i+1]-=qi*Qi[1,0]

    Qi[1,1:]-=Qi[0,1:]/a0*Qi[1,0]
    Zd[i+1]=Qi[1,1]

    return Q,Zd

