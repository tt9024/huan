"Contains functions for the fast calculation of the trace of the inverse \n\
of tridiagonal, pentadiagonal, and block tridiagonal matrices \n\
python functions denoted by 'py_' prefix, C functions with no previx."

import os
import sys

PYTHON_CODE_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_modules_dir=PYTHON_CODE_dir+'/C_models'

dirs_to_add=[C_modules_dir]
sys.path=sys.path+[d for d in dirs_to_add if d not in sys.path]

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
    return TI.sym_tri(d0, d1)

def sym_penta(d0, d1, d2) :
    "Same but with 5 strips"
    if not d0.flags.contiguous:
        d0=array(d0)
    if not d1.flags.contiguous:
        d1=array(d1)
    if not d2.flags.contiguous:
        d2=array(d2)
    return TI.sym_penta(d0, d1, d2)

def py_sym_tri(d0, d1) :
    """
    the trace of inverse of symetric matrix with that 
    3 strips along main diagonals
    """
    n=d0.shape[0]
    A=np.array([d0[n-1],0]);B=np.array([0,d0[n-2],0])
    for k in np.arange(1, n-1):
        A[1]=d0[n-1-k]*A[0]-d1[n-1-k]**2
        B[2]=d0[n-2-k]*(A[0]+B[1])-d1[n-2-k]**2*(1+B[0])
        B[:2]=B[1:]/A[0];A[0]=A[1]/A[0]
    B[2]=A[0]+B[1]
    A[1]=d0[0]*A[0]-d1[0]**2
    return (A[0]+B[1])/(d0[0]*A[0]-d1[0]**2)

def py_sym_penta(d0, d1, d2) :
    """
    why bother?
    """
    raise ValueError('not implemented')

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



