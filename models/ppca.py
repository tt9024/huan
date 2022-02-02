import numpy as np

class kiscoSVD:
    def __init__(self, x):
        self.x=x.copy()

    def _uv1_(self,K1,K2):
        tol=1e-10
        maxiter=10
        e=1
        it=0
        while e > tol and it<maxiter:
            pass

def ppca0(Y0):
    """
    Consider this model (model 1)
    y_i = x_i + e_i, where y_i is a row, such as user's rating,
    x_i in N(0, K) and e_i in N(0, \lambda I)

    model 1:
    init: Y - whiten Y w.r.t. columns
    iter: K, lam
       * E(x_i)   = K ( K + \lam I)^{-1} Y
       * Cov(x_i) = K - K ( K + \lam I)^{-1} K

       K = 1/M * \sum_i^{M}{Cov(x_i) + E(x_i) * E(x_i)^T}
       \lam = 1/(N*M) \sum_{i,j} {C_{i,j} + [Y_{i,j} - E(X_{i,j})]^{2}}
    """

    Y = Y0.copy()
    Y-=np.mean(Y,axis=0)
    Y/=np.std(Y,axis=0)
    tol = 1e-10
    M,N = Y.shape
    err = 1
    lam=1
    u = np.mean(Y, axis=0)
    K = np.dot(Y.T,Y)

    while err > tol :
        # given u, K, \lambda
        Y1 = (Y.copy().T - u).T
        X0= np.dot( K, np.linalg.inv(K + lam*np.eye(N)))
        X = np.dot( X0, Y)
        C = K - np.dot(X0, K)

        # iter
        K = 1.0/M 

def L1SVD(C):
    pass

def kernelSVD(C, Ku, Kv, alph_u, alph_v) :
    """
    """
    pass

def kernalSmooth(lr) :
    """
    procedure: 
    1. estimate output structure, establish a day kernel
    2. SVD on input structure, with day kernel from 1
    3. repeat 1, put in bar kernel from 2
    """
    lr0=lr.copy()
    n,m=lr0.shape
    pass

def iterativeKernelSVD(C) :
    pass

def kernelSmooth(C) :
    pass

class CF :
    """
    The object for study and predict based on a matrix of C in a collaborative way
    state space model, state identification with SVD.
    The input space basis vs the output basis
    Mixing st and lt features may help identify the state

    iterate: 
        base:
        1. develop features inputs basis
        2. given target selection, identify targets basis
        3. match inputs/target with SVD fitting
        4. update inputs basis/target basis

        iterate:
            online fitting with error inputs
            risk optimization
        adjust target selection

    target: st targets for exec  (flow,dob)
            intra-day target, risk

    * position optimization:
      run simulations
    """
    def __init__(self, p):
        pass

