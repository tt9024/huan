###########
# SMC code
###########

import numpy as np
from matplotlib import pylab as pl

######################
# class of functionsa
# excercise 1: generate a non-markovian latency time series
# excercise 2: fit it usinng IS and SMC to find the Zt
# excercise 3: resample it
######################

def gpdf(x,m,v) :
    return np.exp(-(x-m)**2/(2*v))/np.sqrt(2*np.pi*v)

def gen_gaussian(u, cov, size=1) :
    """
    generate n guassian random variables with mean u and covariance cov
    u: kx1 mean vector
    cov: k-by-k covariance 
    """
    ret = []
    L=np.linalg.cholesky(cov)
    n=len(u)
    for i in np.arange(size) :
        ret.append(np.dot(L,np.random.normal(size=n))+u)
    return np.array(ret)
    
def linear_confidence_range(H, X, cov, std_mul=2.5) :
    """
    getting the range based on the Z = HXk. 
    Suppose Xk in [Xk-\sigma * std_mul, Xk+\sigma * std_mul], 
    where \sigma_i is std of xk_i
    get range of Zi in Z. 
    Input: 
        X: the state of shape [N,nx]
        cov: the covariance Pkpp, shape [N, nx, nx]
        H: set to None to use the object's H, shape [nz,nx]
    Return:
        Zmin, Zmax, the min, max of each shape [N,nz]
    """
    N,nx=X.shape
    nz = H.shape[0]
    assert H.shape[1] == nx
    Xs = np.sqrt(cov[:,np.arange(nx),np.arange(nx)])
    Xx = np.hstack((X-Xs,X+Xs))
    zmax = []
    zmin = []
    for hi in np.arange(nz) :
        maxix=((np.sign(H[hi,:]).astype(int)+1)/2)
        minix=1-maxix
        for ix, zm in zip([maxix, minix], [zmax,zmin]) :
            #zm.append(np.dot(H, Xx[:,ix].T))
            zm.append(np.dot(H[hi,:], Xx[:,ix*nx+np.arange(nx)].T))
    return np.array(zmin).T, np.array(zmax).T

class SMC :
    def __init__(self, nx, ny, F, G) :
        """
        F have functions:
            gen( X ) : generate xt given all previous X, X represents x[0..t-1]
            prob( X, xt ) : calculate probability of xt given X, X represents x[0..t-1]
        G have functions: 
            gen ( X ) : generate yt given all previous X, X represents x[0..t]
            prob( X, yt) : calculate probability of yt given X, X represents x[0..t]
        """
        self.nx = nx
        self.ny = ny
        self.F = F
        self.G = G

    def _gen(self, f, g, N, x0=None) :
        """
        generate N x and y based on f and g, where
        x[t] = f(x[:t]) and
        y[t] = g(x[:t+1])
        """
        X = np.zeros((N+1, self.nx))
        Y = np.zeros((N+1, self.ny))
        if x0 is None :
            x0 = np.zeros(self.nx)
        X[:,0] = x0.copy()
        Y[:,0] = G.gen(X[:,:1])
        for i in np.arange(n) + 1 :
            X[:,i] = f(X[:, :i])  # gen next xt
            Y[:,i] = g(X[:,:i+1]) # gen yt based on x[0..t]
        return X, Y

class SimpleStateSpace :
    def __init__(self, nx, nz, Phi, H, Q, R) :
        """
        This is the baseline model
        Notation from Chapter 11 Tutorial: The Kalman Filter (from MIT)
        x_{k+1} = \Phi x_k + w_k
        z_k = H x_k + v_k
        Q = E [ w_k w_k^T ]
        R = E [ v_k v_k^T ]
        P_k = E [ e_k e_k^T ] = E [ (x_k - \hat(x_k)) (x_k - \hat(x_k))^T ]

        And, the cool stuff: 
        \hat(x_k) = \hat(x_k)' + K_k(z_k - H\hat(x_k)')     (1)

        Find best K_k that minimize the P_k. Turns out that: 

        K_k = P_k^{'} H^T (H P_k^{'} H^T + R)^{-1}          (2)
        P_k = (I - K_k H)P_k^{'}                            (3)
        P_{k+1}^{'} = \Phi P_k \Phi^T + Q                   (4)
        
        (3) is because the error in state is an AR1 process: 
        e_{k+1}^{'} = \Phi e_k + w_k 

        ========
        input : 
            Phi : nx-by-nx transition matrix for X
            H:    nz-by-nx measurement matrix for Z
            Q:    nx-by-nx state transition noise cov matrix 
            R:    nz-by-nz measuurement noise cov matrix
        """
        self.nx = nx
        self.nz = nz
        self.Phi = Phi.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()

    def gen(self, N, x0) :
        wk = gen_gaussian(np.zeros(self.nx), self.Q, size=N)
        vk = gen_gaussian(np.zeros(self.nz), self.R, size=N)

        X = []
        Y = []
        xk = x0
        for i in np.arange(N) :
            xk = np.dot(self.Phi, xk) + wk[i,:]
            yk = np.dot(self.H, xk) + vk[i,:]
            X.append(xk)
            Y.append(yk)
            
        return np.array(X), np.array(Y), wk, vk

    def fit_unknown_cov(self, x0, Z) :
        """
        fit the model based up to Z[-1]
        """
        K, nz = Z.shape
        assert(nz == self.nz)
        nx=self.nx
        Q = np.eye(nx)
        R = np.eye(nz)
        #R = self.R.copy()
        #Pkp = np.zeros((nx,nx))
        Pkp = Q.copy()
        xkh = x0.copy()
        X = []
        Zp = []
        Pkpp = []
        Kkp = []
        zp = np.zeros(nz)
        for k, zk in enumerate (Z) :
            Kk = np.dot(Pkp,np.dot(self.H.T,np.linalg.inv(np.dot(self.H, np.dot(Pkp, self.H.T))+R)))
            Kkp.append(Kk.copy())

            # update the estimte
            wkp =  np.dot(Kk, (zk - np.dot(self.H, xkh)))
            xkh = xkh + wkp
            X.append(xkh)
            vkp = zk - zp

            # update the covariance
            Q = (k*Q+np.outer(wkp,wkp))/(k+1)
            R = (k*R+np.outer(vkp,vkp))/(k+1)
            Pk = np.dot((np.eye(nx)-np.dot(Kk,self.H)),Pkp)

            # project 
            xkh = np.dot(self.Phi,xkh)
            zp = np.dot(self.H, xkh)
            Pkp = np.dot(np.dot(self.Phi, Pk), self.Phi.T) + Q
            Zp.append(zp)
            Pkpp.append(Pkp)
        return np.array(X), np.array(Zp), np.array(Pkpp), np.array(Kkp), Q, R


    def fit(self, x0, Z, Q = None, R = None) :
        """
        fit the model based up to Z[-1]
        """
        K, nz = Z.shape
        assert(nz == self.nz)
        nx=self.nx
        if R is None:
            R = self.R.copy()
        if Q is None:
            Q = self.Q.copy()
        #Pkp = np.zeros((nx,nx))
        Pkp = Q.copy()
        xkh = x0.copy()
        X = []
        Zp = []
        Pkpp = []
        for k, zk in enumerate (Z) :
            Kk = np.dot(Pkp,np.dot(self.H.T,np.linalg.inv(np.dot(self.H, np.dot(Pkp, self.H.T))+R)))

            # update the estimte
            xkh = xkh + np.dot(Kk, (zk - np.dot(self.H, xkh)))
            X.append(xkh)

            # update the covariance
            Pk = np.dot((np.eye(nx)-np.dot(Kk,self.H)),Pkp)

            # project 
            xkh = np.dot(self.Phi,xkh)
            zp = np.dot(self.H, xkh)
            Pkp = np.dot(np.dot(self.Phi, Pk), self.Phi.T) + Q
            Zp.append(zp)
            Pkpp.append(Pkp)
        return np.array(X), np.array(Zp), np.array(Pkpp)

    def plot(self, X, Zp, Pkpp, Z) :
        Zmin, Zmax = self.err_range(X, Pkpp)
        fig=pl.figure()
        ax=[]
        for i in np.arange(self.nz):
            ax1=fig.add_subplot(self.nz,1,i+1)
            ax1.set_title('Z'+str(i))
            ax1.plot(Z[1:,i], label='given')
            ax1.plot(Zp[:-1,i],label='Pred')
            ax1.plot(Zmin[:-1,i],'-.', label='Pred lower')
            ax1.plot(Zmax[:-1,i],'-.', label='Pred upper')
            ax1.legend(loc='best')
            ax1.grid()
        pl.show()

    def err_range(self, X, cov, std_mul=2.5, H=None) :
        """
        getting the range based on the Z = HXk. 
        Suppose Xk in [Xk-\sigma * std_mul, Xk+\sigma * std_mul], 
        where \sigma_i is std of xk_i
        get range of Zi in Z. 
        Input: 
            X: the state of shape [N,nx]
            cov: the covariance Pkpp, shape [N, nx, nx]
            H: set to None to use the object's H, shape [nz,nx]
        Return:
            Zmin, Zmax, the min, max of each shape [N,nz]
        """
        return linear_confidence_range(self.H,X,cov,std_mul=std_mul)

def get_sss(nx, nz, Phi=None, H=None, Q=None, R=None) :
    if Phi is None:
        Phi = np.random.normal(0, 0.3, size=(nx,nx))

    if H is None:
        H = np.random.normal(0, 2, size=(nz,nx))

    if Q is None :
        Q = np.random.normal(0, 0.2, size=(nx,nx))
        Q = np.dot(Q.T,Q)/nx
        Q[np.arange(nx), np.arange(nx)]=Q.diagonal()*4

    if R is None :
        R = np.random.normal(0, 0.1, size=(nz,nz))
        R = np.dot(R.T,R)/nx
        R[np.arange(nz), np.arange(nz)]=R.diagonal()*4

    return SimpleStateSpace(nx,nz,Phi,H,Q,R)

def smc_sample(Xn, p, q, prev_sample) :
    """
    samples, weights = smc_sample(Xn, p, q, prev_sample)
    input: 
    Xn: 
    """
    pass

class ex1 :
    """
    refer to element of sequential mc page 19
    The benefit of making the proposal as the f, and if f is markovian, 
    is to simplify 
    """

    def __init__(self, N, phi = 0.75, q = 1.5, B = 0.7, r = 1) :
        self.phi = phi
        self.q = q
        self.B = B
        self.r = r
        self.N = N
        self.lw = np.zeros(N)
    
    def f(self, x, x0) :
        m = x0*self.phi
        return gpdf(x,m,self.q)

    def g(self, y, x) :
        t = len(x)
        m = x*np.exp(np.arange(t)[::-1]*np.log(B))
        return gpdf(y, m, self.r)

    def tgt(self, x, y) :
        n = len(x)
        x0 = np.r_[0, x]
        lp = 0
        for i in np.arange(n) + 1 :
            lp += (np.log(self.f(x[i], x[i-1]))+np.log(self.g(y[i-1], x[:i+1])))
        return lp

    def proposal(self, x, x0) :
        return self.f(x,x0)

    def proposal_gen(self, x0) :
        # gen according to f
        x = x0[-1]*self.phi + np.random.normal(0,np.sqrt(self.q))
        return x

    def gen_next(self, x0, y) :
        x1 = self.proposal_gen(x0)

    def Z(self) :
        return np.sum(np.exp(self.lw))/self.N

    def get_z_ratio(self, lwt) :
        Z0 = self.Z()
        self.lw = lwt
        Z1 = self.Z()
        return Z1/Z0
