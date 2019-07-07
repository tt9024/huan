import numpy as np
import l1

def get_chol(lr) :
    """
    lr is a 2d matrix. Each row is an independent observation, each col is a fixed target.
    i.e. each row is a week, each col is a 5m bar.  This finds the AR coef from cholesky:
    Ty=e, where e is i.i.d. In order to get T, find E[y.T,y] as M, and L * D * L^T = M, where diag of L is 1.
    T = -inv(L). 

    Currently I just use raw lr to find such fitting.  I have 1112 weeks, but needs to fit 1380 bars.
    I found Monday is under fitting (too few features) and Friday over fitting (too many features). 

    Because more bars then weeks, I just randomly picked around 1000 bars to fit
    """
    n,m=lr.shape
    wt = l1.getwt(n,0.4)
    W = np.diag(wt/np.sum(wt))
    c = np.dot(np.dot(lr.T,W),lr)
    L=np.linalg.cholesky(c)
    d = np.diag(L)
    L/=d
    Li=-np.linalg.inv(L)
    Li[np.arange(m),np.arange(m)]=0
    return Li, d

def test_chol(Li, lr) :
    yt = np.dot(Li,lr.T).T




