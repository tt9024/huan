import numpy as np
import scipy
import datetime
import multiprocessing as mp
import time
from matplotlib import pyplot as pl

##################################
###### ICA related stuffs ########
##################################
def test_ica(ncomp=7,nfeat=20,npca=None,nsamp=2000) :
    # the more complicated, the better
    x1=np.sin(np.arange(nsamp)/nsamp/100.0)
    x2=np.tanh((np.arange(nsamp)-nsamp/2.0)/192.0+3.0)
    x3=((np.arange(nsamp)-nsamp*0.4)/nsamp)**3.0
    #x4=np.cos(np.arange(2000)/412.0/1.2+1.2)*np.sin(np.sqrt((np.arange(2000)/1022.0+1.0)))
    x4=np.cos(np.arange(nsamp)/nsamp/120.0/1.2+1.2)

    x5=(np.arange(nsamp)-nsamp/2.0)/(nsamp/6.0)*(1.0/(1+np.exp(-np.arange(nsamp)/nsamp/3.0)))
    x6=np.random.normal(0,1,nsamp)
    x7=np.random.beta(2,5,nsamp)

    X=np.vstack((x1,x2,x3,x4,x5,x6,x7)).T
    mix=np.random.normal(0,1,(7,nfeat))
    Xm=np.dot(X,mix)
    Xm-=np.mean(Xm,axis=0)
    Xm/=np.std(Xm,axis=0)
    if npca is not None:
        u,s,vh=np.linalg.svd(Xm,full_matrices=False)
        Xm=np.dot(np.dot(u[:,:npca],np.diag(s[:npca])),vh[:npca,:])

    from sklearn.decomposition import FastICA
    ica=FastICA(n_components=ncomp)
    X0=ica.fit_transform(Xm)

    for i in np.arange(7) :
        pl.figure() ; pl.plot(X[:,i]); pl.title('X'+ str(i+1))
    for i in np.arange(ncomp) :
        pl.figure() ; pl.plot(X0[:,i]); pl.title('Xm'+str(i+1)), pl.plot(-X0[:,i])

def get_ica_ncomp_beta_yt(ncomp, X, Xt, Y) :
    from sklearn.decomposition import FastICA
    ica=FastICA(n_components=ncomp)
    x=ica.fit_transform(X)
    xt=ica.transform(Xt)
    beta=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),Y)
    yt=np.dot(xt,beta)
    return ica.components_.copy(), ica.mean_.copy(), beta, yt

def ica_eval0(lrd, ncomp, s0, e0, e1=None, yt_bars=None) :
    n,m=lrd.shape
    if e1 is None:
        e1 = n  # all samples
    if yt_bars is None:
        yt_bars = m # all bars
    X=lrd[s0:e0,:].copy()
    Xt=lrd[e0:e1,:].copy()
    Y=np.sum(lrd[s0+1:e0+1,:yt_bars],axis=1)
    Yt=np.sum(lrd[e0+1:e1+1,:yt_bars],axis=1)
    comp,mu,beta,yt=get_ica_ncomp_beta_yt(ncomp,X,Xt,Y)
    return comp,mu,beta,yt,Yt

def ica_eval(lrd) :
    """
    lrd: daily lr, shape n,m
    ncomp: ica ncomp
    s0, e0, start/end of training
    e1 end of testing, default till end
    yt_bars: number of bars to aggregate for target, default 1 day, i.e. m

    evaluate:
    1. the consistency of ica components
    2. the consistency of beta
    3. predictability of components
    """
    # 1. try consistency of 1 day ica
    for ncomp in [10, 20, 40, 100]:
        for d in [1000, 2000, 4000]:
            for s0 in [] :
                pass

