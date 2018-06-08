import numpy as np
import l1_reader
def lr5s_30m(wb5s_lr) :
    """
    wb5s has 5sec bar observed between 10:00:05am to 17:00:00pm each fday
    This gets feature of the first 30minute
    """
    wb5s=wb5s_lr
    Y1=np.sum(wb5s[:,-60*60+30*12:-60*60+149*12],axis=1)

    s0=np.sum(wb5s[:,-60*60+1:-60*60+6],axis=1)
    s0_1=np.sum(wb5s[:,-60*60+6:-60*60+30*12],axis=1)

    s1=np.sum(wb5s[:,-60*60+6:-60*60+74],axis=1)
    s1_1=np.sum(wb5s[:,-60*60+74:-60*60+30*12],axis=1)

    s2=np.sum(wb5s[:,-60*60+120:-60*60+148],axis=1)
    s2_1=np.sum(wb5s[:,-60*60+148:-60*60+30*12],axis=1)

    s3=np.sum(wb5s[:,-60*60+150:-60*60+256],axis=1)
    s3_1=np.sum(wb5s[:,-60*60+256:-60*60+30*12],axis=1)

    return wb5s, np.vstack((s0,s1,s2,s3)).T, np.vstack((s0_1,s1_1,s2_1,s3_1)).T


def fday_5s(lrfn='wb5s_fday_lr.npz', vbsfn='wb5s_fday_vbs.npz') :
    wb5s=np.load('wb5s_fday_lr.npz')['lr']
    wb5sv=np.load('wb5s_fday_vbs.npz')['vbs']
    ix0=2*12*60
    lr5s=[]
    vbs5s=[]
    for h,m in [(12,05),(12,10),(12,30),(14,10),(14,28),(14,29),(14,30),(14,40),(15,0),(16,30),(16,59)] :
        ix1=(h-10)*(12*60)+m*12
        lr5s.append(np.sum(wb5s[:,ix0:ix1],axis=1))
        vbs5s.append(np.sum(wb5sv[:,ix0:ix1],axis=1))
        ix0=ix1
    return wb5s, wb5sv, np.array(lr5s).T, np.array(vbs5s).T

def cor5m(lr5s, yh=None, a=0.125, use_sign=False) :
    n, k=lr5s.shape
    #normalize
    if yh is not None :
        yh0=yh/yh.std()
    else :
        yh0=np.ones(n)
    assert len(yh0) == n
    if use_sign:
        yh0=np.sign(yh0)
    lr5s0=lr5s/np.std(lr5s,axis=0)
    scl=0
    c=np.zeros((k,k))
    for lr, y in zip(lr5s0, yh0) :
        c=(1-a)*c + a*np.outer(lr,lr)*y
        scl=scl*(1-a)+a
    c/=scl
    #norm
    sd=np.sqrt(np.diag(c))
    c/=np.outer(sd,sd)
    return c, scl

def upsample(wb5s, mul) :
    n,k=wb5s.shape
    assert k/mul*mul == k, 'mul not multiple of colume size'
    lrc=np.cumsum(np.vstack((np.zeros(n),wb5s.T)).T,axis=1)
    col=np.arange(0,k+1,mul)
    return lrc[:,col[1:]]-lrc[:,col[:-1]]

def shp_cm(wb5s,yh,wt_decay) :
    sc=np.cumsum(wb5s.T*np.sign(yh),axis=0).T
    wt=l1_reader.getwt(sc.shape[0],wt_decay)
    wt/=np.sum(wt)
    scm=np.dot(sc.T,wt)
    scsd=np.sqrt(np.dot((sc-scm).T**2, wt))
    #scm=np.mean(sc,axis=0)
    #scsd=np.std(sc,axis=0)
    return scm/scsd
