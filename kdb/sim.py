import numpy as np
from sklearn import linear_model
import ean
import l1_reader
import matplotlib.pylab as pl
import copy
import pdb

###  I need a model/method to select features automatically
###  to test out-of-sample
###  for example, pick segments of lr sum that has best correlation
###  for cross validated periods
###  vbs can also be considered

###  Haven't tried the splines/basis 
###  I feel that segment of lr sum is not as good as 
###  the current cycles being encoded in (rise and full)

### resist the temptation of overfitting and fear for fail

### bottom line: the features make sense. Can also 
### improve by looking into 2 minutes of lr and vbs

def Xlr(wb5m) : 
    #fd2=np.sum(wb5m[:,8*12+4:9*12+7,1],axis=1)
    #fd3=np.sum(wb5m[:,11*12+2:11*12+8,1],axis=1)
    fd0=np.sum(wb5m[:,6*12:8*12,1],axis=1)
    fd1=np.sum(wb5m[:,10*12+3:12*12,1],axis=1)
    fd2=np.sum(wb5m[:,18*12:19*12,1],axis=1)
    fd3=np.sum(wb5m[:,23*12-2:23*12+11,1],axis=1)

    # tuesday is tricky, changes a lot at the last year [-50:]
    fd4=np.sum(wb5m[:,(23+5)*12+5:(23+6)*12+2,1],axis=1)
    fd5=np.sum(wb5m[:,(23+10)*12+9:(23+12)*12+2,1],axis=1)
    fd6=np.sum(wb5m[:,(23+12)*12+6:(23+13)*12+2,1],axis=1)
    fd7=np.sum(wb5m[:,(23+18)*12+8:(23+19)*12+4,1],axis=1)
    fd8=np.sum(wb5m[:,(23+19)*12+4:(23+19)*12+8,1],axis=1)
    fd9=np.sum(wb5m[:,(23+21)*12+5:(23+22)*12+3,1],axis=1)

    # wednesday is also tricky, oscillates for the last 3 years
    fd10=np.sum(wb5m[:,(2*23+22)*12+3:(2*23+22)*12+11,1],axis=1)
    fd12=np.sum(wb5m[:,(2*23+18)*12+3:(2*23+19)*12+3,1],axis=1)

    fd11=np.sum(wb5m[:,(3*23+10)*12+9:(3*23+12)*12+2,1],axis=1)
    fd13=np.sum(wb5m[:,(3*23+17)*12+3:(3*23+18)*12,1],axis=1) # this feature is not import

    fd14=np.sum(wb5m[:,(4*23+0)*12-1:(4*23+0)*12+6,1],axis=1)  # has an overnight lr, note for roll
    fd15=np.sum(wb5m[:,-60-42:-60,1],axis=1) 

    ### whole days? 
    fd16=np.sum(wb5m[:,23*12*3+11:23*12*4,1],axis=1)
    fd17=np.sum(wb5m[:,23*12*1:23*12*3,1],axis=1)  # this is too volatile!
    X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd11,fd14,fd15,fd16)).T  # it seems that Monday,Thursday and Friday has the best
    #X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd11,fd14,np.sign(fd14)*fd14*fd14,fd15,fd16)).T
    #X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd11,fd14,fd15,fd16,np.sign(fd14)*fd11*fd14)).T  # it seems that Monday,Thursday and Friday has the best
    #X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd11,fd6,fd10,fd14,fd15,fd16)).T  # it seems that Monday,Thursday and Friday has the best

    #X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd11,fd14,fd15)).T  # it seems that Monday,Thursday and Friday has the best

    return X.copy()

def getYTrain(wb5m,y0=-60,y1=-30,wt_decay=0.1) :
    y=np.sum(wb5m[:,y0:y1,1],axis=1)
    wt=l1_reader.getwt(len(y),wt_decay)
    wt/=np.sum(wt)
    mu=np.dot(y,wt)
    sd=np.sqrt(np.dot(wt, (y-mu)**2))
    y0=ean.outlier(y,mu,sd,in_th=1,out_th=3)
    return y0, mu, sd

def getYTest(wb5m,mu,sd,y0=-60,y1=-30) :
    y=np.sum(wb5m[:,y0:y1,1],axis=1)
    #y0=ean.outlier(y,mu,sd,in_th=1,out_th=3)
    y0=y.copy()
    return y0

def YprevTrain(y0) :
    fd19=np.r_[np.zeros(2), y0[:-2]] 
    fd20=np.r_[np.zeros(18), y0[:-18]] # good negative corr at 18th week
    fd21=np.r_[np.zeros(32), y0[:-32]] # good positive corr at 32th week
    fd22=np.r_[np.zeros(35), y0[:-35]] # good positive corr at 35th week
    X=np.vstack((fd19,np.sign(fd22)*np.abs(fd21))).T
    #X=np.vstack((fd19,fd20,fd21,fd22)).T
    return X.copy()

def YprevTest(y0,ytrain) :
    y00=np.r_[ytrain, y0]
    n=len(y0)
    fd19=y00[-2-n:-2]
    fd20=y00[-18-n:-18] 
    fd21=y00[-32-n:-32]
    fd22=y00[-35-n:-35]
    X=np.vstack((fd19,np.sign(fd22)*np.abs(fd21))).T
    #X=np.vstack((fd19,fd20,fd21,fd22)).T
    return X.copy()

def get_wb5s(wb5s_fname, ylist,y0=-120*60,y1=None,col=1) :
    wb5s=[]
    for y in ylist :
        fn=wb5s_fname+str(y)+'.npz'
        w=np.load(fn)['bar']
        if y1 is None :
            y1=w.shape[1]
        wb5s.append(w[:,y0:y1,col].copy())
        w=[]
    return np.vstack(wb5s)

def Xvbs(vbs) :
    #monday
    #fd0=np.sum(vbs[:,6*12-2:7*12+6],axis=1)
    #fd1=np.sum(vbs[:,10*12+9:11*12+9],axis=1)

    #tuesday
    fd2=np.sum(vbs[:,(23+5)*12+8:(23+7)*12+6],axis=1)
    fd2_=np.sum(np.sign(vbs[:,(23+5)*12+8:(23+7)*12+6]),axis=1)

    #wednesday
    fd3=np.sum(vbs[:,(2*23+21)*12+5:(3*23+2)*12+6],axis=1)
    fd3_=np.sum(np.sign(vbs[:,(2*23+21)*12+5:(3*23+2)*12+6]),axis=1)

    fd4=np.sum(vbs[:,(2*23)*12+8:(2*23+2)*12+3],axis=1)

    #thursday
    #fd18=np.sum(vbs[:,23*12*3+11*12:23*12*4],axis=1)
    fd18=np.sum(vbs[:,23*12*3+11*12:23*12*4+1*12],axis=1)
    fd18_=np.sum(np.sign(vbs[:,23*12*3+11*12:23*12*4]),axis=1)

    fd19=np.sum(vbs[:,23*12+11*12:23*12*3],axis=1)
    fd17=np.sum(vbs[:,23*12*1+11*12:23*12*2],axis=1)
    fd16=np.sum(vbs[:,11*12:23*12],axis=1)

    #friday
    fd5=np.sum(vbs[:,(4*23+5)*12+2:(4*23+6)*12+8],axis=1)
    fd6=np.sum(vbs[:,(4*23+12)*12+7:(4*23+13)*12+7],axis=1)
    #fd5_=np.sum(np.sign(vbs[:,(4*23+5)*12+2:(4*23+6)*12+8]),axis=1)
    #fd6_=np.sum(np.sign(vbs[:,(4*23+12)*12+7:(4*23+13)*12+7]),axis=1)

    #event flow
    fd10=np.sum(vbs[:,(2*23)*12-7:(2*23)*12],axis=1)  #api
    fd11=np.sum(vbs[:,(2*23+16)*12+6:(2*23+18)*12],axis=1) #eia wed 10:30 to 12
    fd12=np.sum(vbs[:,(1*23+15)*12+6:(1*23+18)*12],axis=1) #jen mon 9:30 to 12

    # get more
    #X=np.vstack((fd2,fd3,fd4,fd5,fd6,fd18)).T

    #debug
    #X=np.vstack((fd2,fd18)).T
    X=np.vstack((fd2,fd3,fd18)).T
    #X=np.vstack((fd3,fd18)).T
    #X=np.vstack((fd2,fd3,fd18)).T
    return X.copy()

def Xvol0(v0) :
    #fd0=np.sum(v0[:,-84:-68],axis=1)  # this is gone
    fd1=np.sum(v0[:,(3*23+18)*12:(4*23+1)*12],axis=1)  #Thursday 12pm to Fri 7pm
    #X=np.vstack((fd0,fd1)).T
    X=fd1  # this doesn't hold anymore
    return X

def getXTrain(wb5m,y0=-60,y1=-30,wt_decay=0.1) :
    X1=Xlr(wb5m)
    X2=Xvbs(wb5m[:,:,3])
    Y0,mu,sd=getYTrain(wb5m,y0=y0,y1=y1,wt_decay=wt_decay)
    X3=YprevTrain(Y0)
    X=np.vstack((X1.T,X2.T,X3.T)).T
    #X=np.vstack((X1.T,X2.T,X3.T,(X1[:,6]*X2[:,0]).T)).T

    mu=np.mean(X,axis=0)
    sd=np.std(X,axis=0)

    X-=mu ; X/=sd; X=np.vstack((np.ones(wb5m.shape[0]),X.T)).T
    return X,mu,sd

def getXTest(wb5m,Ytrain,xmu,xsd,ymu,ysd,y0=-60,y1=-30) :
    X1=Xlr(wb5m)
    X2=Xvbs(wb5m[:,:,3])
    Y0=getYTest(wb5m,ymu,ysd,y0=y0,y1=y1)
    X3=YprevTest(Y0,Ytrain)
    X=np.vstack((X1.T,X2.T,X3.T)).T
    #X=np.vstack((X1.T,X2.T,X3.T,(X1[:,6]*X2[:,0]).T)).T
    X-=xmu ; X/=xsd ; X=np.vstack((np.ones(wb5m.shape[0]),X.T)).T
    return X

def getYh(Xtrain,Ytrain, Xtest, clf, wt_decay=None, feat_select=True, fit_sign=False) :
    #omp=linear_model.Lars()
    #omp=linear_model.OrthogonalMatchingPursuitCV()
    #omp=linear_model.LassoCV()
    #omp=linear_model.RidgeCV()

    omp=copy.deepcopy(clf)
    n,k=Xtrain.shape
    if feat_select :
        ix=feat_sel(Xtrain,Ytrain)
        if len(ix) == 0 :
            return np.ones(Xtest.shape[0])*Ytrain.mean(), omp, [0]
        ix=np.r_[0,ix]
    else :
        ix=np.arange(k)
    wt=None
    if wt_decay is not None :
        wt=l1_reader.getwt(len(Ytrain),wt_decay)
    if fit_sign:
        #remove zero cases
        ixz=np.nonzero(Ytrain!=0)[0]
        Xtrain=Xtrain[ixz,:]
        Ytrain0=np.sign(Ytrain[ixz])
        if wt is not None:
            wt=wt[ixz]*np.sqrt(np.abs(Ytrain[ixz]))
            #wt=wt[ixz]
            #pl.figure() ; pl.plot(np.abs(Ytrain)) ; pl.plot(np.sqrt(np.abs(Ytrain0)))
    else :
        Ytrain0=Ytrain

    omp.fit(Xtrain[:,ix], Ytrain0,sample_weight=wt)
    #ys=np.abs(Ytrain)
    #wty=np.sqrt(ys/ys.std()+1)
    #wty/=wty.std()
    #omp.fit(Xtrain[:,ix], Ytrain0,sample_weight=wt*wty)

    yh=omp.predict(Xtest[:,ix])
    return yh,omp,ix.copy()

def rollY(wb5m, train_idx, test_idx, clf, y0=-60, y1=-30, wt_decay=0.1,train_wt_decay=0.25,feat_select=True,fit_sign=False) :
    X,xmu,xsd=getXTrain(wb5m[train_idx,:,:],y0=y0, y1=y1,wt_decay=wt_decay)
    Y,ymu,ysd=getYTrain(wb5m[train_idx,:,:],y0=y0,y1=y1,wt_decay=wt_decay)
    Xt=getXTest(wb5m[test_idx,:,:],Y,xmu,xsd,ymu,ysd,y0=y0,y1=y1)
    Yt=getYTest(wb5m[test_idx,:,:],ymu,ysd,y0=y0,y1=y1)
    yh,omp,coefix=getYh(X, Y, Xt, clf,wt_decay=train_wt_decay,feat_select=feat_select,fit_sign=fit_sign)
    #print np.corrcoef(yh, Yt)[0,1]
    return X,Y,Xt,Yt,yh,omp,coefix

def wt_corr(x1d,y,wt=None,wt_decay=0.1,whiten=True) :
    n=len(x1d)
    assert n==len(y), 'x1d and y length mismatch'
    if wt is None :
        wt=l1_reader.getwt(n,wt_decay)
        wt/=np.sum(wt)

    x0=x1d.copy()
    y0=y.copy()
    if whiten :
        x0-=x0.mean()
        sd=x0.std()
        if sd > 1e-10 :
            x0/=sd
        y0-=y0.mean()
        sd=y0.std()
        if sd > 1e-10 :
            y0/=sd

    W=np.diag(wt)
    mc=np.dot(x0,np.dot(W,y0))
    ms0=x0*y0-mc
    ms= np.sqrt(np.dot(ms0,np.dot(W,ms0)))
    return mc, ms

def wt_corr2d(X2d,y,wt_decay=0.1,whiten=True) :
    n,k=X2d.shape
    assert n==len(y), 'X2d and y shape mismatch'
    wt=l1_reader.getwt(n,wt_decay)
    wt/=np.sum(wt)
    
    mcorr=[]
    mcstd=[]
    for i in np.arange(k) :
        mc,ms=wt_corr(X2d[:,i],y,wt=wt,whiten=whiten)
        mcorr.append(mc)
        mcstd.append(ms)
    mcorr=np.array(mcorr)
    mcstd=np.array(mcstd)
    return mcorr, mcstd

def feat_sel0(X,y,lt_cor_th=0.001,st_smp_cnt=16,wt_decay=0.25) :
    mcorr,mcstd=wt_corr2d(X,y,wt_decay=wt_decay,whiten=True)
    #lt=min(10*st_smp_cnt,len(y))
    #mcorr,mcstd=wt_corr2d(X[-lt:,:],y[-lt:],wt_decay=wt_decay,whiten=True)
    mcst, msst=wt_corr2d(X[-st_smp_cnt:,:],y[-st_smp_cnt:],wt_decay=0,whiten=True)
    
    mcs=mcorr/mcstd
    #check nan
    ixnan=np.nonzero(np.isnan(mcs))[0]
    mcs[ixnan]=0

    #mcsts=mcst/msst
    #ixnan=np.nonzero(np.isnan(mcsts))[0]
    #mcsts[ixnan]=0

    #ix=np.nonzero(mcs*mcsts>1e-3)[0]
    ix=np.nonzero(mcs*mcst>1e-2)[0]
    if len(ix) == 0 :
        print 'no feature selected'
        return ix

    # how about adding a second check
    #lt=min(3*st_smp_cnt,len(y))
    #mcorr2,mcstd2=wt_corr2d(X[-lt:,:],y[-lt:],wt_decay=wt_decay,whiten=True)
    #mcs2=mcorr2/mcstd2
    #ixnan=np.nonzero(np.isnan(mcs2))[0]
    #mcs2[ixnan]=0

    mcshp=mcorr[ix]/mcstd[ix]
    ix0=np.nonzero(np.abs(mcshp)>lt_cor_th)[0]
    #print len(ix0), (mcs*mcst)[ix]
    print len(ix0),
    return ix[ix0]

def feat_sel(X,y,lt_cor_th=0.0001,st_smp_cnt=16,wt_decay=0.25,vbose=False):
        #ix1=feat_sel2(X,y,lt_cor_th=lt_cor_th,st_smp_cnt=st_smp_cnt,wt_decay=wt_decay,vbose=vbose)

        ##ix2=feat_sel1(X,y,lt_cor_th=lt_cor_th,st_smp_cnt=st_smp_cnt*4,wt_decay=wt_decay)
        #ix1=feat_sel3(X,y,st_smp_cnt=st_smp_cnt,wt_decay=wt_decay)
        ix1=feat_sel4(X,y,st_smp_cnt=st_smp_cnt,vbose=False)
        
        #print ix1
        return ix1

def feat_sel1(X,y,lt_cor_th=0.0001,st_smp_cnt=16,wt_decay=0.25,st_th=1e-2,lt_th=1e-5,vbose=True) :
    mcorr,mcstd=wt_corr2d(X,y,wt_decay=wt_decay,whiten=True)
    mcst, msst=wt_corr2d(X[-st_smp_cnt:,:],y[-st_smp_cnt:],wt_decay=0,whiten=True)
    mcs=mcorr/mcstd
    #check nan
    ixnan=np.nonzero(np.isnan(mcs))[0]
    mcs[ixnan]=0

    st_th0=st_th
    while st_th > 1e-6 :
        ix=np.nonzero(mcs*mcst>st_th)[0]
        if len(ix) == 0 or len(ix) == 2 or len(ix) == 3:  #this is somewhat arbitary
            if len(ix) > 0  and st_th<st_th0/10.0 :
                break
            st_th*=0.9
        else :
            break
    if len(ix) == 0 :
        print 'no feature selected for st'
        return []
    print st_th, len(ix),
    lt=min(4*st_smp_cnt,len(y))
    mclt,mslt=wt_corr2d(X[-lt:-st_smp_cnt,ix],y[-lt:-st_smp_cnt],wt_decay=0,whiten=True)
    mclt/=mslt

    mcorrlt,mcstdlt=wt_corr2d(X[-lt*4:-lt,ix],y[-lt*4:-lt],wt_decay=0.1,whiten=True)
    mcorrlt/=mcstdlt
    while True :
        ix0=np.nonzero(mcorrlt*mclt>lt_th)[0]
        if len(ix0) == 0:
            lt_th *= 0.9
            if lt_th < 1e-6 :
                print 'no feature filtered at lt',
                break
        else :
            ix=ix[ix0]
            break

    mcshp=mcorr[ix]/mcstd[ix]
    ix0=np.nonzero(np.abs(mcshp)>lt_cor_th)[0]
    #print len(ix0), (mcs*mcst)[ix]
    print ix[ix0],
    return ix[ix0]

# This gets too many features
# it turns out that feat_sel1 is the best as it focus on st, and it gives
# the best predictive power. However feat_sel1 has one issue: 
# is very sensitive when adding or removing.  
# it may be useful to code up an explicity cross validation on the
# latest samples for inclusion of feature set
# or better, to come up with a model for such selection
def feat_sel2(X,y,lt_cor_th=0.0001,st_smp_cnt=16,wt_decay=0.25,st_th=1e-2,lt_th=4e-2,vbose=False) :
    mcorr,mcstd=wt_corr2d(X,y,wt_decay=wt_decay,whiten=True)
    # debug
    mcst1, msst1=wt_corr2d(X[-st_smp_cnt:,:],y[-st_smp_cnt:],wt_decay=0,whiten=True)
    mcst2, msst2=wt_corr2d(X[-st_smp_cnt-1:-1,:],y[-st_smp_cnt-1:-1],wt_decay=0,whiten=True)
    mcst3, msst3=wt_corr2d(X[-st_smp_cnt-2:-2,:],y[-st_smp_cnt-2:-2],wt_decay=0,whiten=True)
    mcst4, msst4=wt_corr2d(X[-st_smp_cnt-3:-3,:],y[-st_smp_cnt-3:-3],wt_decay=0,whiten=True)
    mcst=(mcst1*0.5+mcst2*0.3+mcst3*0.1+mcst4*0.1)
    #mcst=(mcst1+mcst2+mcst3)/3

    mcs=mcorr/mcstd
    #check nan
    ixnan=np.nonzero(np.isnan(mcs))[0]
    mcs[ixnan]=0

    lt=min(5*st_smp_cnt,len(y))
    mclt,mslt=wt_corr2d(X[-lt:-st_smp_cnt,:],y[-lt:-st_smp_cnt],wt_decay=0,whiten=True)
    mclt/=mslt
    #check nan
    ixnan=np.nonzero(np.isnan(mclt))[0]
    mclt[ixnan]=0

    ix0=np.nonzero(mclt*mcst>0)[0] #demand st and lt agree
    if len(ix0)==0 :
        print 'no feature selected'
        return []

    ix0=np.arange(len(mcs))
    # select either short term or long term
    ix1 = []
    st_th0=st_th
    while st_th >=st_th0/10:
        ix1=np.nonzero(mcs[ix0]*mcst[ix0]>st_th)[0]
        if len(ix1) == 0 :
            st_th*=0.9
        else :
            break
    if len(ix1) == 0 :
        print 'no feature selected for st'
        return []
    #print st_th, len(ix1),

    # select long term features
    mcorrlt,mcstdlt=wt_corr2d(X[-lt*4:-lt,:],y[-lt*4:-lt],wt_decay=0.1,whiten=True)
    mcorrlt/=mcstdlt
    lt_th0=lt_th
    while  lt_th >= lt_th0:
        ix2=np.nonzero(mcorrlt[ix0]*mclt[ix0]>lt_th)[0]
        if len(ix2) == 0:
            lt_th *= 0.9
        else :
            break

    #print lt_th,len(ix2),

    ix=np.union1d(ix0[ix1], ix0[ix2])
    #print len(ix),

    mcshp=mcorr[ix]/mcstd[ix]
    ix0=np.nonzero(np.abs(mcshp)>lt_cor_th)[0]
    #print len(ix0), (mcs*mcst)[ix]
    if vbose :
        print ix[ix0],
    return ix[ix0]

def score_sign_ref(clf,X0,Y0,Xt,Yt,wt=None) :
    clf0=copy.deepcopy(clf)
    clf0.fit(Xt,Yt) 
    yh=clf0.predict(Xt)
    return np.dot(Yt,np.sign(yh))/len(yh)

def score_sign(clf,X0,Y0,Xt,Yt,wt=None) :
    clf0=copy.deepcopy(clf)
    clf0.fit(X0,Y0,sample_weight=wt)
    yh=clf0.predict(Xt)
    return np.dot(Yt,np.sign(yh))/len(yh)

def score_cor(clf,X0,Y0,Xt,Yt,wt=None) :
    clf0=copy.deepcopy(clf)
    clf0.fit(X0,Y0,sample_weight=wt)
    yh=clf0.predict(Xt)
    return np.dot(Yt,(yh/yh.std()))/len(yh)

def score_sign_insample(clf,X0,Y0,wt_train=0.5,lb=16,k=3,step=2,wt_decay=0.1):
    clf0=copy.deepcopy(clf)
    n,m=X0.shape
    wt=l1_reader.getwt(n,wt_train)
    wt/=np.sum(wt)
    clf0.fit(X0,Y0,wt)
    yh=clf0.predict(X0[-lb-k*step:,:])
    x=Y0[-lb-k*step:]*np.sign(yh)
    return eval_latest_avg(x,lb=lb,k=k,step=step,wt_decay=wt_decay),clf0

def eval_latest_avg(x,lb=16,k=3,step=2,wt_decay=0.1) :
    n=len(x)
    avg=[]
    for ix1,ix2 in zip(n-np.arange(k)*step-lb, n-np.arange(k)*step) :
        #print ix1, ix2
        avg.append(np.mean(x[ix1:ix2]))
    if len(avg) > 1 :
        wt=l1_reader.getwt(k,wt_decay)[::-1]
        wt/=np.sum(wt)
    else :
        wt = np.array([1])
    #print avg, wt
    return np.dot(avg,wt)

def eval_feat_corr(x,y,lb=16,k=3,step=2,wt_decay=0.1,whiten=True) :
    assert len(y) >= lb+k*step
    x0=x[-lb-k*step:].copy()
    y0=y[-lb-k*step:].copy()
    if whiten :
        x0=(x0-x0.mean())/x0.std()
        y0=(y0-y0.mean())/y0.std()
    return eval_latest_avg(x0*y0,lb=lb,k=k,step=step,wt_decay=wt_decay)


def feat_sel3(X,y,st_smp_cnt=16,lt_smp_cnt=200,wt_decay=0.25,wt_train=0.5,clf=None,cor100=0.05,vbose=False) :
    n,m=X.shape
    assert n > lt_smp_cnt and lt_smp_cnt >= st_smp_cnt, 'len error'
    assert len(y) == n, 'X,y shape error'
    mt_smp_cnt=(lt_smp_cnt+st_smp_cnt)/2
    mc,ms=wt_corr2d(X,y,wt_decay=wt_decay,whiten=True)
    mclt, mslt=wt_corr2d(X[-lt_smp_cnt:,:],y[-lt_smp_cnt:],wt_decay=0,whiten=True)
    mcmt, msmt=wt_corr2d(X[-mt_smp_cnt:,:],y[-mt_smp_cnt:],wt_decay=0,whiten=True)
    mcst, msst=wt_corr2d(X[-st_smp_cnt:,:],y[-st_smp_cnt:],wt_decay=0,whiten=True)
    ixst=np.nonzero(mc*mcst>0)[0]
    ixmt=np.nonzero(mc*mcmt>0)[0]
    ixlt=np.nonzero(mc*mclt>0)[0]
    #pdb.set_trace()
    ixa=np.intersect1d(np.intersect1d(ixst,ixmt),ixlt) #cnadidate features
    if cor100 > 0 :
        ixcor=np.nonzero(np.abs(mc)*np.sqrt(n)/10>cor100)[0]
        ixa=np.intersect1d(ixa,ixcor)
    if len(ixa) == 0:
        #nothing to be chosen
        return [7]
    if vbose :
        print 'ixst: ', len(ixst), ixst
        print 'ixmt: ', len(ixmt), ixmt
        print 'ixlt: ', len(ixlt), ixlt
        print 'ixcor:', len(ixcor), ixcor
        print 'got ixa:',   len(ixa), ixa

    if wt_train is not None :
        wtlt=l1_reader.getwt(n-lt_smp_cnt,wt_train)
        wtmt=l1_reader.getwt(n-mt_smp_cnt,wt_train)
        wtst=l1_reader.getwt(n-st_smp_cnt,wt_train)
        wtlt/=np.sum(wtlt)
        wtmt/=np.sum(wtmt)
        wtst/=np.sum(wtst)
    else :
        wtlt = None
        wtmt = None
        wtst = None
    if clf is None:
        #clf = linear_model.RidgeCV(alphas=[0.1,0.5,1,2,5,10,20,40,60,100])
        clf = linear_model.RidgeCV()
    tol=1e-10
    ixi=np.array([0])
    maxit=100
    it=0
    scb=0
    scbp=[]
    while len(ixa) > 0 and it<maxit:
        if vbose :
            print 'START ======= iter ', it, 'fs ', ixi, 'score ', scb, ' possible ', ixa
        sc=[]
        for i0 in ixa :
            # try to find a best feature from idx
            ixt=np.r_[ixi, i0]
            X0=X[:,ixt]
            sc0=0
            if vbose :
                print i0, 
            for ct,wt in zip([lt_smp_cnt, mt_smp_cnt, st_smp_cnt],[wtlt, wtmt, wtst]) :
                sc0+=score_sign(clf,X0[:-ct,:],y[:-ct],X0[-ct:,:],y[-ct:],wt=wt)
                if vbose :
                    print ct, sc0, 
            sc.append(sc0)
            if vbose :
                print
        # pick a best one
        ixch0=np.argsort(sc)[-1]
        ixch=ixa[ixch0]
        if sc[ixch0] - scb > tol :
            if vbose :
                print 'adding ', ixch, ' score imp ', sc[ixch0]-scb
            scb=sc[ixch0]

            # but would removing existing features in ixi improve?
            while len(ixi) > 1:
                sc=[]
                for i0 in np.arange(len(ixi)-1)+1 :
                    # try to remove i0
                    if vbose :
                        print 'try removing '
                    sc0=0
                    X0=X[:,np.r_[np.delete(ixi,i0),ixch]]
                    for ct,wt in zip([lt_smp_cnt, mt_smp_cnt, st_smp_cnt],[wtlt,wtmt,wtst]) :
                        sc0+=score_sign(clf,X0[:-ct,:],y[:-ct],X0[-ct:,:],y[-ct:],wt=wt)
                    if vbose :
                        print ixi[i0], sc0
                    sc.append(sc0)

                ixrm0=np.argsort(sc)[-1]
                if sc[ixrm0] < scb :
                    break
                ixrm=ixi[ixrm0+1]
                if vbose :
                    print 'removing ',ixrm, ' score imp ', sc[ixrm0]-scb
                ixi=np.delete(ixi, ixrm0+1)
                scb=sc[ixrm0]

            ixi=np.r_[ixi,ixch]
            ixa=np.delete(ixa,ixch0)
        else :
            ixa=[]
        if vbose :
            print 'DONE ======= iter ', it, 'fs ', ixi, 'score ', scb
        it+=1
    
    if len(ixi)== 1:
        return [7]
    return ixi[1:]

# ultimately, it seems that mcs*mcsst is the only good indicator found.
# forward stage selection suffers from the cost in sample not same as out-of-smaple.
# And it selects too few features.
# a compromise would be to use feat_sel2 to select more features
# and use backwards stage to remove features as feat_sel3
# take lb=20, for 5 points, wt_avg them with wt=0.1
# score_avg()
# then for all the features that could be removed, 
# select a feature with the lowest weighted corr with wt = 0.1
# (or you could remove all features that didn't help)
# iterate 
def feat_sel4(X,y,lb=20,st_smp_cnt=16,st_th=8e-3,lt_th=4e-2,wt_train=0.5,clf=None,vbose=False,xlr_cnt=9,ix_check=[10,12,13,14]) :
    n,m=X.shape
    assert n > st_smp_cnt, 'len error'
    assert len(y) == n, 'X,y shape error'
    # this is going to select more features than feat_sel2
    ixi=feat_sel2(X,y,st_smp_cnt=st_smp_cnt,st_th=st_th,lt_th=lt_th,vbose=False)
    if vbose :
        print 'got ixi:',   len(ixi), ixi
    if clf is None:
        clf = linear_model.RidgeCV(alphas=[0.1,0.5,1,2,5,10,20,40,60,100])
    #rank features according to their corr
    cor=[]
    for i0 in ixi :
        cor.append(np.abs(eval_feat_corr(X[:,i0],y,lb=lb)))
    ixc=np.argsort(cor)
    ixi=ixi[ixc]

    ix0=np.nonzero(ixi>xlr_cnt)[0]
    if len(ix0) > 0 :
        ixi=np.r_[ixi[ix0],np.delete(ixi,ix0)]

    # a very special rule 
    if 10 in ixi and 8 in ixi and 11 in ixi :
        # remove 10
        ix10=np.nonzero(ixi==10)[0]
        ixi=np.delete(ixi,ix10)
        if vbose :
            print ' removed 10'
    if vbose :
        print 'got corr ', cor, 'rearrange ixi ', ixi

    scb,clf0=score_sign_insample(clf,X[:,ixi],y,wt_train=wt_train,lb=lb)
    scbp=[]

    while len(ixi) > 1 :
        if vbose :
            print 'START ======= iter ', ixi, 'score ', scb
        for i0 in np.arange(len(ixi)) :
            if ixi[i0] not in ix_check :
                continue

            # try to remove i0
            if vbose :
                print ixi[i0], ' : ',

            ixi0=np.r_[0,np.delete(ixi,i0)]
            sc,clf0=score_sign_insample(clf,X[:,ixi0],y,wt_train=wt_train,lb=st_smp_cnt)
            if vbose :
                print sc-scb, 
            if sc > scb :
                scb=sc
                ixi=np.delete(ixi,i0)
                if vbose :
                    print ' removed '
                break
            else :
                if vbose :
                    print
        else :
            break;

    if vbose :
        print 'DONE ======= ', ixi, 'score ', scb
    return ixi

### latest signal (I am going to stop here)
### This uses the feat_sel1() with default settings
###    feat_sel1(X,y,lt_cor_th=0.0001,st_smp_cnt=16,wt_decay=0.25,st_th=1e-2,lt_th=1e-5) 
### Best Roll:
###    hist=400 week
###    roll=20  week
###    train_wt_decay=0.2
###    feat_select=feat_sel1
###    clf=RidgeCV() or SGDRegressor(loss='huber',alpha=0.5)
### i.e. 
###    X0, Y0, X1, Y1, yh0, omp=sim.rollY(wb5m,np.arange(400)+i*20, np.arange(20)+400+i*20,clf,train_wt_decay=0.2,feat_select=True)

###
### considering the recent performance trend, the model is adjusted to 
### coef2,yt2,yh2,dt2,cor32,ax_arr2=sim.eval_roll(wb5m,600,20,0.5,clf,True,train_ix0=600)
### using feat_sel2(st_th=1e-2, lt_th=5e-2)
### clf=RidgeCV()
### I should stop working on this, move on to execution
###
### Compared with logistic regression (since the cost function is yt*np.sign(yh)), 
### regression is still better, since by converting ytrain to np.sign(ytrain), 
### magnitude is lost and its meaningful. Using weight in training doesnot make that up.
### 
### So the best so far:
### clf=clf=linear_model.RidgeCV(alphas=[0.01,0.1,0.5,1,2,5,10,20,40,60,100])
### coef0,yt0,yh0,dt0,cor30,ax_arr2=sim.eval_roll(wb5m,600,40,0.5,clf,True,train_ix0=600,fit_sign=False)
### np.sum(yt0*yh0) = 0.99
### 
def eval_roll0(wb5m, hist, roll, train_wt_decay, clf, feat_select, train_ix0=400, ax_arr=None, fit_sign=False,if_plot=False) :
    import datetime
    coef=[]
    yt=[]
    yh=[]
    dt=[]
    cor3=[]
    n=wb5m.shape[0]
    k=(n-train_ix0)/roll
    max_coef=23
    for i in np.arange(k) :
        if hist==-1 :
            ix0=0
        else :
            ix0=train_ix0+i*roll-hist
        ix1=train_ix0+i*roll
        #print ix0, ix1
        X0, Y0, X1, Y1, yh0, omp, coefix=rollY(wb5m,np.arange(ix0,ix1), np.arange(ix1,ix1+roll),clf,train_wt_decay=train_wt_decay,feat_select=feat_select,fit_sign=fit_sign)
        c0=np.zeros(max_coef)
        if not fit_sign :
            c0[coefix]=omp.coef_
        coef.append(c0.copy())
        yh=np.r_[yh,yh0]
        yt=np.r_[yt,Y1]
        cor3.append(np.corrcoef(Y1,yh0)[0,1])

        for t0 in wb5m[ix1:ix1+roll,-1,0] :
            dt.append(datetime.datetime.fromtimestamp(wb5m[ix1,-1,0]))

    coef=np.array(coef)

    if if_plot :
        #plotting
        if ax_arr is None:
            fig=pl.figure()
            ax1=fig.add_subplot(3,1,1)
            ax2=fig.add_subplot(3,1,2,sharex=ax1)
            ax3=fig.add_subplot(3,1,3,sharex=ax1)
            ax1.plot(dt,np.cumsum(yt),'.-',label='yt')
        else :
            ax1,ax2,ax3=ax_arr

        pstr='h%dr%dfs%dwt%.2f'%(hist,roll,1 if feat_select else 0, 0 if train_wt_decay is None else train_wt_decay)
        ax1.plot(dt,np.cumsum(yh),'.-',label=pstr)
        ax2.plot(dt,np.cumsum(yt*yh/yh.std()),'.-',label='yt*yh,'+pstr)
        ax2.plot(dt,np.cumsum(yt*np.sign(yh)),'.-',label='yt*sg(yh),'+pstr)

        #plotting coef
        m0=['r','g','b','y','m']
        m1=['.','x','^','v','+']
        for i in np.arange(max_coef) :
            marker=m0[i%len(m0)]+m1[i/len(m0)]+'-'
            ax3.plot(dt[::roll],coef[:,i],marker,label=str(i))

        ax1.grid(); ax2.grid(); ax3.grid(); 
        ax1.legend(); ax2.legend(); ax3.legend()

    else :
        ax1=ax2=ax3=None
    return coef, yt, yh, dt, cor3, [ax1,ax2,ax3]

######
#fitted parameters:
#yt,yh,yh2,yh3=sim.eval_roll(wb5m,600,40,0.7,clf,True,train_ix0=600)
#yh=yh_40*0.75+yh_20*0.25
#sum(yt*np.sign(yh))=1.075
#grid search confirmed the result
####
def eval_roll(wb5m, hist=600, roll=40, train_wt_decay=0.5, clf=None, feat_select=True, train_ix0=600, ax_arr=None, fit_sign=False,if_plot=False) :
    coef, yt, yh, dt, cor3, ax=eval_roll0(wb5m,hist,roll,train_wt_decay,clf,feat_select,train_ix0=train_ix0,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    coef2, yt2, yh2, dt2, cor32, ax=eval_roll0(wb5m,hist,roll/2,train_wt_decay,clf,feat_select,train_ix0=train_ix0,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    return yt,yh*0.75 + yh2*0.25

def grid_search_roll(wb5m, hist=600, roll=40, train_wt_decay=0.5, clf=None, feat_select=True, train_ix0=600, ax_arr=None, fit_sign=False,if_plot=False) :
    coef, yt, yh, dt, cor3, ax=eval_roll0(wb5m,600,40,train_wt_decay,clf,feat_select,train_ix0=600,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    coef2, yt2, yh2, dt2, cor32, ax=eval_roll0(wb5m,hist,20,train_wt_decay,clf,feat_select,train_ix0=train_ix0,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    coef3, yt3, yh3, dt3, cor33, ax=eval_roll0(wb5m,hist,roll,train_wt_decay,clf,feat_select,train_ix0=train_ix0-7,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    yh3=yh3[7:]
    coef4, yt4, yh4, dt4, cor34, ax=eval_roll0(wb5m,hist,16,train_wt_decay,clf,feat_select,train_ix0=train_ix0-10,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    yh4=yh4[10:]
    coef5, yt5, yh5, dt5, cor35, ax=eval_roll0(wb5m,hist,20,train_wt_decay,clf,feat_select,train_ix0=train_ix0-3,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    yh5=yh5[3:]
    coef6, yt6, yh6, dt6, cor36, ax=eval_roll0(wb5m,hist,20,train_wt_decay,clf,feat_select,train_ix0=train_ix0-1,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    yh6=yh6[1:]
    coef7, yt7, yh7, dt7, cor37, ax=eval_roll0(wb5m,hist,23,train_wt_decay,clf,feat_select,train_ix0=train_ix0-5,ax_arr=None,fit_sign=fit_sign,if_plot=False)
    yh7=yh7[5:]
    yarr = [yh, yh2, yh3, yh4, yh5,yh6,yh7]
    n=len(yh)
    for i in np.arange(len(yarr)-1)+1 :
        if len(yarr[i]) < n :
            yarr[i] = np.r_[yarr[i], yh[-(n-len(yarr[i])):]]
        yarr[i] = yarr[i][:n]

    """
    #yh3=np.r_[yh3,yh[-d:]]
    w=np.arange(roll/2).astype(float)+1.0
    w1=np.r_[w,np.ones(roll/2)*roll/2]
    w2=np.r_[w,w]*4
    w3=(np.arange(roll).astype(float)+roll)*8

    n=wb5m.shape[0]
    k=(n-train_ix0)/roll
    w1=np.tile(w1,k)
    w2=np.tile(w2,k)
    w3=np.tile(w3,k)
    w1_= (w2+w3)/(w1+w2+w3)/2
    w2_= (w1+w3)/(w1+w2+w3)/2
    w3_= (w2+w3)/(w1+w2+w3)/2

    #return yt, yh*w1_+yh2[:n*k]*w2_+yh3[:n*k]*w3_
    #return yt, yh*0.75+yh2[:n*k]*0.2+yh3[:n*k]*0.05
    """
    return yt, yarr

def grid_search(wb5m, hist_arr, roll_arr,train_wt_decay_arr,clf,train_ix0_arr) :
    hist_arr=[200, 400, 600]
    roll_arr=[10,20,40]
    train_wt_decay_arr=[None, 0.1, 0.5,1]
    train_ix0_arr=[400, 600, 800]

def get_signal(wb5m) :
    clf=linear_model.SGDRegressor(loss='huber',alpha=0.5)
    cor5=[]
    for i in np.arange(32) :
        X0, Y0, X1, Y1, yh0, omp=sim.rollY(wb5m,np.arange(400)+i*20, np.arange(20)+400+i*20,clf,train_wt_decay=0.2,feat_select=True)
        cor5.append(np.corrcoef(yh0, Y1)[0,1])
    return np.array(cor5)


def runSimFday(wb5m_yh,yh,yh0=-60,yh1=-30,ycheck=-58,yhold=-20,yend=-1) :
    yy0=np.sum(wb5m_yh[:,y0:ycheck,1],axis=1)
    yy1=np.sum(wb5m_yh[:,y0:ycheck,1],axis=3)
    # use sign of yy0
    yhh=yh.copy()
    ysd=yhh.std()
    ix1=np.nonzero(yh*yy0>0)[0]
    yhh[ix1]+= yy0[-100:][ix1]/yy0.std()*ysd*1
    #ix2=np.nonzero(yh*yy1[-100:]>0)[0]  # vbs is not as powerful
    #yhh[ix2]+= np.sign(yy1[-100:][ix2])*ysd*.1
    yhh=np.sign(yhh)*(np.abs(yhh)**0.01)
    yhh/=np.max(np.abs(yhh))
    # since the signal is good, it's almost np.sign

    ylr=np.cumsum(wb5m[:,ycheck:yh1,1],axis=1)
    pnl=(ylr.T*yhh).T

    # stop loss doesn't work
    th=0.0125
    #th=0.02
    tick=0.0001
    for i,(p0,yhh0) in enumerate(zip(pnl,np.abs(yhh))) :
        th1=th*yhh0
        #th1=th
        #ix=np.nonzero(p0<-th1)[0]
        ix=np.nonzero(p0>th1)[0]
        if len(ix)>0 :
            eix=np.nonzero(p0[ix[0]:] <= th1*1.2)[0]
            if len(eix) > 0 :
                eix=ix[0]+eix[0]
                #th0=p0[eix]-tick*yhh0
                th0=p0[eix]
                #print 'stop loss at ',ix[0],eix,p0[ix[0]],p0[eix], 'final pnl is ',pnl[i,-1],th0,'saved ', th0-pnl[i,-1]
                pnl[i,-1]=th0

    ltx=-1
    pnl0=pnl[:,ltx].copy()
    #pnl0=pnl[:,-10].copy()

    pnl=pnl[:,-1]
    ix=np.nonzero(pnl0<0)[0]
    #for a0,a1,a2 in zip(yhh[ix],pnl[ix],pnl0[ix]) :
    #    print a0, a1, a2
    # let negative ones run longer

    ltx2=-10
    #print ltx2, ltx, np.sum(wb5m[-100:,-30+ltx+1:ltx2,1],axis=1)[ix]*yhh[ix]
    pnl0[ix]-=(np.sum(wb5m[-100:,-30+ltx+1+10:ltx2,1],axis=1)[ix]*yhh[ix])
    #pnl0[ix]+=(np.sum(wb5m[-100:,-30+ltx+1:ltx2,1],axis=1)[ix]*yhh[ix])

    pnl[ix]=pnl0[ix].copy()

    axp.plot(np.cumsum(pnl), '.-',label='pnl=%.2f,shp=%.2f'%(np.sum(pnl), pnl.mean()/pnl.std()))

    ax.legend()
    axp.legend()
    return X,y0,yh,yhh,pnl

def x_corr(X,y0) :
    c0=[]
    N,K=X.shape
    for i in np.arange(K) :
        #print i,
        cc=[]
        for ix in [np.arange(N-200), np.arange(50)+(N-200), np.arange(50)+(N-150), np.arange(50)+(N-100), np.arange(50)+(N-50)] :
            if X[ix,i].std() <1e-7 or X[ix,i].std() < np.abs(X[ix,i].mean())/100 :
                cor1=0
            else :
                cor1=np.corrcoef(X[ix,i],y0[ix])[0,1]
            cc.append(cor1)

        #print cc
        c0.append(np.array(cc).copy())

    return np.array(c0)

