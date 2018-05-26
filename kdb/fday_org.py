import numpy as np
from sklearn import linear_model
import ean
import l1_reader
import matplotlib.pylab as pl
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

def test(wb5m,if_plot=True) :
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

    # prev
    #fd18=np.r_[0, np.sum(wb5m[:-1,-60:-30,1],axis=1)] 
    #fd19=np.r_[0, fd18[:-1]]
    
    fd19=np.r_[np.zeros(2), np.sum(wb5m[:-2,-60:-30,1],axis=1)] 
    fd20=np.r_[np.zeros(18), np.sum(wb5m[:-18,-60:-30,1],axis=1)] # good negative corr at 18th week
    fd21=np.r_[np.zeros(32), np.sum(wb5m[:-32,-60:-30,1],axis=1)] # good positive corr at 32th week
    fd22=np.r_[np.zeros(35), np.sum(wb5m[:-35,-60:-30,1],axis=1)] # good positive corr at 35th week

    ## short range
    yy0=np.sum(wb5m[:,-60:-58,1],axis=1)
    yy1=np.sum(wb5m[:,-60:-58,3],axis=1)
    yy0_=np.sign(yy0)

    #X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd5,fd6,fd7,fd8,fd9,fd10,fd11,fd12,fd13,fd14,fd15)).T

    # consider adding fd13 in, but it has a flip in 1 of the previous year
    #X=np.vstack((fd0,fd1,fd3,fd11,fd14)).T  # it seems that Monday,Thursday and Friday has the best
    #X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd11,fd14,fd15,fd16).T  # it seems that Monday,Thursday and Friday has the best
    X=np.vstack((fd0,fd1,fd2,fd3,fd4,fd11,fd14,fd15,fd16)).T  # it seems that Monday,Thursday and Friday has the best


    Xvbs=get_Xvbs(wb5m[:,:,3])
    if len(Xvbs.shape)== 1:
        X=np.vstack((X.T,Xvbs)).T
    else :
        X=np.hstack((X,Xvbs))

    """
    # this vol0 doesn't seem to help
    Xv0=get_vol0(wb5m[:,:,4])
    if len(Xv0.shape)== 1:
        X=np.vstack((X.T,Xv0)).T
    else :
        X=np.hstack((X,Xv0))
    """

    #X=np.vstack((X.T,fd19,fd22)).T
    #X-=np.mean(X,axis=0) ; X/=np.std(X,axis=0); X=np.vstack((np.ones(len(fd0)),X.T)).T

    y=np.sum(wb5m[:,-60:-30,1],axis=1)
    wt=l1_reader.getwt(len(y),0.1)
    wt/=np.sum(wt)
    mu=np.dot(y,wt)
    sd=np.sqrt(np.dot(wt, (y-mu)**2))
    y0=ean.outlier(y,mu,sd,in_th=1,out_th=3)

    fd19=np.r_[np.zeros(2), y0[:-2]] 
    fd20=np.r_[np.zeros(18), y0[:-18]] # good negative corr at 18th week
    fd21=np.r_[np.zeros(32), y0[:-32]] # good positive corr at 32th week
    fd22=np.r_[np.zeros(35), y0[:-35]] # good positive corr at 35th week
    X=np.vstack((X.T,fd19,np.sign(fd22)*np.abs(fd21))).T
    X-=np.mean(X,axis=0) ; X/=np.std(X,axis=0); X=np.vstack((np.ones(len(fd0)),X.T)).T

    print mu, sd
    fig=pl.figure()
    ax=fig.add_subplot(2,1,1)
    axp=fig.add_subplot(2,1,2)

    mu=0 ; sd=1  # don't scale it
    #omp=linear_model.SGDRegressor(loss='huber',penalty='l1',alpha=1)
    #y0=np.sign(y0)

    #omp=linear_model.Lars()
    #omp=linear_model.OrthogonalMatchingPursuitCV()
    #omp=linear_model.LassoCV()
    omp=linear_model.RidgeCV(); 
    omp.fit(X[:-50,:],y0[:-50])
    yh=omp.predict(X[-50:,:]) * sd + mu
    if ax is not None:
        ax.plot(omp.coef_, '.-', label='-50')
    
    print np.corrcoef(yh, y0[-50:])[0,1]

    omp=linear_model.RidgeCV(); omp.fit(X[:-100,:],y0[:-100])
    yh=omp.predict(X[-100:-50,:]) * sd + mu
    print np.corrcoef(yh, y0[-100:-50])[0,1]
    if ax is not None:
        ax.plot(omp.coef_, '.-', label='-100:-50')

    omp=linear_model.RidgeCV(); omp.fit(X[:-200,:],y0[:-200])
    yh=omp.predict(X[-200:-100,:]) * sd + mu
    print np.corrcoef(yh, y0[-200:-100])[0,1]
    if ax is not None:
        ax.plot(omp.coef_, '.-', label='-200:-100')

    omp=linear_model.RidgeCV(); omp.fit(X[:-100,:],y0[:-100])
    yh=omp.predict(X[-100:,:]) * sd + mu
    print np.corrcoef(yh, y0[-100:])[0,1]
    if ax is not None:
        ax.plot(omp.coef_, '.-', label='-100')

    # use sign of yy0
    yhh=yh.copy()
    ysd=yhh.std()
    ix1=np.nonzero(yh*yy0[-100:]>0)[0]
    yhh[ix1]+= yy0[-100:][ix1]/yy0.std()*ysd*1
    #ix2=np.nonzero(yh*yy1[-100:]>0)[0]
    #yhh[ix2]+= np.sign(yy1[-100:][ix2])*ysd*.1
    yhh=np.sign(yhh)*(np.abs(yhh)**0.01)
    yhh/=np.max(np.abs(yhh))

    
    ylr=np.cumsum(wb5m[-100:,-58:-30,1],axis=1)
    yraw=ylr[:,-1]

    pnl=(ylr.T*yhh).T
    #pnl=yraw*yhh
    
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

def get_Xvbs(vbs) :
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

    #X=np.vstack((fd2,fd3,fd4,fd5,fd6,fd18)).T
    X=np.vstack((fd2,fd3,fd18)).T
    #X=np.vstack((fd2,fd2_)).T
    #X=fd18

    return X.copy()

def get_vol0(v0) :
    #fd0=np.sum(v0[:,-84:-68],axis=1)  # this is gone
    fd1=np.sum(v0[:,(3*23+18)*12:(4*23+1)*12],axis=1)  #Thursday 12pm to Fri 7pm

    #X=np.vstack((fd0,fd1)).T
    X=fd1
    return X

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

