import numpy as np
def make_risk_reward_data(r0, e0, rm, pm, rp, pp,xmax):

    nrm=rm.shape[0]
    nrp=rp.shape[0]
    npts=nrm+nrp+1
    br=np.zeros(npts)
    ar=np.zeros(npts)
    xr=np.zeros(npts)

    br[nrm-1]=-pm[0]*(rm[0]-r0)+e0
    ar[nrm-1]=-rm[0]
    xr[nrm-1]=-pm[0]
    for k in range(1,nrm):
        br[nrm-1-k]=br[nrm-k]-pm[k]*(rm[k]-rm[k-1])
        ar[nrm-1-k]=-rm[k]
        xr[nrm-1-k]=-pm[k]

    br[nrm]=e0
    ar[nrm]=-r0

    br[nrm+1]=pp[0]*(rp[0]-r0)+e0
    ar[nrm+1]-=rp[0]
    xr[nrm]=pp[0]
    for k in range(1,nrp):
        br[nrm+1+k]=br[nrm+k]+pp[k]*(rp[k]-rp[k-1])
        ar[nrm+1+k]=-rp[k]
        xr[nrm+k]=pp[k]

    xr[nrm+nrp]=xmax

    return xr,ar,br

def add_risk_reward(x1,y1,xr,ar,br):

    n=x1.shape[0]
    npts=xr.shape[0]
    x0=np.zeros(n+npts-1)
    y0=np.zeros(n+npts-1)

    x0[0]=x1[0]
    y0[0]=y1[0]+x1[0]*ar[0]
    ir=0
    i0=1
    i1=1

    while (x1[i1]>xr[ir]):
        x0[i0]=xr[ir]
        y0[i0]=(y1[i1-1]/x1[i1-1])*(x0[i0]-x1[i1])+y1[i1]+x0[i0]*ar[ir]+br[ir]
        i0+=1
        ir+=1
    x0[i0]=x1[i1]
    y0[i0]=y1[i1]+x1[i1]*ar[ir]+br[ir]
    i0+=1

    for i1 in np.arange(2,n-1):
        while (x1[i1]>xr[ir]):
            x0[i0]=xr[ir]
            y0[i0]=((y1[i1]-y1[i1-1])/(x1[i1]-x1[i1-1]))*(x0[i0]-x1[i1])+y1[i1]+x0[i0]*ar[ir]+br[ir]
            i0+=1
            ir+=1
        x0[i0]=x1[i1]
        y0[i0]=y1[i1]+x1[i1]*ar[ir]+br[ir]
        i0+=1

    i1+=1
    while (ir<(npts-1)):
        x0[i0]=xr[ir]
        y0[i0]=(y1[i1]/x1[i1])*(x0[i0]-x1[i1-1])+y1[i1-1]+x0[i0]*ar[ir]+br[ir]
        i0+=1
        ir+=1

    x0[i0]=x1[i1]
    y0[i0]=y1[i1]+x1[i1]*ar[npts-1]

    return x0,y0

def add_risk_reward0(x1,y1,r0,e0):

    n=x1.shape[0]
    x0=np.zeros(n)
    y0=np.zeros(n)

    x0[0]=x1[0]
    y0[0]=y1[0]-r0*x1[0]

    for i0 in np.arange(1,n-1):
        x0[i0]=x1[i0]
        y0[i0]=y1[i0]-r0*x1[i0]+e0

    x0[n-1]=x1[n-1]
    y0[n-1]=y1[n-1]-r0*x1[n-1]

    return x0,y0

def add_trade_cost0(x0,y0,t0):

    n=x0.shape[0]
    x1=np.zeros(n+2)
    x2=np.zeros(n+2)
    y1=np.zeros(n+2)

    y1[0]=0.0
    x2[0]=0.0
    x1[0]=1.0
    i0=1
    i1=1

    while (i0<n-1):
        if (y0[i0]<t0):
            break
        i0+=1

    if (i0==1) :
        y1[1]=t0
        x2[1]=x0[1]+(x0[0]/y0[0])*(y1[1]-y0[1])
        x1[1]=x2[1]
    elif (i0<(n-1)) :
        y1[1]=t0
        x2[1]=x0[i0]+((x0[i0]-x0[i0-1])/(y0[i0]-y0[i0-1]))*(y1[1]-y0[i0])
        x1[1]=x2[1]
    else:
        y1[1]=t0
        x2[1]=x0[i0-1]+(x0[i0]/y0[i0])*(y1[1]-y0[i0-1])
        x1[1]=x2[1]
    i1=2

    while (i0<n-1):
        if (y0[i0]>-t0) :
            y1[i1]=y0[i0]
            x2[i1]=x0[i0]
            x1[i1]=x0[i0]
            i1+=1
        else:
            break
        i0+=1

    if (i0==1):
        y1[i1]=-t0
        x2[i1]=x0[1]+(x0[0]/y0[0])*(y1[i1]-y0[1])
        x1[i1]=x2[i1]
    elif (i0<(n-1)):
        y1[i1]=-t0
        x2[i1]=x0[i0]+((x0[i0]-x0[i0-1])/(y0[i0]-y0[i0-1]))*(y1[i1]-y0[i0])
        x1[i1]=x2[i1]
    else:
        y1[i1]=-t0
        x2[i1]=x0[i0-1]+(x0[i0]/y0[i0])*(y1[i1]-y0[i0-1])
        x1[i1]=x2[i1]
    i1+=1

    y1[i1]=0.0
    x2[i1]=0.0
    x1[i1]=1.0

    y1=y1[:i1+1]
    x2=x2[:i1+1]
    x1=x1[:i1+1]

    return x1,x2,y1

def p2_from_p1(p1,x1,x2):

    n=x1.shape[0]

    i1=1
    while(i1<n-1):
        if (x1[i1]>p1):
            break
        i1+=1

    if (i1==1):
        p2=x2[1]+(x2[0]/x1[0])*(p1-x1[1])
    elif (i1<(n-1)):
        p2=x2[i1]+((x2[i1]-x2[i1-1])/(x1[i1]-x1[i1-1]))*(p1-x1[i1])
    else:
        p2=x2[i1-1]+(x2[i1]/x1[i1])*(p1-x1[i1-1])

    return p2

def opt(p0,e0,r0,t0,w):

    n=e0.shape[0]

    p=np.zeros(n)

    npts=2*(n+2)

    x0=np.zeros(npts)
    y0=np.zeros(npts)

    X1=np.zeros((n,npts))
    X2=np.zeros((n,npts))
    x1=np.zeros(npts)
    x2=np.zeros(npts)
    y1=np.zeros(npts)
    n1=np.zeros(n).astype(int)

    x1=np.array([1.0,0.0,0.0,1.0])
    y1=np.array([0.0,t0[n]*w[n-1],-t0[n]*w[n-1],0.0])

    for k in np.arange(n-1,-1,-1):

        [x0,y0]=add_risk_reward0(x1,y1,r0[k]*w[k],e0[k]*w[k])
        [x1,x2,y1]=add_trade_cost0(x0,y0,t0[k]*w[k])

        n1[k]=x1.shape[0]
        X1[k,:n1[k]]=x1
        X2[k,:n1[k]]=x2

    p1=p0
    for k in np.arange(n):
        p[k]=p2_from_p1(p1,X1[k,:n1[k]],X2[k,:n1[k]])
        p1=p[k]

    return p
