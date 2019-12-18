import numpy as np
import datetime
import matplotlib.pylab as pl
import scipy.stats
import pdb

def f0(p) :
    a=p[-1]
    if a=='k' or a=='K':
        return float(p[:-1])*1000
    if a=='m' or a=='M':
        return float(p[:-1])*1000000
    if a>='0' and a<='9':
        return float(p)
    raise ValueError('wrong val at f0')

def f1(p) :
    d=np.genfromtxt(p, delimiter=',', dtype=[('i8'), ('i4'), ('|S12'), ('|S12')], usecols=(0,1,6,7))
    v=[]
    dt=[]
    for d0 in d :
        if len(d0[2])==0 or len(d0[3])==0:
            continue
        dt0=datetime.datetime.strptime(str(d0[0])+str(d0[1]),'%Y%m%d%H%M')
        v.append([float(dt0.strftime('%s')),f0(d0[2]),f0(d0[3])])
        dt.append(dt0)
    v=np.array(v)
    a=np.cumsum(v[:,2]) # actual stock, as cumsum of actual change
    f=np.r_[v[0,1],a[:-1]+v[1:,1]] # forecasted stock, as previous actual stock plus this forecast
    return np.array(dt), a,f,np.array(v)
#pl.figure();pl.plot(dt,a,label='actual'),pl.plot(dt,f,'.',label='forecast');pl.grid();pl.legend();pl.show()

MDict={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
def api(p) :
    d=np.genfromtxt(p, delimiter=',', dtype=[('|S10'), ('i8'), ('i8'), ('|S10'), ('|S12'), ('|S12')], usecols=(0,1,2,3,4,5))
    v=[]
    dt=[]
    for d0 in d :
        m=d0[0]
        if m not in MDict.keys() :
            raise ValueError('month not found ' + m)
        yyyymmdd='%04d%02d%02d'%(d0[2], MDict[m], d0[1])
        hhmm=d0[3][:2]+d0[3][-2:]
        dt0=datetime.datetime.strptime(yyyymmdd+hhmm,'%Y%m%d%H%M')
        v.append([float(dt0.strftime('%s')),f0(d0[4]),f0(d0[5])]) # utc, act, forecast
        dt.append(dt0)

    v=np.array(v)
    ix = np.argsort(v[:, 0])
    v0=np.vstack( (v[ix, 0], v[ix, 2], v[ix, 1]) ).T # utc, forecast, act, to be consistent with f1
    dt=np.array(dt)[ix]
    a=np.cumsum(v0[:, 2])
    f=np.r_[v0[0,1],a[:-1]+v0[1:,1]]
    return np.array(dt), a,f,np.array(v0)

def f2(p) :
    c=[]
    for k in np.arange(100) + 1 :
        c.append(np.corrcoef(p[:-k], p[k:])[0,1])
    return c

def f3(p0,dn,ifp,ax=None, ax2=None) :
    """
    removing outliers in p0
    """
    ps=np.std(p0)
    ix=np.nonzero(np.abs(p0)>6*ps)[0]
    p=np.delete(p0,ix)
    #p-=p.mean()
    #p/=p.std()
    d=getattr(scipy.stats,dn)
    param = d.fit(p)
    b=len(p)/20

    # need inverse cdf! 
    p.sort()
    c=d.cdf(p,*param[-2],loc=param[-2],scale=param[-1])
    n=len(p)
    tk=np.linspace(c[0],c[-1],int((c[-1]-c[0])/(float(20)/float(n))))
    ix=np.searchsorted(c,tk)





    cnt,bv=np.histogram(p,bins=b)
    c1=d.cdf(bv[1:], *param[:-2], loc=param[-2], scale=param[-1])
    c2=d.cdf(bv[:-1], *param[:-2], loc=param[-2], scale=param[-1])
    x=(bv[1:]+bv[:-1])/2.0
    pc=cnt.astype(float)/float(len(p))
    pp=c1-c2
    E=pp*len(p)
    cs=np.sum((cnt.astype(float)-E)**2/E)

    chi2=scipy.stats.chi2.cdf(cs,df=len(p)-1)
    if ifp:
        if ax is None :
            fig=pl.figure()
            ax=fig.add_subplot(2,1,1)
            ax2=fig.add_subplot(2,1,2)
        ax.plot(x,pc,'x')
        pstr='('
        for p0 in param :
            pstr+='%.2f '%(p0)
        pstr+='),\nchi2(%.2f,df=%d)'%(cs,b-1)
        ax.plot(x,pp,'.-',label=dn+'\n:'+pstr)
        ax.grid() ; ax.legend()
        ax2.plot(E,cnt,'x')
        # fit a line for cnt and E
        Xk=np.vstack((np.ones(len(E)),E)).astype(float).T
        beta=np.dot(np.linalg.inv(np.dot(Xk.T,Xk)),np.dot(Xk.T,cnt))
        yh=np.dot(Xk,beta)
        ax2.plot(E,yh,'-.',label='icpt:%.2f,slp:%.2f]'%(beta[0],beta[1]))
        ax2.legend()
        ax2.grid()
    return x,pp,pc,chi2,cs

def f4_00(xk0,x,v,i) :
    ki=np.searchsorted(x,v)
    if ki > 0 :
        xk0[:ki,i]=0
    xk0[ki:,i]=x[ki:]
    return xk0

def f4_01_(x,ik) :
    # first column is always x, ik[0] is ignored
    xk=np.empty( (len(x), len(ik)) )
    xk[:,0]=x
    for i, v in enumerate(ik[1:]):
        xk=f4_00(xk,x,v,i+1)
    return xk

def f4_01(x,ik) :
    # first column is always x, ik[0] is ignored
    xk=np.empty( (len(x), len(ik)) )
    xk[:,0]=x
    for i, v in enumerate(ik[1:]):
        xk[:v,i+1]=0
        xk[v:,i+1]=x[v:]-x[v-1]
        #xk=f4_00(xk,x,v,i+1)
    return xk

def f4_0_beta(Xk,y) :
    # does this trick matter?
    #pdb.set_trace()
    return np.dot(np.linalg.inv(np.dot(Xk.T,Xk)),np.dot(Xk.T,y)) 

def f4_0_yh(Xk,y) :
    return np.dot(Xk,f4_0_beta(Xk,y))

def f4_0(Xk,y) :
    #MSE
    e=y-f4_0_yh(Xk,y)
    return np.dot(e.T,e)/len(y)

def f4_1(x,y,ik,tol,st=1) :
    e=1e+16
    n=len(x)
    K=len(ik)
    ik0=ik.copy()
    if len(ik0) < 2 :
        raise ValueError('len ik >= 2')
    while e>tol:
        #search ik for each components
        ikarr=[]
        sarr=[]
        Xk=f4_01(x,ik0)
        s = f4_0(Xk,y)
        if K > 2:
	    for i, k0 in enumerate(ik0[1:-1]) : #first ik always 0
                xk0=Xk.copy()
	        i0=ik0.copy()
	        if k0-st > i0[i] :
		    i0[i+1]=k0-st
		    ikarr.append(i0)
                    sarr.append(f4_0(f4_00(xk0,x,k0-st,i+1),y)-s)
	        i0=ik0.copy()
	        if k0+st < i0[i+2] :
		    i0[i+1]=k0+st
		    ikarr.append(i0)
                    sarr.append(f4_0(f4_00(xk0,x,k0+st,i+1),y)-s)
        # last one
        xk0=Xk.copy()
        i0=ik0.copy()
        ks=ik0[-1]
        if ks-st > ik0[-2] :
            i0[-1]=ks-st
            ikarr.append(i0)
            sarr.append(f4_0(f4_00(xk0,x,k0-st,K-1),y)-s)
        i0=ik0.copy()
        if ks+st < n-1 :
            i0[-1]=ks+st
            ikarr.append(i0)
            sarr.append(f4_0(f4_00(xk0,x,k0+st,K-1),y)-s)

        ix=np.argsort(sarr)[-1]
        e=sarr[ix]
        ik0=ikarr[ix]
        print e, ik0
    return ik0, f4_0(f4_01(x,ik0),y)
    
def f4_2(x, ik0, k,step_sz=1) : 
    min_diff=6*step_sz
    rg=x[-1]-x[0]
    ik=[]
    cnt=int(len(x)/(len(ik0)+k)/min_diff)
    while cnt > 0 :
        i0=np.sort(np.r_[ik0, np.random.rand(k)*rg+x[0]])
        if np.min( i0[1:]-i0[:-1] ) >=2*step_sz :
            ik.append(i0)
        cnt-=1
    return ik

def f4(x,y,k,tol=1e-12,min_diff=6,stepsz=1) :
    #x,y needs to be normalized, x sorted
    nk=0
    stepsz=1
    sc=[]
    ik=np.array([x[0]])
    while nk < k:
        scarr=[] # for this iteration
        ikarr=[]
        iks=f4_2(x,ik,1,stepsz)
        # find trial points iks via randomly adding one on top of existing ik
        for ik0 in iks :
            # take ik0 as the initial knots, find the best fit
            ik_, s_= f4_1(x,y,ik0,yy,tol,stepsz)
            ik0.append(ik_)
            sc0.append(s_)
        ix=np.argsort(sc0)[-1]
        ik.append(ik0[ix])
        sc.append(sc0[ix])
    return ik, sc

class Smooth :
    def __init__(self,x,ifcopy=True) :
        self.x=x
        if ifcopy :
            self.x=x.copy()

    def dither(self, cnt) :
        while cnt > 0 :
            x=self.x.copy()
            self.x[1:-1]+=x[:-2]
            self.x[1:-1]+=x[2:]
            self.x[1:-1]/=3.0
            self.x[0]=(self.x[0]+self.x[1])/2.0
            self.x[-1]=(self.x[-2]+self.x[-1])/2.0
            cnt -= 1
    
    def tanh(self, alpha):
        pass

    def ema(self, alpha):
        pass

class PiLi :
    def __init__(self,x,y,dither_cnt=0) :
        ix=np.argsort(x)
        self.x=x[ix]/np.std(x)
        self.x-=(self.x[0]-1)
        self.y=y[ix]
        self.Ca=self.x*self.y
        self.Xa=self.x*self.x
        #self.Cs=np.cumsum(Ca[::-1])[::-1]
        #self.Rs=np.cumsum(Xa[::-1])[::-1]
        self.n=len(self.x)
        self.S=[np.array([0,self.n])]
        self.SK=[self.score(self.S[0])]

    def r__s(self,cs,rs) :
        #self._s=((cs[0]**2)-2*cs*cs[0]+(cs**2)*rs[0]/rs)/(rs[0]-rs)[1:]
        cs1=cs[1:]
        rs1=rs[1:]
        self._s=((cs[0]**2)-2*cs1*cs[0]+(cs1**2)*rs[0]/rs1)/(rs[0]-rs1)
        return np.argsort(self._s)[-1]+1

    def r__si(self,si,ei) :
        cs=np.cumsum(self.Ca[si:ei][::-1])[::-1]
        rs=np.cumsum(self.Xa[si:ei][::-1])[::-1]
        #cs=self.Cs[si:ei]-self.Cs[si]
        #rs=self.Rs[si:ei]-self.Rs[si]
        #pdb.set_trace()
        return self.r__s(cs,rs)+si

    def __iter(self,ik,tol=1e-12,max_iter=100) :
        bscore=1e+16
        bik=[]
        ik0=ik.copy()

        #print 'iter ', ik
        while max_iter>0:
            si=2
            while si<len(ik0) :
                #pdb.set_trace()
                ik0[si-1]=self.r__si(ik0[si-2],ik0[si])
                si+=1
            sk=self.score(ik0)

            #print 'got ', ik0, sk
            if bscore-sk<=tol :
                return sk,ik0
            bscore=sk
            max_iter-=1
        print 'iteration failed to converge '
        return bscore, ik0

    def r__beta(self,ik) :
        Xk=f4_01(self.x,ik)
        ## adding an intercept here
        Xk = np.vstack( (np.ones(self.n), Xk.T) ).T
        return f4_0_beta(Xk,self.y)

    def r__yh(self,ik) :
        Xk=f4_01(self.x,ik)
        ## adding an intercept here
        Xk = np.vstack( (np.ones(self.n), Xk.T) ).T
        return f4_0_yh(Xk,self.y)

    def __nll(self,e) :
        #negative log likelihood
        d=getattr(scipy.stats,'norm')
        param = d.fit(e) #[u,std]
        return d.nnlf(param,e), param[0], param[1]

    def __res_fit(self,e,ax=None) :
        d=getattr(scipy.stats,'norm')
        param = d.fit(e) #[u,std]
        b=max(len(e)/20, 10)
        cnt,bv=np.histogram(e,bins=b)
        c1=d.cdf(bv[1:], *param[:-2], loc=param[-2], scale=param[-1])
        c2=d.cdf(bv[:-1], *param[:-2], loc=param[-2], scale=param[-1])
        x=(bv[1:]+bv[:-1])/2.0
        pc=cnt.astype(float)/float(len(e))
        pp=c1-c2
        E=pp*len(e)
        if ax is not None :
            ax.plot(E,cnt,'x',label='residual[u(%.2f) std(%.2f)]'%(param[0],param[1]))
            Xk=np.vstack((np.ones(len(E)),E)).astype(float).T
            beta=np.dot(np.linalg.inv(np.dot(Xk.T,Xk)),np.dot(Xk.T,cnt))
            yh=np.dot(Xk,beta)
            ax.plot(E,yh,'-.',label='normal fit[icpt(%.2f),slp(%.2f)]'%(beta[0],beta[1]))
            ax.set_xlabel('expected count');ax.set_ylabel('residual count')
            ax.grid();ax.legend()
        return param[0],param[1] #u,std

    def __plot_yy(self,y,yh,ax) :
        ax.plot(yh,y,'.',label='pred-actual corr(%.2f)'%(np.corrcoef(y,yh)[0,1]))
        ax.set_xlabel('y prediction');ax.set_ylabel('actual y')
        Xk=np.vstack((np.ones(len(yh)),yh)).astype(float).T
        beta=np.dot(np.linalg.inv(np.dot(Xk.T,Xk)),np.dot(Xk.T,y))
        ax.plot(yh ,np.dot(Xk,beta),'-.',label='icpt:%.2f,slp:%.2f]'%(beta[0],beta[1]))
        ax.grid();ax.legend()

    def __1more(self,tol=1e-12,min_step=1,max_iter=100) :
        skarr=[]
        ikarr=[]
        # do it multiple times
        ik0=self.S[-1].copy()
        k0=len(ik0)
        
        """
        Rnd = 0
        MaxRnd=max(min(k0*(k0-1)/4,100),3)
        while Rnd < MaxRnd :
            ik1=np.r_[0,np.sort(np.random.permutation(self.n-2)[:k0-1]+1),self.n]
            sk0,ik1=self.__iter(ik1,tol=tol,max_iter=max_iter)
            skarr.append(sk0)
            ikarr.append(ik1.copy())
            Rnd+=1
        ix=np.argsort(skarr)[0]
        print '[Random] got best score: ',  skarr[ix], ', points: ', ikarr[ix]
        """

        """
        # remove k and add k+1 randomly
        for k in np.arange(k0-2) + 1 :
            m = 1
            while m < k0-k :
                ik1=np.sort(np.r_[np.delete(ik0,np.arange(k)+m), (np.random.rand(k+1)*(self.n-2*min_step)+0.5).astype(int)+min_step])
                if np.max(ik1[1:]-ik1[:-1])<min_step*2:
                    continue
                sk0,ik1=self.__iter(ik1,tol=tol,max_iter=max_iter)
                skarr.append(sk0)
                ikarr.append(ik1.copy())
                m+=1
        ix=np.argsort(skarr)[0]
        print '[remove k] got best score: ',  skarr[ix], ', points: ', ikarr[ix]
        """
        # adding one in-between
        for i in np.arange(k0-1) :
            d=ik0[i+1]-ik0[i]-min_step*2
            if d>0 :
                i0 = int(np.random.rand(1)[0]*d+0.5)+ik0[i]+min_step
                ik1=np.sort(np.r_[i0,ik0])
                sk0,ik1=self.__iter(ik1,tol=tol,max_iter=max_iter)
                skarr.append(sk0)
                ikarr.append(ik1.copy())
        ix=np.argsort(skarr)[0]
        print '[add one] got best score: ',  skarr[ix], ', points: ', ikarr[ix]

        self.S.append(ikarr[ix])
        self.SK.append(skarr[ix])

    def score(self, ik) :
        # this assume ik includes the last ending index
        # as in the case of self.S[0]
        #pdb.set_trace()

        Xk = np.vstack( (np.ones(self.n),f4_01(self.x, ik[:-1]).T)).T
        return f4_0(Xk, self.y)

    def bestk(self,k):
        k0=len(self.S)
        while k >= k0 :
            print 'best k having ', k0-1, ', getting 1 more '
            self.__1more()
            print 'COMPLETE! GOT: ', self.S[k0], ' score: ', self.SK[k0]
            k0+=1
        return self.S[k]

    def plot(self,k) :
        ik=self.S[k]
        beta=self.r__beta(ik[:-1])
        yh=self.r__yh(ik[:-1])
        fig=pl.figure()
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(self.x,self.y,'x',label='y')
        ax1.plot(self.x,yh,'.-',label='fit')
        ax2=fig.add_subplot(2,1,2,sharex=ax1)
        ax2.plot(self.x,self.y-yh,label='residual')
        ax1.grid();ax2.grid();ax1.legend();ax2.legend()
        fig2=pl.figure()
        ax3=fig2.add_subplot(1,1,1)
        self.__res_fit(self.y-yh,ax3)
        fig3=pl.figure()
        ax4=fig3.add_subplot(1,1,1)
        self.__plot_yy(self.y,yh,ax4)

def plot_klist(self,klist) :
    fig=pl.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(self.x,self.y,'x',label='y')
    for k in klist :
        ik=self.S[k]
        beta=self.r__beta(ik[:-1])
        yh=self.r__yh(ik[:-1])
        ax.plot(self.x,yh,'.-',label='k='+str(k))
    ax.grid();ax.legend()


