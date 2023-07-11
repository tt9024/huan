import scipy.linalg as lg
import numpy as np
import copy
import dill

"""
The indicators
"""

def getVolBucket(vol, barsec = 300, bucket_hour = 1.0) :
    """
    Li = getVolBucket(vol, ndays, n, barsec, bucket_hour)
    Getting 'L', the look back bars for each daily bar i so that
    \sum vol_k from i-L to i is approximately Vb, where Vb is
    average volume of the given bucket time. 
    vol: shape (ndays,n), where n is number of bars per day
    barsec: bar second
    bucket_hour: float in hours, within which the average volume
                 is used to calculate the Li
    return
    Li:  length n vector, with number of bars to look back, 
         so that vol[i-Li+1:i+1] is proximately Vk
         Note Li>=1
    """
    ndays, n = vol.shape
    vm = np.mean(vol,axis=0)
    Vk = np.sum(vm)/(n*barsec) * (bucket_hour*3600.0)
    cs = np.cumsum(np.tile(vm,2)[::-1])
    tk = np.r_[0,cs[:n-1]]+Vk
    ix = np.searchsorted(cs,tk)
    Li = (ix-np.arange(n))[::-1]
    return np.clip(Li,1,1e+10).astype(int)

class VBS :
    def __init__(self, ndays, n, lr, vol, vbs, barsec=300) :
        """
        ndays: number of days
        n:     bars per day
        lr:    vector of ndays*n log return 
        vol:   vector of ndays*n volume
        vbs:   vector of ndays*n buy-sell volume
        """
        self.ndays = ndays
        self.n = n
        self.barsec = barsec
        self.lr = lr.reshape((ndays,n))
        self.vol = vol.reshape((ndays,n))
        self.vbs = vbs.reshape((ndays,n))

    def ind2(self, decay_Li = 0.8, rollwnd = 50, ewa = None, bucket_hour = 1.0, vbs = None, save_state=False, prev_vbs=None, prev_vol=None) :
        """
        run indicator2 with roll
        """
        n = self.n
        ndays=self.ndays
        if vbs is None :
            vbs = self.vbs
        assert vbs.shape[0]==ndays and vbs.shape[1]==n

        if prev_vbs is None:
            assert rollwnd < ndays
            prev_vbs = vbs[:rollwnd,:].copy()
        else:
            assert prev_vbs.shape[0]>=2*rollwnd
            assert prev_vol is not None
            prev_vbs = prev_vbs[-2*rollwnd:,:].copy()
        vbs = np.vstack((prev_vbs,vbs))

        vol = self.vol
        if prev_vol is None:
            assert rollwnd < ndays
            prev_vol = vol[:rollwnd,:].copy()
        else:
            assert prev_vol.shape[0]>=2*rollwnd
            assert prev_vbs is not None
            prev_vol = prev_vol[-2*rollwnd:,:].copy()
        vol = np.vstack((prev_vol,vol))

        # Li estimated in 2*rollwnd
        Li = getVolBucket(self.vol[:2*rollwnd,:], barsec=self.barsec, bucket_hour=bucket_hour)
        ix0 = rollwnd
        ret = []
        while ix0 < vbs.shape[0]:
            # note ix0+rollwnd may be larger than ndays, 
            # but vbs/vol has been prepended with rollwnd
            x = self._ind2_run(vbs[ix0-rollwnd:ix0+rollwnd,:].flatten(),Li,decay_Li=decay_Li,rollwnd=rollwnd,ewa=ewa, save_state = False)
            ret.append(x)
            Li = getVolBucket(vol[ix0-rollwnd:ix0+rollwnd,:], barsec=self.barsec, bucket_hour=bucket_hour)
            ix0+=rollwnd

        # save the state for the last Li on the last rollwnd's state
        if save_state:
            self._ind2_run(vbs[-rollwnd:,:].flatten(),Li,decay_Li=decay_Li,rollwnd=rollwnd,ewa=ewa, save_state = True)
            # long term mean/std for smoothing purpose
            self.xm = np.mean(vbs[-10*rollwnd:,:], axis=0)
            self.xs = np.std(vbs[-10*rollwnd:,:], axis=0)

        return np.vstack(ret)[-ndays:,:]

    def _ind2_run(self, vbs, Li, decay_Li = 0.8, rollwnd = 50, ewa = None, save_state = False) :
        """
        buy/sell break out w.r.t. normal range for the bar time
        vbs: flattened vector of (ndays + rollwnd) * n, where
             the initial rollwnd days are used for initialization
        Li:  length n vector so that vol[i-Li[i]+1:i+1] is approximately Vk
             see getVolBucket. 
        decay_Li: (0, 1), specifies how previous sample decays. 
                  vbs at (t-k) will be sampled as vbs[t-k] * (decay**k)
                  so the value at t would be 
                      \sum_{k=0}^Li vbs[t-k]*decay**k
        rollwnd:  the rolling window from which mean and std are drawn from.
        if ewa is not None, then the result is exponentially moving avg'ed:
        S(t+1) = (1-ewa) * S(t) + ewa * X(t+1), where X(t+1) is the sample at t+1

        return: 
        indicator of shape [ndays, n], for the latest ndays indicators
        """
        n = self.n
        Nd = len(vbs)//n
        ndays=Nd - rollwnd
        assert Nd*n==len(vbs) and ndays>=0, "vbs shape mismatch"

        # calculate the summation of decayed vbs 
        K = np.max(Li)+1
        W = np.tile(decay_Li**np.arange(K,-1,-1),(n*Nd,1))
        X = lg.hankel(np.r_[np.zeros(K),vbs[:-K]], vbs[-K-1:])
        XWc = np.cumsum((X*W),axis=1)
        #Note this could include one more bar in, which seems to make a big difference
        #in case of 4 hour bucket.
        #XLi = (XWc[:,-1]-XWc[np.arange(n*Nd),np.tile(-(Li+1)-1,Nd)]).reshape((Nd,n))
        XLi = (XWc[:,-1]-XWc[np.arange(n*Nd),np.tile(-Li-1,Nd)]).reshape((Nd,n))

        # calculate the mean and std from a rolling window
        x0c = np.cumsum(XLi,axis=0)
        x0m = np.dot(np.diag(1.0/(np.arange(rollwnd)+1)), x0c[:rollwnd,:]) # this is initial day's average
        if ndays > 0 :
            x0m_ = (x0c[-ndays:,:]-x0c[:ndays,:])/rollwnd  # this is exactly rollwnd day's average
            x0m = np.vstack((x0m, x0m_))

        # note this is not a running std of XLi within rollwnd,
        # because each day has a different x0m, it is std against this
        # running mean for each day within the rollwnd
        x0d2c=np.cumsum((XLi-x0m)**2,axis=0)
        x0std = np.sqrt(np.dot(np.diag(1.0/(np.arange(rollwnd)+1)),x0d2c[:rollwnd,:]))
        if ndays > 0 :
            x0std_ = np.sqrt((x0d2c[-ndays:,:]-x0d2c[:ndays,:])/rollwnd)
            x0std = np.vstack((x0std, x0std_))

        # save latest state for real-time processing
        if save_state:
            self.decay_Li = decay_Li
            self.rollwnd = rollwnd
            self.Li = copy.deepcopy(Li)
            self.K = K

            self.x0std = copy.deepcopy(x0std[-1,:])
            self.X = copy.deepcopy(X[-1,-K:])  # needed for calculate decay'ed Li
            self.XLi = copy.deepcopy(XLi[-rollwnd:,:]) # needed for update x0m
            self.Xm = copy.deepcopy(x0m[-rollwnd:,:]) # needed for update x0std

            # for debuging
            #self.XLi0 = copy.deepcopy(XLi)
            #self.X0 = copy.deepcopy(X)
            #self.X0M = copy.deepcopy(x0m)
            #self.X0STD = copy.deepcopy(x0std)

        if ndays  == 0 :
            return []

        # XLi is the current day measure, x0m and x0std need to be the
        # mean and std upto current day but not including current day
        # to emphasis on the shock
        x = (XLi[-ndays:,:]-x0m[-ndays-1:-1,:])/x0std[-ndays-1:-1,:]

        # replace nan if there were any
        x = x.flatten()
        ixn = np.nonzero(np.isnan(x))[0]
        if len(ixn) > 0 :
            x[ixn] = 0

        if ewa is not None :
            B = int(np.clip(np.log(1e-5)/np.log(1-ewa),2,300))
            ew = (1-ewa)**np.arange(B-1,-1,-1)*ewa
            xx = lg.hankel(np.r_[np.zeros(B-1),x[:-(B-1)]],x[-B:])
            xx[0,:]/=(ewa)
            x=np.dot(xx,ew)
        return x.reshape((ndays,n))

    def ind3(self, decay_Li = [0.999,0.9], rollwnd = [50,50], ewa = [0.1,0.1], bucket_hour = [4.0,1], vbs=None) :
        """
        Get a positive and negative push indicator.
        Based on the previous ind2 of vbs, check the sign of same decay'ed lr. 
        divide ind2 into two series, the positive push, i.e. same sign with lr and otherwise.
        They are multiplied with  tanh'ed p-value of vol, which is ind2 on vol. 
        ewa for positive and negative can be different
        
        return 
        [vbs, positive_push, negtive_push]
        where each has shape of [ndays,n]
        """
        n = self.n
        ndays=self.ndays

        i2 = self.ind2(decay_Li=decay_Li[0],rollwnd=rollwnd[0],bucket_hour=bucket_hour[0],ewa=None, vbs=vbs).flatten()
        v2 = np.tanh(self.ind2(decay_Li=decay_Li[0],rollwnd=rollwnd[0],bucket_hour=bucket_hour[0],ewa=None,vbs=self.vol).flatten())+1.0
        i2*=v2

        lr2 = np.tanh(self.ind2(decay_Li=decay_Li[1],rollwnd=rollwnd[1],bucket_hour=bucket_hour[1],ewa=None,vbs=self.lr).flatten())

        xx=i2*lr2
        ixp = np.nonzero(xx>0)[0]
        ixn = np.nonzero(xx<0)[0]
        xx=i2*(np.abs(lr2))
        i3p = np.zeros((len(i2)))
        i3p[ixp]=xx[ixp]
        i3n = np.zeros((len(i2)))
        i3n[ixn]=xx[ixn]

        i3=[]
        for x, ewa in zip([i2, i3p, i3n], [None]+ewa):
            # replace nan if there were any
            x = x.flatten()
            ixn = np.nonzero(np.isnan(x))[0]
            if len(ixn) > 0 :
                x[ixn]=0

            # honor ewa
            if ewa is not None :
                B = int(np.clip(np.log(1e-5)/np.log(1-ewa),2,300))
                ew = (1-ewa)**np.arange(B-1,-1,-1)*ewa
                xx = lg.hankel(np.r_[np.zeros(B-1),x[:-(B-1)]],x[-B:])
                xx[0,:]/=(ewa)
                x=np.dot(xx,ew)
            i3.append(x.copy())

        return i3[0].reshape((ndays,n)), i3[1].reshape((ndays,n)), i3[2].reshape((ndays,n))

    # TODO - xp of [4,1] or [2,0.5] seems to be a good one
    def ind4(self, decay_Li = [0.999,0.999], rollwnd = [50,50], ewa = [None,None], bucket_hour = [4.0,1], vbs=None, vol=None) :
        """
        Get a positive and negative push indicator similar with ind3, with the following differences:
        1. use ind1 as a scaling factor
        2. when scaling with tanh, leave the same sign unchange, but scale down different sign using 1-tanh
        """
        if vbs is None :
            vbs = self.vbs
        vbs = vbs.reshape((self.ndays, self.n))

        ind2 = self.ind2(decay_Li = decay_Li[0], rollwnd = rollwnd[0], ewa = ewa[0], bucket_hour = bucket_hour[0], vbs = vbs).flatten()
        ind1 = self.ind1(decay_Li = decay_Li[1], rollwnd = rollwnd[1], ewa = ewa[1], bucket_hour = bucket_hour[1]).flatten()
        
        ixn = np.nonzero(ind2*ind1<0)[0]
        ixp = np.nonzero(ind2*ind1>0)[0]
        xp = ind2.copy() 
        xn = ind2.copy()
        if len(ixn) > 0 :
            xp[ixn]*=(1-np.abs(np.tanh(ind1[ixn])))
            xn[ixn]*=(1+np.abs(np.tanh(ind1[ixn])))

        if len(ixp) > 0 :
            xn[ixp]*=(1-np.abs(np.tanh(ind1[ixp])))
            xp[ixp]*=(1+np.abs(np.tanh(ind1[ixp])))
        return xp.reshape((self.ndays,self.n)), xn.reshape((self.ndays,self.n)), ind2.reshape((self.ndays,self.n)), ind1.reshape((self.ndays,self.n))

    # TODO - xm of [4,1], sf=0.25, 
    def ind5(self, decay_Li = [0.999,0.999], rollwnd = [50,50], ewa = [None,None], bucket_hour = [4.0,0.5], vbs=None, smooth_factor=0.25) :
        """
        Same as above ind4, except this uses ind2 as factor. 
        ind1 uses vbs/vol, so the volume is extracted out, only the unbalance. However, this unbalance could be big when vol is small.
             when vbs=-1,vol=1, it's too noisy.  The option added to use vbs/sqrt(vol)
        Also, to further add influence of ind2 (the volume), a tanh'ed factor of ind2 is added to ind1. 
        Finally, to smooth the overall indicator line, exponential factor of is applied.
        """
        if vbs is None :
            vbs = self.vbs
        vbs = vbs.reshape((self.ndays, self.n))

        ind2 = self.ind2(decay_Li = decay_Li[0], rollwnd = rollwnd[0], ewa = ewa[0], bucket_hour = bucket_hour[0], vbs = vbs).flatten()
        ind1 = self.ind1(decay_Li = decay_Li[1], rollwnd = rollwnd[1], ewa = ewa[1], bucket_hour = bucket_hour[1], sqrt=True, smooth_factor=0.5).flatten()

        ixn = np.nonzero(ind1*ind2<0)[0]
        ixp = np.nonzero(ind1*ind2>0)[0]
        x = ind1.copy() 
        if len(ixn) > 0 :
            x[ixn]*=(2-np.abs(np.tanh(ind2[ixn])))
        if len(ixp) > 0 :
            x[ixp]*=(1+np.abs(np.tanh(ind2[ixp])))
        xm = np.sign(x)*(np.abs(x)**smooth_factor)

        return xm.reshape((self.ndays,self.n)), x.reshape((self.ndays,self.n)), ind2.reshape((self.ndays,self.n)), ind1.reshape((self.ndays,self.n))

    def ind0(self,sqrt_vol=False) :
        """
        simplest form, get the vbs/vol, and replace na
        just to be consistent with the initial tests
        if sqrt_vol is true, then return vbs/sqrt(vol)
        """
        vbs = self.vbs.flatten()
        vol = self.vol.flatten()
        if sqrt_vol :
            vol = np.sqrt(vol)
        x = np.zeros(len(vbs))
        ixz=np.nonzero(vol)[0]
        if len(ixz) > 0 :
            x[ixz] = vbs[ixz]/vol[ixz]
        return x.reshape((self.ndays,self.n))

    def ind1(self, decay_Li = 0.8, rollwnd = 50, ewa = None, bucket_hour = 1.0, sqrt=False, smooth_factor=1.0) :
        """
        ind2 on ind1
        sqrt: use sqrt(vol) in ind0
        smooth_facctor: ind0 = sign * |ind0|**smoothfactor
        """
        vbs = self.ind0(sqrt_vol=sqrt)
        vbs = np.sign(vbs)*(np.abs(vbs)**smooth_factor)
        return self.ind2(decay_Li=decay_Li,rollwnd=rollwnd,ewa=ewa,
                bucket_hour=bucket_hour, vbs=vbs)

    """
    BELOW indicatoprs causes rr_est2 to not converge
    """
    def ind00(self):
        """
        just return the vbs, multiply |lr|
        """
        return np.sign(self.vbs)*np.sqrt(np.abs(self.vbs * self.lr))
    def ind01(self):
        """
        just return the vbs, multiply sqrt(|lr|) for same sign only
        """
        lr0=self.lr.flatten()
        x0 = self.ind0().flatten()
        ixp = np.nonzero(x0*lr0>0)[0]
        x=np.zeros(len(x0))
        x[ixp] = x0[ixp]*np.sqrt(np.abs(lr0[ixp]))
        return x.reshape((self.ndays,self.n))

    def ind000(self) :
        """
        just the vbs, not sure if this is the test_default,
        I forgot, so have to try
        """
        return self.vbs

class Ind_RealTime(object) :
    def __init__(self, vbs_obj, smooth_std=None):
        """
        A general real-time function for z-score like 
        running indicator, which maintains a running
        mean and std, from rollwnd (with XLi).
        Each update is aggregation of recent Li updates
        with a decay of decay_Li
        smooth_std: an integer, v0 is tanh'ed upto a maximum absolute value 
            of smooth_std times this bar's std, given by self.xs[k]

        Detail of the state:

        self.X:    stores the daily raw v0 plus a K+1 previous day's raw v0
                   vector of self.K + k
        self.XLi0: stores today's calculated indicator value x
                   vector of k, upto n
        self.XLi:  stores the rollwnd days of x
                   shape [rollwnd, n]
        self.Xm:   stores the rolling mean of previous rollwnd days (for calculating deviation)
                   shape [rollwnd, n], sync with XLi
                   Note, each day's mean is from previous rollwnd days
        self.x0m:  the current estimation of mean at each bar
                   vector of n
        self.x0std: same with x0m
        self.xm/xs: mean/std of raw v0 at each daily bar
                    vector of n

           |<- K ->|<------            n       --------------------->|
 
        X  +-------+-------------------------------------------------+
           |   K   |       current day's raw v0                      |
           +-------+-------------------------------------------------+

        XLi0       +-------------------------------------------------+
                   |       current day's ind val x                   |
                   +-------------------------------------------------+

        XLi        +-------------------------------------------------+  ---
                   |       previous day's ind val x                  |   ^
                   +------------------------------------------------ +   |
                   |    the day before previous day's ind val x      |   |
                   +-------------------------------------------------+  rollwnd                   
                   ...                                             ...   |
                   +                                                 +   |
                   |                                                 |   v
                   +-------------------------------------------------+  --- 

        Xm         +-------------------------------------------------+  ---
                   |  previous day's running mean of ind val x       |   ^
                   +------------------------------------------------ +   |
                   | the day before prev day's running mean of ind x |   |
                   +-------------------------------------------------+  rollwnd                   
                   ...                                             ...   |
                   +                                                 +   |
                   |                                                 |   v
                   +-------------------------------------------------+  --- 

        Detail of Usage: 

        Upon creation, the vbs_obj is supposed from backtest with the latest
        historical data, with "save_state" set to True (see tests below).

        Upon object initialization, call init_state()
        Upon a new day, call _upd_state(), which in turn calls init_state()

        EoD: run upd_state() and persist
        SoD: retrieve object, init_state(). If loaded intra-day, get raw v0 so far,
             and run on_bar()

        See init_state(), on_eod() and on_bar()
        """
        self.obj = vbs_obj
        self.smooth_std = smooth_std
        self.n = vbs_obj.n
        self.rollwnd = vbs_obj.rollwnd
        assert self.rollwnd > 2, "Ind state rollwnd too small"

        self.decay_Li = vbs_obj.decay_Li
        self.Li = copy.deepcopy(vbs_obj.Li)
        assert self.Li.shape[0] == self.n, "Ind Li state Li mismatch"

        self.K = vbs_obj.K
        assert self.K == np.max(self.Li) + 1, "Ind K not max(Li) + 1!"

        self.xm = copy.deepcopy(vbs_obj.xm)
        self.xs = copy.deepcopy(vbs_obj.xs)
        assert len(self.xm) == self.n and len(self.xs) == self.n, "Ind xm/xs length mismatch"

        self.X = np.r_[vbs_obj.X, self.xm]
        assert self.X.shape[0] == self.K + self.n, "Ind state X mismatch"

        self.XLi = copy.deepcopy(vbs_obj.XLi)
        self.Xm = copy.deepcopy(vbs_obj.Xm)
        assert self.XLi.shape[0] == self.rollwnd and self.XLi.shape[1] == self.n, \
                "Ind state XLi mismatch"
        assert self.Xm.shape[0] == self.rollwnd and self.Xm.shape[1] == self.n, \
                "Ind state Xm mismatch"

        # check nan
        x = np.r_[self.Li, self.X.flatten(), self.XLi.flatten(), \
                self.Xm.flatten(), self.xm.flatten(), self.xs.flatten()]
        ixn = np.nonzero(np.isnan(x))[0]
        assert len(ixn) == 0, "Ind state found nan in Li/X/XLi"

        self.init_state()

    def init_state(self) :
        """
        taking what is currently XLi, X, and Xm being update, initialize 
        intra-day states: 
            x0m and x0std, from XLi and Xm
            XLi0, to be x0m 
        Note calling multiple times after the eod and before first update
        yields same with calling once
        """

        # update x0m and x0std, note x0std is not a simply std
        # it is the average deviation of XLi from a running Xm in
        # rollwnd
        self.x0m = self.Xm[-1,:]
        self.x0std = np.sqrt(np.mean((self.XLi-self.Xm)**2,axis=0))
        # check value of x0std
        while True :
            ixn = np.nonzero(self.x0std<1e-5)[0]
            if len(ixn) == 0 :
                break
            ix = ixn[0]
            print ("got small running x0std, " + str(self.x0std[ix]) + \
                    "fix it from neighboring values to ")
            self.x0std[ix] = np.mean(self.x0std[np.arange(ix-5,ix+5)])
            print (str(self.x0std[ix]))
        self.XLi0 = copy.deepcopy(self.x0m)

    def on_eod(self) :
        """
        update at the end of day with the current state X and XLi0
        XLi: roll and include XLi0 
        Xm:  roll and include mean of the new XLi
        X:   keep the last K raw and fill remaining with xm
        And initialize the intra-day state of X and XLi0
        """
        self.XLi = np.vstack((self.XLi[1:,:], np.array(self.XLi0)))
        self.Xm = np.vstack((self.Xm[1:,:], np.mean(self.XLi,axis=0)))
        self.X = np.r_[self.X[-self.K:], self.xm]
        self.init_state()

    def on_bar(self, k, v0):
        """
        This needs to be updated in order.
        NOTE: the first bar at k=0 should be updated at the CLOSING
        time of the first bar. i.e. for 5 minute bar, update with k = 0
        should be at 18:05.  
        During 18:00 to 18:05, use on_snap() with k = 0
        v0 is a scalar value at k
        """
        return self._upd_bar(k,v0,update_state=True)

    def on_snap(self, k, a, v0) :
        """
        update at a snap time with a value of v0.
        k, a: snap time, defined by
        bar k and a fraction a \in [0, 1],
        a = 0 at the open of bar k, a = 1 at 
        closing of bar.
        v0 is the indicator value observed at 
        this snap time

        It adds to v0 with u0*(1-a), with u0 being
        the mean value at bar k.
        """
        u0 = self.xm[k]
        v0 += (u0*(1-a))
        return self._upd_bar(k, v0, update_state = False)

    def _upd_bar(self, k, v0, update_state = True) :
        """
        calculate the indicator value x from the raw input v0 at
        the closing time of bar k, k = 0, 1, ..., n-1. Last bar
        time is the closing time, i.e. 17:00.
        It is supposed to be updated in time order, but it allows
        out-of-order update in current intra-day.
        """
        K = self.K
        decay_Li = self.decay_Li
        n = self.n
        Li = self.Li

        if np.isnan(v0) :
            v0 = self.xm[k]

        if self.smooth_std is not None:
            std0 = self.xs[k]*12;
            v0 = np.tanh(v0/std0)*std0

        Li0=Li[k]
        W = decay_Li**np.arange(Li0-1,-1,-1)
        XLi0 = np.sum(np.r_[self.X[K+k-Li0+1:K+k],v0]*W)
        x1 = (XLi0 - self.x0m[k])/self.x0std[k]

        # update the intra-day state of XLi and X
        if update_state:
            self.X[K+k] = v0
            self.XLi0[k] = XLi0

        return x1 if not np.isnan(x1) else 0

    def persist(self, open_file) :
        dill.dump(self, open_file)

    @staticmethod
    def retrieve(open_file) :
        obj = dill.load(open_file)
        return obj


def run_vobj_rt(VBS_obj, val, n):
    """
    run indicator upon value upon a latest VBS_obj and
    return [ndays,n] indicator value
    """
    rt_obj = vbsind.Ind_RealTime(VBS_obj)
    nndays=len(val)
    ndays=nndays//n
    ret = []
    for d in np.arange(ndays):
        ret_d = []
        vd = val[d*n:(d+1)*n]
        for k, v0 in enumerate(vd):
            ret_d.append(rt_obj.on_bar(k,v0))
        ret.append(ret_d)
        rt_obj.on_eod()
    return np.array(ret)

def test_realTime(symbol = 'CL', start_date = '20190101', end_date = '20210701', decay_Li = 0.7, bucket_hour=1.0) :
    import pyres_test as test
    rollwnd = 50
    ndays, n, dt, lr, vol, vbs, lpx = test.get_data_ALL(symbol, start_date, end_date)

    # adjust the ndays to be a multiple of rollwnd
    ndays = ndays//rollwnd * rollwnd
    lr = lr[:ndays*n]
    vol = vol[:ndays*n]
    vbs = vbs[:ndays*n]
    lpx = lpx[:ndays*n]

    # create the reference run
    ind = VBS(ndays, n, lr, vol, vbs, barsec=300)
    refval = ind.ind2(decay_Li = decay_Li, rollwnd = rollwnd, ewa = None, bucket_hour = bucket_hour, save_state=True)

    # run update until the last rollwnd
    ind2 = VBS(ndays-rollwnd, n, lr[:-n*rollwnd], vol[:-n*rollwnd], vbs[:-n*rollwnd], barsec=300)
    refval0 = ind2.ind2(decay_Li = decay_Li, rollwnd = rollwnd, ewa = None, bucket_hour = bucket_hour, save_state=True)

    # run realtime on the last rollwnd
    ind_rt = Ind_RealTime(ind2)
    refval1 = []
    for d in np.arange(rollwnd) :
        print ("running day %d of %d"%(d, rollwnd))
        val1 = []
        vbs1 = vbs[(-rollwnd+d)*n:][:n]
        for k, v0 in enumerate(vbs1) :
            val1.append(ind_rt.on_bar(k, v0))
        refval1.append(val1)
        ind_rt.on_eod()
    refval2 = np.vstack((refval0, np.array(refval1)))

    # demand them to be equal
    assert np.max(np.abs(refval - refval2)) < 1e-10, "check failed!"

    # test the on_snap
    ind_rt2 = Ind_RealTime(ind2)
    refval3 = []
    refval4 = []
    for d in np.arange(rollwnd) :
        print ("running day %d of %d"%(d, rollwnd))
        val1 = []
        vbs1 = vbs[(-rollwnd+d)*n:][:n]
        for k, v0 in enumerate(vbs1) :
            x_snap = ind_rt2.on_snap(k, 1, v0)
            x_snap2 = ind_rt2.on_snap(k, 0.5, v0/2)
            x_bar = ind_rt2.on_bar(k, v0)
            assert np.abs(x_snap - x_bar) < 1e-10
            val1.append(x_bar)

            refval4.append(x_snap2)
            refval4.append(x_snap)

        refval3.append(val1)
        ind_rt2.on_eod()
    refval3 = np.vstack((refval0, np.array(refval3)))

    # demand them to be equal
    assert np.max(np.abs(refval - refval3)) < 1e-10, "check failed!"

    # persist and retrieve
    ind_rt3 = Ind_RealTime(ind2)
    refval5 = []

    fn = '/tmp/dilltestind.tmp'
    with open(fn, 'wb') as f :
        ind_rt3.persist(f)
        ind_rt3.persist(f)
    with open(fn, 'rb') as f :
        ind_rt3 = Ind_RealTime.retrieve(f)
        ind_rt3 = Ind_RealTime.retrieve(f)

    for d in np.arange(2) :
        print ("running day %d of %d"%(d, rollwnd))
        val1 = []
        vbs1 = vbs[(-rollwnd+d)*n:][:n]
        for k, v0 in enumerate(vbs1) :
            val1.append(ind_rt3.on_bar(k, v0))
        refval5.append(val1)
        ind_rt3.on_eod()

    with open(fn, 'wb') as f :
        ind_rt3.persist(f)
        ind_rt3.persist(f)
    with open(fn, 'rb') as f :
        ind_rt3 = Ind_RealTime.retrieve(f)
        ind_rt3 = Ind_RealTime.retrieve(f)

    for d in np.arange(rollwnd-2)+2 :
        print ("running day %d of %d"%(d, rollwnd))
        val1 = []
        vbs1 = vbs[(-rollwnd+d)*n:][:n]
        for k, v0 in enumerate(vbs1) :
            val1.append(ind_rt3.on_bar(k, v0))
        refval5.append(val1)
        ind_rt3.on_eod()

    refval5 = np.vstack((refval0, np.array(refval5)))
    # demand them to be equal
    assert np.max(np.abs(refval - refval5)) < 1e-10, "check failed!"


    return refval3[-rollwnd*n:,:].flatten(), np.array(refval4)

