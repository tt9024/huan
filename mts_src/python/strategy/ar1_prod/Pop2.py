import numpy as np
import copy
import dill

class Pop2 (object) :
    """
    sustain the previous forecasts according to historical covariance measurements
    """

    @staticmethod
    def default_param() :
        param = {'shp_dither': 0, \
                 'erf_dither': 0, \
                 'use_conditional': False, \

                 # weighted average of best shp 1, 2, .... n
                 # simple average of most recent -1,-2,...,-n
                 # multiple forecast group ensemble: 0 
                 'pick': -1, \

                 # in case pick=0,
                 # scale up to be more sensitive to shp 
                 # weighting for multi-group picking along a row
                 # applies as power of shp as probability 
                 #for multi-model mean/var
                 'multi_model_weight_scale': 1, \
                 'do_best_weight': True \
                 }
        return param

    def __init__(self, n, ixf, p0=0, fscl=None, fh=None, param = None) :
        """
        Set up the forecast table

        n: number of periods per day
        ixf: length nf vector of the forecast index
        fscl: shape [n,nf] of scale from Pop2's forecast onto model's forecast
        fh: shape [nndays,nf] of Pop2's forecast, to be compared with eod's f

        fcst[k,i] is the forecast of bar k given by previous i bar, i=0,1,...,n-1
        where [k, 0] is the forecast given at bar time immediate before.
        the over-night forecasts are at [-n:,:], persisted to the next day
        The table is expected to be checked in real-time and
        updated at on_bar(k)
        """
        self.n = int(n)
        nf = len(ixf)
        self.nf = nf
        self.ixf = np.array(ixf).copy().astype(int)
        self.ixd = (self.ixf - np.r_[-1, self.ixf[:-1]]).astype(int)

        self.fcst = np.zeros((2*n, n, 2))  # last index 0: return, 1: variance
        self.erf = np.zeros((n,nf))
        self.vf = np.zeros((n,nf))
        self.vr = np.zeros((n,nf))

        # setup ixf and weight
        ixnf = []
        for k in np.arange(self.n) :
            ix0 = []
            for j, (d, i1) in enumerate(zip(self.ixd, self.ixf)) :
                if k > i1 :
                    continue
                i0 = max(k, i1+1-d)
                ix0.append((j,i0,i1+1))
            ixnf.append(ix0)
        self.ixnf = ixnf
        
        # setup ixf index map
        ixmap = []
        for i,d in enumerate(self.ixd):
            ixmap=np.r_[ixmap,[i]*d]
        self.ixmap=np.array(ixmap).astype(int)

        # setup initial scale
        if fscl is None :
            fscl = np.ones((n,nf))
        self.fscl = fscl.copy()

        # parameters
        if param is None :
            param = Pop2.default_param()
        self.delete_zero_shp = False  # if set True, may reduce decay of old forecasts
        self.shp_dither = param['shp_dither']
        self.erf_dither = param['erf_dither']
        self.multi_model_weight_scale = param['multi_model_weight_scale']
        self.use_conditional = param['use_conditional']
        self.pick = param['pick']
        assert np.abs(self.pick) <= n
        self.do_best_weight = param['do_best_weight']

        self.min_neff = 12  
        self.neff_cnt_ratio = 0.65
        self.decay_row = False  # whether to decay older signal in a same ixf group

        # initial states
        self.p0 = p0
        self.ph = None
        self.fh = copy.deepcopy(fh) # shape [n,nf] history of Pop2's forecast 
        self.fhd = []  # list to be appended with f at each on_bar()

        # debug
        self.fcvc_prob = []
        self.fv = []


    def eod(self,  f, shp=None, v=None) :
        """
        update eod with return, forecast and variance
        f:  shape [nndays, nf] of forecasts at same period.
            f[0,0] is the forecast made at first day, bar, after 
            seeing the return of first bar
        dither_ratio: ratio (>0) of total population to use to dither 
                      a value with its neighboring values. the bigger
                      the smoother
        """
        nndays = f.shape[0]
        assert f.shape[1] == self.nf

        n = self.n
        ndays = nndays//n
        assert ndays*n == nndays
        assert ndays > 10, "too few days in roll window!"

        self.shp = None

        #update fcst scale_map
        nf = self.nf
        fd = f.reshape((ndays, n, nf))
        if self.fh is None :
            self.fh = f.copy()
        else :
            if len(self.fhd) == n :
                self.fh = np.vstack((self.fh,self.fhd))
                self.fhd = []
                fh = self.fh.reshape((ndays,n,nf))
                self.fscl = np.std(fd,axis=0)/np.std(fh,axis=0)
                ix = np.nonzero(np.isnan(self.fscl))[0]
                if len(ix) > 0 :
                    print ('trouble in setting the scale for ', ix, ', set them to 1')
                    self.fscl[ix] = 1
            else :
                print ('fhd not updated, fscl unchanged')


        # setup n-by-n map for self.shp and self.fscl 
        # self.wt[:,:] is the shp
        wt = np.zeros((self.n,self.n))
        self.wt = wt
        self.cur_bar = -1

        self.vr = None
        self.erf = None
        self.vf = None

        # roll the fcst table
        self.fcst=np.vstack((self.fcst[-n:,:,:],np.zeros((n,n,2))))

    def eod_noscale(self):
        n = self.n
        self.shp = None

        # setup n-by-n map for self.shp and self.fscl 
        # self.wt[:,:] is the shp
        wt = np.zeros((self.n,self.n))
        self.wt = wt
        self.cur_bar = -1

        self.vr = None
        self.erf = None
        self.vf = None

        # roll the fcst table
        self.fcst=np.vstack((self.fcst[-n:,:,:],np.zeros((n,n,2))))

    def on_bar(self, k, f, v):
        """
           fv =  on_bar(k, f, v)
        k: the bar time, 0 is the first bar from 18:00 to 18:05
        f,v: length nf vector of forecast, variance given at bar k

        return fv target positions given the forecast f at bar k
        fv is [256x4] array, [fc,vc,shp,v], fv_arr is [12x3] array [fc,vc,shp]
        """
        fc,vc=self._fc_vc_from_f(k,f, v)
        self._upd_fcst(k, fc, vc)
        #fv = self._fv_from_fcst(k)
        #self.cur_fv = fv

        #if k == self.cur_bar + 1 :
        #    self.fhd.append(f)
        self.cur_bar = k

        # update fh
        # TODO - enable f_scale scaling if needed
        ###f,v = self.gen_nf(self.ixf, scale_f=False, scale_f_shp_factor=None)

        # debug
        #self.fv.append(fv)
        #self.fcvc_prob.append(fcvc_prob)

    def gen_nf(self, ixf, scale_f=False, scale_f_shp_factor=None) :
        #    f,v = self.gen_nf(ixf, scale_f, scale_f_shp_factor)
        #
        # generate nf = len(ixf) forecasts after state from on_bar().
        # Given the current bar k, and current fv, it aggretates
        # nf forecasts according to ixf, and scale the forecasts
        # accordingly.  
        # scale_f: if true, try to scale the forecasts according to 
        #          coef's forecast scale, in particularly the default ixf
        #          it can be used for debuging purpose on the default ixf
        # scale_f_shp_factor: if not None and greater than 0, it further
        #          scales the nf forecasted return (f) according to their
        #          estimated shp. The shp is estimated from eod based on
        #          historical lr and fv
        # return f, v as vectors of length(ixf)

        k = self.cur_bar
        fv = self._fv_from_fcst(k)
        self.cur_fv = fv
        #fv = self.cur_fv

        if ixf is None :
            ixf = self.ixf
        ixd = ixf - np.r_[-1, ixf[:-1]]
        fv_nf = []
        for d, ix in zip(ixd, ixf) :
            i0 = ix+1-d
            i1 = ix+1
            f0 = fv[i0:i1,0]

            # TODO - review to remove it, 
            # why is it necessary?
            if scale_f :
                #f0 *= self.wt[k,i0:i1,1]
                raise "not supported!"

            v0 = fv[i0:i1,1]
            shp0 = np.clip(fv[i0:i1,2],0,1e+10)

            if self.pick == 0 and np.sum(shp0) > 0:
                wt = shp0/np.sum(shp0)
                f_ = np.dot(f0,wt)*d
                v_ = np.dot(v0,wt)*d
                shp_ = np.dot(shp0,wt)*np.sqrt(d)
            else :
                f_ = np.sum(f0)
                v_ = np.sum(v0)
                shp_ = np.sum(shp0)/np.sqrt(d)

            fv_nf.append([f_,v_,shp_])

        fv_nf = np.array(fv_nf)
        if scale_f_shp_factor is not None and scale_f_shp_factor > 0:
            f,v = self._scale_f_by_shp(self.cur_bar, fv_nf, ixf, scale_factor=scale_f_shp_factor)
        else :
            f = fv_nf[:,0]
            v = fv_nf[:,1]
        return f,v

    def _fc_vc_from_f(self, k, f, v) :
        """
            fc, vc = self._fc_vc_from_f(k, f. v)

        get the conditional fc and vc based from the forecast
        f,v:  length nf vector of forecast, variance from coef
        fc: length n vector of future conditional return
        vc: length n vector of future conditional variance of return

        The conditional variance is calculated as bi-normal 
        distribution:
        fc = erf * f/vf
        vc = v - erf^2/vf
        """
        fc = []
        vc = []
        for ix, (f0, v0, d) in enumerate(zip(f,v, self.ixd)):
            fc0 = [f0/d]*d
            vc0 = [v0/d]*d
            if self.use_conditional :
                fc0=[(f0*self.erf[k,ix]/self.vf[k,ix])/d]*d

                # take vc0 as a ratio of it's conditional variance over unconditional variance
                # and apply to bar ix's unconditional variance
                ratio0 = (self.vr[k,ix]-self.erf[k,ix]*self.erf[k,ix]/self.vf[k,ix])/self.vr[k,ix]

                #print ('ratio0', ratio0)
                i0 = self.ixf[ix]+1-d
                vc0 = ratio0*self.vr[(k+i0+np.arange(d).astype(int))%self.n, 0]

            fc = np.r_[fc, fc0]
            vc = np.r_[vc, vc0]

        return fc,vc

    def _upd_fcst(self, k, fc, vc):
        n = self.n
        self.fcst[np.arange(n).astype(int)+k, np.arange(n).astype(int), 0] = fc
        self.fcst[np.arange(n).astype(int)+k, np.arange(n).astype(int), 1] = vc
        #import pdb
        #pdb.set_trace()

    def _fv_from_fcst(self, k) :
        if self.pick < 0 :
            return self._fv_from_fcst_recent_n(k)
        raise "unknown pick value"

    def _fv_from_fcst_recent_n(self, k):
        n = self.n
        pk = -self.pick   # number of recents to average
        fv= []
        for i in np.arange(n) :
            i0 = i+k
            j0 = i
            j1 = min(n, j0+pk)
            # just pick the most recent

            d = j1-j0
            wt = np.ones(d)/d
            shp = self.wt[i0%n,j0:j1]
            if self.do_best_weight:
                ssm = np.sum(shp)
                if ssm > 0 :
                    wt = shp/ssm
            fv0 = np.dot(self.fcst[i0,j0:j1,:].T,wt)
            shp0 = np.dot(shp, wt)
            fv.append(np.r_[fv0, shp0])
        return np.array(fv)

    def _scale_f_by_shp(self, k, fv_nf, ixf, scale_factor=1.0) :
        """
        scale nf return forecasts based on their shp, 
        scale_factor is the power to the weight.  
        the weight is   (shp/shp.mean())^scale_factor, 
        where shp is in fv_nf[:,2]
        """
        if ixf is None :
            ixf = self.ixf
        ixd = ixf-np.r_[-1,ixf[:-1]]
        shp = fv_nf[:,2]/np.sqrt(ixd)
        fscl = (shp/shp.mean())**scale_factor
        return fv_nf[:,0]*fscl, fv_nf[:,1]


def run_pop(f, v, n, ixf, pick = [-47, -87, -113] , shp_dither=0, erf_dither=0, use_conditional=False, do_best_weight = False, out_ixf=None, p2_in=None, d0=0) :
    """
    out_ixf is a list of ixf to be output. default to be [ixf]
    d0 is the starting day of ndays in f and v
    """
    nndays = f.shape[0]
    ndays = nndays//n
    assert f.shape[1] == len(ixf)

    if out_ixf is None:
        out_ixf = [ixf]
    param = Pop2.default_param()
    param['shp_dither'] = shp_dither
    param['erf_dither'] = erf_dither
    param['use_conditional'] = use_conditional
    param['do_best_weight'] = do_best_weight

    print ('start simulating days of ', ndays-d0)
    if p2_in is None :
        p2 = Pop2(n,ixf, param=param) 
        p2.eod_noscale()
    else :
        p2 = copy.deepcopy(p2_in)
        # assuming p2_in has done eod, i.e. loaded from dill
    f_arr = []
    v_arr = []
    days = []
    for d in np.arange(ndays-d0)+d0:
        print ('sim day', d)
        ix0 = d*n
        f0 =  f[:ix0,:]
        v0 =  v[:ix0,:]
        days.append(d)
        #fvnarr = []
        for k in np.arange(n) :
            f_ = []
            v_ = []
            p2.on_bar(k, f[ix0+k,:],v[ix0+k,:] ) # fv is all that needed
            for pk in pick :
                p2.pick = pk
                #p2.on_bar(k, f[ix0+k,:],v[ix0+k,:] ) # fv is all that needed
                fix = []
                vix = []
                for out_ixf0 in out_ixf :
                    f0,v0 = p2.gen_nf(out_ixf0)
                    fix = np.r_[fix, f0]
                    vix = np.r_[vix, v0]
                f_.append(fix)
                v_.append(vix)

            # save the fvnf_arr
            f_arr.append(np.vstack(f_))
            v_arr.append(np.vstack(v_))
        p2.eod_noscale()

    return np.array(f_arr), np.array(v_arr),p2
 
