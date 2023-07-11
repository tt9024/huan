import Outliers as OT
import vbsind
import Pop

import numpy as np
import scipy.linalg as lg
import copy
import dill
import os
import datetime

bar_ix_dict = {None: 0, 'lr': 0, 'vol': 1, 'vbs': 2, 'lpx': 3, 'vbslr':0}

class BarIndicator :
    def __init__(self, ind_ix, ind_obj=None) :
        self.obj = copy.deepcopy(ind_obj)
        self.ix = ind_ix

    def on_bar(self, k, bar_array) :
        x0 = bar_array[self.ix]
        if self.obj is not None :
            return self.obj.on_bar(k, x0)
        return x0

    def on_snap(self, k, a, bar_array) :
        x0 = bar_array[self.ix]
        if self.obj is not None :
            return self.obj.on_snap(k, a, x0)
        return x0

    def init_state(self) :
        if self.obj is not None :
            self.obj.init_state()

    def on_eod(self) :
        if self.obj is not None :
            self.obj.on_eod()

class InputState :
    def __init__(self, data_ind_array, std, statelr, save_mi=False, save_stlrh=False):
        """
        This holds a set of indicators, generates stlr from md_dict

        data_ind_array: a data_array with third elements being the vind_obj
            for example [ 'CL', ind_ix, rt_obj ]
            where rt_obj is instance of Ind_RealTime, created using the
            saved vind_obj from the backtest, or None, just take raw value.
            ind_ix is the index into the bar data in md_dict.
            None if just take value. for example, for bar data of
              [lr, vol, vbs, lpx, ... ]
            'vbs' would have ind_ix = 2 to use vbs as input
        Assume vind_obj, std and statelr have been updated with the same data,
        and should be ready to be updated.  In particular, the vind_obj
        is assumed to have been at end of rollwnd. i.e. it is ready
        to run upd_snap/bar.  Typically unpacked from a dill dump file.

        It holds ind_array, which is responsible to generate m input
        from md_update on bar/snap update.  It also hold ind_in, shape
        [m, n] storing current day's input.

        on each bar close, stlr (length ni) is generated and a next_fun to
        use for snap updates before the closing of current bar.

        Snap update during an open bar, bar update happens
        upon the bar close.  State is updated at bar close, and
        at eod, after the last bar update.
        """
        self.ind_array = []
        for di in data_ind_array :
            sym, ix, obj = di
            self.ind_array.append([sym, BarIndicator(ix, obj)])
        self.std = copy.deepcopy(std)
        self.statelr = copy.deepcopy(statelr)
        self.n = self.std.n
        self.m = len(self.ind_array)
        self.ni = self.std.v.shape[0]
        self.next_fun = None
        self.stlr = np.zeros((self.ni,self.n))
        self.save_mi = save_mi
        self.save_stlrh = save_stlrh
        self.mi = []
        self.stlrh = []

    def _init(self) :
        self.ind_in = np.zeros((self.m, self.n))
        for ind0 in self.ind_array :
            ind0[1].init_state()
        self.stlr = np.zeros((self.ni,self.n))

    def on_bar(self, k, md_dict) :
        # call at closing time of bar k, k=0,1,...,n-1
        # where bar 0 is the first bar after open, i.e. 18:05.
        # md_dict has format of {symbol : bar_data}
        # where bar_data is an array i.e. [ lr, vol, vbs, lpx, ... ]
        # It creates m input, and update the
        # statelr in real-time
        # return state vector of ni
        mi = []
        for ind0 in self.ind_array :
            sym, obj = ind0
            mi.append(obj.on_bar(k, md_dict[sym]))
        self.ind_in[:,k] = np.array(mi)
        if self.save_mi:
            self.mi.append(np.array(mi))

        stlr, self.next_fun = self.statelr.realtime(self.ind_in[:,:k+1])
        self.stlr[:,:k+1] = stlr
        return stlr

    def on_snap(self, k, a, md_dict) :
        if self.next_fun is None :
            # haven't initialized yet, waiting for the first bar
            return self._default_next_fun()
        mi = []
        for ind0 in self.ind_array :
            sym, obj = ind0
            mi.append(obj.on_snap(k, a, md_dict[sym]))
        return self.next_fun(np.array(mi))

    def on_eod(self) :
        """
        This updates the indicators, statelr and the std.
        """
        for ind0 in self.ind_array :
            sym, obj = ind0
            obj.on_eod()
        # this is needed even we have self.stlr saved
        stlr = self.statelr.update(self.std, self.ind_in)
        self.std.update(stlr)
        if self.save_stlrh:
            self.stlrh.append(self.stlr)
        self._init()

    def _default_next_fun(self) :
        """ Todo: address this issue
        """
        print ("using default next fun, return all zeros")
        return np.zeros(self.ni)

class Forecast:
    def __init__(self, c, ni, stlrh = None, ixf=None, ot_ym = 5.0, ot_scl = 1.0) :
        """
        Forecast during bar k with a fraction of a,
        where a in [0,1], for example, at 18:01,
        k = 0, a = 0.2, for a 5min bar, is given as

             a*f1 + (1-a)*f2 + fs

        wereh
            f1 is forecast at open of bar k
            f2 is forecast at close of bar k with state at close being 0
            fs is contribution using bar k's closing coef with the snap

        Input:
        c:     shape [n, nf, nh], and c[k, :, nh-n+k+1:] are all zero,
               as the future values.
        ni:    number of input states
        stlrh: shape (ni, nh), with all the current day's value being zero,
               stlrh shape [ni, nh] with latest day's input at stlrh[:, -n:]
        ixf:   forecast index into the future
        ot_ym: maximum multiple of std to throttle state indicator values
        ot_scal: should be 1 always
        """
        self.c = c.copy()
        self.n, self.nf, self.nh = c.shape
        self.ni = int(ni)
        self.nh = self.nh // self.ni
        assert ixf is not None, 'ixf cannot be None'

        self.ixf = ixf.astype(int).copy()
        self.ot_ym = ot_ym
        self.ot_scl = ot_scl
        self.ixl = np.arange(-self.nh,0).astype(int)
        self.ixs = np.mod(self.ixl, self.n).astype(int)

        # f0 holds overnight's forecast at f0[-1]
        self.f0 = np.zeros((self.n+1, self.nf))
        self.f1 = 0
        self.f2 = 0

        if stlrh is None:
            stlrh = np.zeros((self.ni, self.nh))
        self.stlrh = stlrh.copy()

    def _init(self, stdv):
        """
        v is the updated std from previous day
        Note timing of this call:
        This is assuming that the eod has updated the f1 of last bar using
        that day's std, and then call this function with updated std and
        stlrh having updated with stlr as the latest -n elements

        Note MODEL have to make sure that:
        1. eod can only be called once a day
        2. bar updates having k in order, i.e. k go from 0 to n-1
        however, init() can be called multiple times after eod and before
        first bar update, (with same std/stlrh in int() all).
        init()  should be called upon object creation.
        """

        ni = self.ni
        n = self.n

        # update v
        self.stdvh = stdv[:, self.ixs]
        self.vf = self._v_from_sd(stdv, self.ixf) # note it's variance, not std
        stlrh0 = np.hstack((self.stlrh[:,n:],np.zeros((ni,n))))

        # update f0 using new v, except the last element
        # which contains the last f1 during eod
        lrch = np.ravel(OT.soft1(stlrh0[:,self.ixl],self.stdvh, self.ot_ym, self.ot_scl),order='F')
        self.f0[:-1,:] = np.dot(self.c,lrch)

        # clear the stlr
        self.stlr = np.zeros((ni,n))

        # initialize for the starting of the first bar
        self.next_k = 0
        self.f1 = self.f0[-1,:]
        self.f2 = self.f0[0,:]

    def on_eod(self, new_stdv):
        """
        Assuming on_bar() already called on bar=k-1, i.e.
        f1 reflects the forecast at closing time of bar k-1.
        """
        assert self.next_k == self.n-1, "eod called on bar " + str (self.next_k)
        self.f0[-1,:] = self.f1
        self.stlrh = np.hstack((self.stlrh[:,self.n:], self.stlr))
        self._init(new_stdv)

    def on_bar(self, k, stlr0) :
        """
        update when bar k is closed.
        recalculate f1 and f2 so that
        f1 is the forecast at next bar open
        f2 is the forecast at next bar close
        don't update f2 if last bar, wait for eod
        with new std
        """
        assert self.next_k == k, "expect on bar of " + str(self.next_k) + ", got " + str(k)
        ni = self.ni
        n = self.n
        nh = self.nh

        self.stlr[:,k] = stlr0
        self.f1, _ = self.on_snap(k, 1, stlr0)
        if k < n-1:
            # fs is the intra-day forecast
            lrnc = np.ravel(OT.soft1(self.stlr,self.stdvh[:,-n:], self.ot_ym, self.ot_scl),order='F')
            #fs = np.dot(self.c[k+2,:,(-nh-n)*ni:],lrnc)
            fs = np.dot(self.c[k+1,:,-n*ni:],lrnc)

            self.f2 = self.f0[k+1] + fs
            self.next_k += 1

    def on_snap(self, k, a, stlr0) :
        """update when bar k is open
           a: a fraction [0,1], when a=0 at open of bar k
              and 1 at close of bar k
        """
        k = int(k)
        assert self.next_k == k, "expect on snap of " + str(self.next_k) + ", got " + str(k)
        ni = self.ni
        n = self.n
        nh = self.nh

        lrch0 = np.ravel(OT.soft1(stlr0,self.stdvh[:,-n+k]*np.sqrt(a), self.ot_ym, self.ot_scl),order='F')
        f_rt = np.dot(self.c[k,:,(nh-n+k)*ni:(nh-n+k+1)*ni],lrch0)
        f = self.f1*(1-a) + self.f2*a + f_rt
        v = self.vf[k,:]
        return f, v

    def _v_from_sd(self,sd,ixf) :
        """
        calculates and outputs the variance 'v' associated with the
        predicted returns.
        Input:
            sd: shape [ni, n] of standard deviation, note only
                sd[0,:] is assumed to be the lr, and is used
            ixf: length nf vector of index of n bars into future

        Return:
            v: shape [n, nf] of variance glueing lr (sd[0,:]),
               at each bar close, into ixf
        Note it's variance, not the same as std.v, which is standard dev
        """
        ni,n = sd.shape
        nf=ixf.shape[-1]
        csv=np.cumsum(lg.hankel(np.r_[sd[0,1:],sd[0,0]]**2,sd[0,:]**2),axis=1)
        if len(ixf.shape)==1:
            v=np.hstack((csv[:,0:1],csv[:,ixf[1:]]-csv[:,ixf[:-1]]))
        else :
            v=np.zeros((n,nf))
            v[0,:]=np.r_[csv[0,0],csv[0,ixf[0,1:]]-csv[0,ixf[0,:-1]]]
            for k in np.arange(1,n):
                v[k,:]=np.r_[csv[k,0],csv[k,ixf[k,1:]]-csv[k,ixf[k,:-1]]]
        return v

class ForecastModel (object) :
    def __init__(self, fcst_obj, state_obj) :
        assert fcst_obj.n == state_obj.n and fcst_obj.ni == state_obj.ni, "n or ni mismatch when creating ForecastModel!"
        self.fcst_obj = copy.deepcopy(fcst_obj)    # has the c, stlrh
        self.state_obj = copy.deepcopy(state_obj)  # has the ind, std, statelr
        self._init()

    def _init(self) :
        self.state_obj._init()
        stdv = self.state_obj.std.v
        self.fcst_obj._init(stdv)

    def on_bar(self, k, md_dict) :
        stlr = self.state_obj.on_bar(k, md_dict)
        self.fcst_obj.on_bar(k, stlr[:,-1])

    def on_snap(self, k, a, md_dict) :
        stlr0 = self.state_obj.on_snap(k,a,md_dict)
        return self.fcst_obj.on_snap(k,a,stlr0)

    def on_eod(self) :
        self.state_obj.on_eod()
        new_stdv = self.state_obj.std.v
        self.fcst_obj.on_eod(new_stdv)

    def run_days(self, md_days) :
        """
        run the model with multi day dict:
        md_days: market data to run, format of {sym: daily_bar},
                 where daily_bar has shape of [ndays, n, 4], with
                 last dim being lr, vol, vbs, lpx
        return:
          f, v, stlrh, where f,v shape [ndays*n, nf], and stlrh shape [ni,nh]
        """

        # validate the md_days
        symbols = list(md_days.keys())
        assert len(symbols) > 0, "empty md_days"
        ndays, n, l = md_days[symbols[0]].shape
        assert n == self.fcst_obj.n, "%s dimention mismatch with fcst object!"%(symbols[0])
        assert l >= 4, "%s bar dimension less than 4!"%(symbols[0])
        for sym in symbols[1:] :
            ndays0, n0, l0 = md_days[sym].shape
            assert ndays0==ndays and n0 == n and l0 == l, "%s shape mismatch for md_days"%(sym)

        f = []
        v = []
        self._init()
        for d in np.arange(ndays).astype(int) :
            # run for the day d
            for k in np.arange(n).astype(int) :
                # run for the day
                md_dict = {}
                for sym in md_days.keys() :
                    md_dict[sym] = md_days[sym][d,k,:]
                f0,v0 = self.on_snap(k,1,md_dict)
                f.append(f0)
                v.append(v0)
                self.on_bar(k,md_dict)
            self.on_eod()

        return np.array(f), np.array(v), self.fcst_obj.stlrh.copy()

    def pop_baseline(self, f, v, p0):
        nf = 12
        w = np.r_[np.linspace(0.5,2,8), np.linspace(2, 0.1, nf-8)]
        t0 = np.ones(nf+1)*0.003992
        pp = Pop.opt(p0,f,v, t0, w)

        # tanh and clip
        mp = 50.0; clp = 35.0
        pp0 = np.clip(np.tanh(pp[0]/mp)*mp,-clp,clp)/clp
        return pp0, pp

    @staticmethod
    def gen(ind_array, std, statelr, c, stlrh = None, ixf=None, ot_ym = 5.0, ot_scl = 1.0, save_mi=False, save_stlrh=False):
        """
        ind_array:  array of [symbol, (ind_type, bucket, decay), ind_obj]
        """
        data_ind_array = [] # [ 'CL', ind_ix, rt_obj ]
        for ia in ind_array:
            sym, ind, ind_obj = ia
            ind_ix = bar_ix_dict[ind[0]]
            if ind_obj is None :
                rt_obj = None
            else :
                rt_obj = vbsind.Ind_RealTime(ind_obj)
            data_ind_array.append([sym, ind_ix, rt_obj])
        state_obj = InputState(data_ind_array, std, statelr, save_mi=save_mi, save_stlrh=save_stlrh)
        ni = std.v.shape[0]
        fcst_obj = Forecast(c, ni, stlrh = stlrh, ixf=ixf, ot_ym=ot_ym, ot_scl=ot_scl)
        return ForecastModel(fcst_obj, state_obj)

    @staticmethod
    def gen_param(coef_dict, param_dict, cix=-1, dump_fn = None) :
        # prepare a dict for running TestForecastModel
        test_dict = {}
        coef_keys = ['barsec', 'ndays', 'Ninit', 'Ninit2', \
                     'Nupd', 'OTym', 'OTscl', 'ixf','f','v', \
                     'n', 'c_0', 'c_0_dates']
        param_keys = ['sim_param', 'stlrh', 'vind_obj']
        for ck in coef_keys:
            test_dict[ck] = coef_dict[ck]
        for pk in param_keys:
            test_dict[pk] = param_dict[pk]
        test_dict['c'] = test_dict['c_0'][cix]
        test_dict['c_0'] = None
        test_dict['data_days'] = coef_dict['ind']['data_days']

        return test_dict

    @staticmethod
    def load(coef_dict, param_dict, cix=-1):
        """
        coef_dict loaded from the coef dump file
        param_dict loaded from the param dump file
        """
        ind_array = param_dict['vind_obj'] # in format of [symbol, ind_type, ind_obj]
        ixf = coef_dict['ixf']
        std = param_dict['std']
        statelr = param_dict['statelr']
        stlrh = param_dict['stlrh']
        c_0 = coef_dict['c_0']
        c_0_dates = coef_dict['c_0_dates']
        c = c_0[cix]

        print ("ForecastModel using coef with c_0_dates %d"%(c_0_dates[cix]))
        fm = ForecastModel.gen(ind_array, std, statelr, c, stlrh = stlrh, ixf=ixf)
        param = ForecastModel.gen_param(coef_dict, param_dict, cix=cix)
        return fm, param

