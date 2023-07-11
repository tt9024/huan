import numpy as np
import copy
import dill

class NfNx :
    def __init__(self, n, ixf, stdv=None) :
        """
        ixf: length nf vector
        stdv: length n vector of the standard deviation of return, default to be all 1
        """
        self.n = int(n)
        self.ixf = np.array(ixf).copy().astype(int)
        self.nf = len(ixf)
        self.nx = ixf[-1]
        if stdv is None :
            stdv = np.ones(n)
        self.upd_stdv(stdv)

    ###
    # roll
    ###
    def upd_stdv(self, stdv):
        self.stdv=stdv.copy()
        self.w,self.wix=self._w_from_ixf_stdv(stdv,self.ixf)

    ###
    # from f_nf to f_nx
    ###
    def fnx_days(self,fnf_days):
        """ 
        fnf_days: shape (nndays,nf)
        return f_nx: shape (nndays,nx) of extended forecasts
        """
        nndays,nf=fnf_days.shape
        return np.dot(fnf_days,self.wix)*self.w[np.arange(nndays).astype(int)%self.n,:]

    def fnx_k(self,k,fnf):
        return np.dot(fnf,self.wix)*self.w[k,:]

    ###
    # from f_nx to f_nf
    ###
    def fnf_days(self, fnx_days):
        """ 
        fnx_days: shape (nndays, nx)
        return fnf_days: shape (nndays, nf)
        """
        return np.dot(self.wix,fnx_days.T).T

    def fnf_k(self, fnx):
        return np.dot(self.wix,fnx)

    def _wix_from_ixf(self, ixf):
        nf = len(ixf)
        nx = ixf[-1]+1
        wix = np.zeros((nf,nx))
        nf_ix = np.r_[-1,ixf]+1
        for k in np.arange(nf).astype(int):
            wix[k,nf_ix[k]:nf_ix[k+1]]=1
        return wix

    def _w_from_ixf_stdv(self, stdv, ixf0) :
        """
            w,wix=w_from_ixf_stdv(stdv,ixf)

        stdv: length n vector of standard deviation, stdv[0] is the std of first bar
        ixf0: length nf vector of forecast index, ixf0[0] has to be 0, ixf0[-1]+1 <= n
              so that ixd, the forecast period is
              ixd = ixf-np.r_[0,ixf0[:-1]]
        return:
        w,wix:  shape (n,nx) and (nf,nx) so that 
            f_nx(k) = f_nf(k) \dot wix * w[k,:]
            where f_nf(k) is the nf forecasts made at end of bar k
            and f_nx(k) is the nx bar-by-bar forecasts extended from f_nf

        f_nf(k,i) is extended with ixd(i) forecasts f_j, where f_j = f_nf(k,i)/v*v_j, where
             v is total variance in ixd, and v_j is the variance of bar j
             this is to keep the f_nf(k,i)/v(ixd(i)) = f_j/v_j
        """
        ixf = np.array(ixf0).astype(int)
        n = stdv.shape[0]
        nx = ixf[-1]+1
        nf = len(ixf)
        assert nx<=n
        assert ixf[0]==0
        v=stdv**2
        sdc = np.cumsum(np.tile(v,(1,3)))
        ix1 = ((np.tile(ixf,(n,1))).T+np.arange(n).astype(int)).T+1
        ix0 = ((np.tile(np.r_[-1,ixf[:-1]],(n,1))).T+np.arange(n).astype(int)).T+1
        w0 = sdc[ix1]-sdc[ix0]
        # populate to from n-nf to n-nx, where
        # v_k_i/vc_k_i, for k in n, i in nx
        # v is the variance at k,i, vc is the sum of variance from nf
        wix = self._wix_from_ixf(ixf)
        wc = np.dot(w0,wix) # total variance in n-by-nx

        wvix = ((np.tile(np.arange(nx),(n,1)).T+(np.arange(n)+1))%n).astype(int).T
        w=v[wvix]/wc  # v_j/v(ixd(i)) in n-by-nx
        return w,wix

class BitMaskDefault:
    """
    Simple filter so that the masked results are ready to be added to the fcst_table,
    fnx -> (mask) -> fnx_mask -> add to fcst_table
    i.e. fnx_mask has the same length of fnx, with all the masked position 0
    Alternatively, BitMask() returns chosen index from fnx, for possible
    aggregation based on a variance.  This is currently under development in 
    ni_pick's QR on maximum sharp regression of M^{-1}N. 
    """
    def __init__(self, mask):
        n,n1=mask.shape
        assert n==n1
        self.mask=np.zeros(mask.shape)
        mask2 = np.vstack((mask,mask))
        ix = np.arange(n).astype(int)
        for k in np.arange(n).astype(int) :
            self.mask[k,:] = mask2[ix+k,ix]

    def fnx_k(self, k, fnx):
        return fnx*self.mask[k]

    def fnx_days(self, k, fnx):
        nk,n=fnx.shape
        ndays = int(np.ceil(nk/n))+1
        return fnx*(np.tile(self.mask,(ndays,1))[k:k+nk,:])

class BitMask:
    def __init__(self, mask):
        """
        mask: shape n-by-n 0/1, indexed for fcst_table
              mask[i,j] applies to fcst_table[i,j]
        It holds:
        flist: list of n arrays of set index, within 'n'
        fcnt:  shape (n,n) cnt of f at each bar, with 
               fcnt[k,i] being number of f (from flist[k]) for f at k+i
               so that 
               flist[k][Nk-fcnt[k,i]:Nk] is the index to use
        """
        n0,n1=mask.shape
        assert n0==n1
        self.n=n0

        msk_k = np.triu(np.ones((self.n,self.n)),0).astype(int)
        self.mask = mask.copy().astype(int)
        self.flist = [[] for i in np.arange(self.n)]
        self.fcnt = np.cumsum(self.mask[:,::-1],axis=1)
        ix = np.nonzero(self.mask==1)
        if len(ix[0]) > 0:
            ix = np.vstack((ix[0],ix[1])).T.astype(int)
            for (ix0,ix1) in ix:
                self.flist[ix0].append(ix1)
        self.flist=[np.array(ix).astype(int) for ix in self.flist]
        self.fcnt=np.array([ix[::-1] for ix in self.fcnt]).astype(int)

    @staticmethod
    def gen_default(n) :
        return BitMask(np.ones((n,n)).astype(int))

    def _get_ix(self, k, j):
        """
        k: [0,n-1], current bar index
        j: [0,n-1], future bar relative to current bar.
        """
        k0 = int((k+j)%self.n)
        ix = self.flist[k0]
        Nk=len(ix)
        return ix[Nk-self.fcnt[k0,j]:Nk]

    def _get_fnx(self, k, js, fnx):
        """
        k: current bar
        js: len Nj list of future bars
        fnx: shape[>Nj,n] of forecast, where fnx[0,:] current bar
        return:
        fnx_list: list of chosen fnx
        """
        fnx0 = []
        for j in js:
            fnx0.append(fnx[j,self._get_ix(k,j)])
        return fnx0

    def get_fnx(self, k, fnx):
        nx, _ = fnx.shape
        return self._get_fnx(k, np.arange(nx).astype(int), fnx)

class FcstTable:
    """
    fcst_table, from getting (multiple) input(s) from fni_nf and update a fcst_table
    From fni_nf to fcst_table:
        fni_nf --(fscale)--> fni_nfs --(ixf)--> fni_nx --(mask)--> fni_nxm --(add_to_diag)--> fcst_table
        Note: There could be multiple fni_nf applying to the same fcst_table
        See: ni_update_k(k, fni_nf_dict), ni_update_day(fni_nf_dict)
    From fcst_table to fni_nf
        fcst_table --(weighted_avg_shp)--> fnx
        See: fnx_k(k, nx), fnx_day(nx)
    """
    def __init__(self, n, ni_dict, prev_day_trih=None, save_fcst=False, stdv=None):
        """
        ni_dict: {'model_name': {'ixf', 'fscale','tbl_shp'} }
                  ixf: length nf forecast index
                  fscale: shape (n,nf) scale of fni_nf
                  tbl_shp: shape (n,n) for shp on fcst_table of that fni
        stdv: length n vector of standard deviation of the target instrument
              used for decompose nf to nx
        """
        self.n=n
        self.ni_dict = ni_dict
        mask = np.zeros((n,n))
        fshp = np.zeros((n,n))
        if stdv is None:
            stdv=np.ones(n)
        assert len(stdv)==n
        self.stdv=stdv.copy()
        for mn in ni_dict.keys() :
            nid=ni_dict[mn]
            ni_dict[mn]['nfnx']=NfNx(n,nid['ixf'],self.stdv)
            tbl_mask = np.zeros((n,n)).astype(int)
            nzix=np.nonzero(nid['tbl_shp']>0)
            if len(nzix[0])>0:
                tbl_mask[nzix]=1
            ni_dict[mn]['bm']  =BitMaskDefault(tbl_mask)
            mask += tbl_mask
            fshp = np.max(np.array([fshp, nid['tbl_shp']]),axis=0)

        mask=np.array(mask).astype(int)
        wt_cnt=np.ones((n,n))
        ix=np.nonzero(mask>1)
        if len(ix[0]) > 0:
            wt_cnt[ix]=1.0/mask[ix].astype(float)
        self.wt_cnt=np.vstack((wt_cnt,wt_cnt))
        self.fshp=np.vstack((fshp,fshp))
        self.wt_shp_cs = np.cumsum(self.fshp[:,::-1]+1e-12,axis=1)[:,::-1]
        self.mask = mask # for debug

        self.fcst=np.zeros((n*2,n))
        if prev_day_trih is not None:
            self.fcst[:n,:n]=prev_day_trih
        self.nix=np.arange(self.n).astype(int)
        self.fcsth = np.array([]).reshape((0,self.n))
        self.save_fcst=save_fcst

    def ni_update_k(self, k, fni_nf_dict):
        """
        fni_nf_dict: {'model_name': fni_nf}
        Updates the fcst table from each ni
        Note fni_nf_dict should include all models
        """
        for mn in self.ni_dict.keys():
            nid=self.ni_dict[mn]
            fni_nf = fni_nf_dict[mn]*nid['fscale'][k]
            fni_nx = nid['bm'].fnx_k(k,nid['nfnx'].fnx_k(k,fni_nf))
            # apply to fcst
            self.fcst[self.nix+k,self.nix]+=fni_nx

        # normalize with wt_cnt
        self.fcst[self.nix+k,self.nix]*=self.wt_cnt[self.nix+k,self.nix]

    def ni_update_day(self, fni_nf_dict):
        """
        fni_nf_dict[model_name] is now a (n,nf) 
        """
        for k in np.arange(self.n).astype(int):
            for mn in self.ni_dict.keys():
                nid=self.ni_dict[mn]
                fni_nf = fni_nf_dict[mn][k,:]*nid['fscale'][k]
                fni_nx = nid['bm'].fnx_k(k,nid['nfnx'].fnx_k(k,fni_nf))
                # apply to fcst
                self.fcst[self.nix+k,self.nix]+=fni_nx

            # normalize with wt_cnt
            self.fcst[self.nix+k,self.nix]*=self.wt_cnt[self.nix+k,self.nix]

    def fnx_k(self, k, nx=48) :
        # get a simple wt-avg from shp
        fcst=self.fcst[k:k+nx,:]
        wt_shp=(np.triu(self.fshp[k:k+nx,:],0).T/np.diag(self.wt_shp_cs[k:k+nx,:])).T
        return np.sum(fcst*wt_shp,axis=1)

    def fnx_day(self, nx=48) :
        fnx=[]
        for k in np.arange(self.n).astype(int):
            fnx.append(self.fnx_k(k,nx=nx))
        return np.array(fnx)

    def fnx_hist_latest(self, klist, nx=48):
        """
        getting the latest (diagonal) forecast, klist index
        from all the history of fcsth, mainly for testing
        klist: list of bars in history.
               k=0 starting from the first in fcsth
               k<fcsth.shape[0]+n
        """

        fnx=[]
        fcsth=np.vstack((self.fcsth,self.fcst))
        for k in klist:
            fnx.append(np.diag(fcsth[k:k+nx,:nx]))
        return np.array(fnx)

    def eod(self, stdv=None):
        n=self.n
        if self.save_fcst:
            self.fcsth=np.vstack((self.fcsth,self.fcst[:n,:]))
        self.fcst[:n,:]=self.fcst[-n:,:]
        self.fcst[-n:,:]=0

        if stdv is not None:
            self.stdv=stdv.copy()
            for mn in self.ni_dict.keys():
                nid=self.ni_dict[mn]
                self.ni_dict[mn]['nfnx']=NfNx(n,nid['ixf'], self.stdv)

def gen_ni_dict(model_name, n, ixf, fscale=None, tbl_shp=None):
    nf=len(ixf)
    if fscale is None:
        fscale=np.ones((n,nf))
    if tbl_shp is None:
        tbl_shp=np.ones((n,n))
    return {model_name:{'ixf':copy.deepcopy(ixf), \
                        'fscale':fscale.copy(), \
                        'tbl_shp':tbl_shp.copy()
                       }
           }

def run_days(f,v,n,ixf,ixf_out,fscale=None, tbl_shp=None, prev_day_triu=None, lr=None):
    nndays,nf=f.shape
    assert v.shape == f.shape
    assert len(ixf)==nf
    if lr is not None:
        assert len(lr)==nndays
    for ixf0 in [ixf, ixf_out]:
        assert ixf0[0] == 0 and ixf0[-1] < n

    stdv=np.sqrt(np.r_[v[n-1,0],v[:n-1,0]])
    ni_dict=gen_ni_dict('rundays', n, ixf, fscale=fscale, tbl_shp=tbl_shp)
    ft=FcstTable(n,ni_dict,save_fcst=True,stdv=stdv)
    fnx = []
    nx = ixf_out[-1]+1

    for d in np.arange(nndays//n).astype(int):
        print ('day ', d)
        ni_upd={'rundays':f[d*n:(d+1)*n,:]}
        ft.ni_update_day(ni_upd)
        fnx.append(ft.fnx_day(nx))
        stdv=np.sqrt(np.r_[v[(d+1)*n-1,0],v[d*n:(d+1)*n-1,0]])
        ft.eod(stdv=stdv)

    nfnx=NfNx(n, ixf_out)
    fnx=nfnx.fnf_days(np.vstack(fnx))
    if lr is not None:
        # try to get a sharp out of the fcst_table
        # note the first day may have half table missing if prev_day_triu is None
        ndays=nndays//n
        fcst_table_shp=(ft.fcsth.T*np.r_[lr[1:],0]).T.reshape((ndays,n,n))

    return ft, fcst_table_shp, fnx

####
# Unit Tests
###

def test_nf_nx() :
    ixf1 = np.array([0, 1, 2, 5, 9])
    stdv = np.arange(10)+1
    fnf = np.arange(10).reshape((2,5))
    n=10
    nfnx1 = NfNx(10, ixf1, stdv=stdv)
    fnx_days = nfnx1.fnx_days(fnf)

    fnx_days0 = np.array([[0.,1.,2.,0.68181818, 0.98181818, 1.33636364,  1.04065041, 1.31707317, 1.62601626, 0.01626016], \
                          [5.,6.,7.,1.93288591, 2.63087248, 3.43624161, 3.91935484, 4.83870968, 0.0483871,  0.19354839]])
    assert np.max(np.abs(fnx_days-fnx_days0)) < 1e-7

    fnx0 = nfnx1.fnx_k(0,fnf[0,:])
    fnx1 = nfnx1.fnx_k(1,fnf[1,:])
    assert np.max(np.abs(fnx_days-np.vstack((fnx0,fnx1)))) < 1e-10

    # check fcst_table
    fcst_table = np.zeros((20,10))
    fcst_table2 = np.zeros((20,10))
    ixn0 = get_ixn(1,10)
    ixn1 = get_ixn(2,10)

    upd_fcst_days(fcst_table, fnx_days, ixn1)
    upd_fcst_1fnx(fcst_table2, 0, fnx_days[0,:], ixn0)
    upd_fcst_1fnx(fcst_table2, 1, fnx_days[1,:], ixn0)
    assert np.max(np.abs(fcst_table-fcst_table2)) < 1e-10

    # get from diag and get back to nf
    fnx0 = np.diag(fcst_table)
    fnx1 = np.diag(fcst_table[1:,:])
    assert np.max(np.abs(fnx_days- np.vstack((fnx0,fnx1)))) < 1e-10

    fnf0 = nfnx1.fnf_k(fnx0)
    fnf1 = nfnx1.fnf_k(fnx1)

    fnf01 = nfnx1.fnf_days(np.vstack((fnx0,fnx1)))
    assert np.max(np.abs(fnf01-np.vstack((fnf0,fnf1)))) < 1e-10
    assert np.max(np.abs(fnf01-fnf)) < 1e-10

    ixf2 = np.array([  0,   1,   2,   5,   9,  15,  25,  41,  67, 107, 172, 275])
    stdv = np.arange(276)+1
    fnf = np.arange(5*276*12).reshape((5*276,12))
    n = 276
    nfnx2=NfNx(n, ixf2, stdv=stdv)
    fnx_days = nfnx2.fnx_days(fnf)
    fnf_days = nfnx2.fnf_days(fnx_days)
    assert np.max(np.abs(fnf_days-fnf)) < 1e-10

def test_mask() :
    mask = np.array([ [0, 1, 0, 1, 1],
                      [1, 0, 1, 0, 1],
                      [1, 1, 0, 1, 0],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 1, 0]])
    bm = BitMask(mask)
    assert np.max(np.abs(bm._get_ix(0,0)-np.array([1,3,4]))) == 0
    assert np.max(np.abs(bm._get_ix(0,1)-np.array([2,4])))==0
    assert np.max(np.abs(bm._get_ix(0,2)-np.array([3])))==0
    assert np.max(np.abs(bm._get_ix(0,3)-np.array([4])))==0
    assert len(bm._get_ix(0,4)) ==0

    fnx = np.arange(25).reshape((5,5))
    fnx0 = bm._get_fnx(1,np.arange(5),fnx)
    fnx0_ref = [np.array([0, 2, 4]), np.array([6, 8]), np.array([12, 14]), np.array([18]), np.array([24])]
    for arr0, arr1 in zip(fnx0, fnx0_ref) :
        if len(arr0)==0:
            assert len(arr1)==0
        else :
            assert np.max(np.abs(arr0-arr1)) == 0

    fnx0 = bm._get_fnx(3,np.arange(5),fnx)
    fnx0_ref = [np.array([0, 1, 2, 4]), np.array([6, 7, 8]), np.array([13, 14]), np.array([19]), np.array([])]
    for arr0, arr1 in zip(fnx0, fnx0_ref) :
        if len(arr0)==0:
            assert len(arr1)==0
        else :
            assert np.max(np.abs(arr0-arr1)) == 0


ni_dict={\
         'mdl1':{'ixf': np.array([0,1,2,5,9]),\
                 'fscale': np.ones((10,5)), \
                 'tbl_shp': np.eye(10)}, \
         'mdl2':{'ixf': np.array([0,1,2,5,9]),\
                 'fscale': np.ones((10,5))*0.5, \
                 'tbl_shp': np.triu(np.ones((10,10)),0)}, \
        }

def test_fcst_table() :
    ft = FcstTable(10,ni_dict,save_fcst=True, stdv=np.arange(10)+1)
    ni_upd={'mdl1':np.arange(5),'mdl2':np.arange(5)}
    ft.ni_update_k(0,ni_upd)
    fnx=ft.fnx_k(0,nx=10)
    fnx_ref_0=np.array([0., 0.08333333, 0.1875, 0.07305195, 0.12272727, \
            0.20045455, 0.19512195, 0.32926829, 0.6097561 , 0.01219512])
    assert np.max(np.abs(fnx-fnx_ref_0)) < 1e-8
    ft.ni_update_k(1,ni_upd)
    ft.ni_update_k(2,ni_upd)
    fnx=ft.fnx_k(2,nx=10)
    fnx_ref_2=np.array([0.1875, 0.07305195, 0.12272727, 0.20045455, 0.19512195,\
            0.32926829, 0.6097561 , 0.01219512, 0.0565931 , 0.15789474])
    assert np.max(np.abs(fnx-fnx_ref_2)) < 1e-8
    ft.ni_update_k(9,ni_upd)
    ft.eod()
    
    n=10
    ft.fcst[-n:,:]=ft.fcst[:n,:]
    ft.fcst[:n,:]=ft.fcsth[:n,:]
    fnx=ft.fnx_k(2,nx=10)
    assert np.max(np.abs(fnx-fnx_ref_2)) < 1e-8

def test_all() :
    test_nf_nx()
    test_mask()
    test_fcst_table()
    print ('good!')


###
# fast indexing from fnx to fcst_table
###

def get_ixn(nk, nx):
    """
    setup index for applying shape (nk,nx) fnx onto fcst_table
    return:
       ixn, shape (nk+nx-1,nx) index, so that
            fcst_table[:(nk+nx-1),:nx] = np.r_[fnx.flatten(),0][ixn]
    """
    zero_ix=-1 
    ixn = np.ones((nk+nx-1)*nx).astype(int)*int(zero_ix)
    for k in np.arange(nk).astype(int):
        ixn[np.arange(nx).astype(int)*(nx+1)+k*nx]=np.arange(nx*k,nx*(k+1)).astype(int)
    return ixn.reshape((nk+nx-1,nx))

def upd_fcst_days(fcst_table, fnx, ixn):
    nndays,nx=fnx.shape
    fcst_table[:(nndays+nx-1),:nx]+=np.r_[fnx.flatten(),0][ixn]

def upd_fcst_1fnx(fcst_table, k, fnx, ixn):
    nx=len(fnx)
    fcst_table[k:k+nx,:nx]+=np.r_[fnx,0][ixn]

def fcst_table_shp(fcst_table, lr):
    """
    fcst_table: shape nndays,n of forecasts
    lr: length nndays vector of return
    return:
       per bar forecast shp
    """
    nndays,n=fcst_table.shape
    ndays=nndays//n
    shpd = (fcst_table.T*lr).T.reshape((ndays,n,n))
    shp = np.mean(shpd,axis=0)/np.std(shpd,axis=0)

def test_speed() :
    #########################################################
    # reference function, use faster version with ixn 
    def upd_fcst_fnx_ref(fcst_table, k, fnx):
        nx = len(fnx)
        n2,n = fcst_table.shape
        fcst_table[np.arange(nx).astype(int)+k, np.arange(nx).astype(int)] = fnx

    def upd_fcst_fnx_days_ref(fcst_table, fnx_day):
        nndays,nx = fnx_day.shape
        n2,n=fcst_table.shape
        nxix = np.arange(nx).astype(int)
        for nd in np.arange(nndays).astype(int):
            fcst_table[nxix+nd, nxix] = fnx_day[nd,:]
    ###########################################################

    fnx = np.arange((276*276*2)).reshape((276*2,276))
    fcst_table = np.zeros((276*2,276))
    fcst_table2 = np.zeros((276*2,276))
    ixn = get_ixn(276,276)

    import time
    t0 = time.time()
    for k in np.arange(10000) :
        k0 = int(k%276)
        k1 = k0+276
        fnx0 = fnx[k0:k1,:]
        upd_fcst_fnx_days_ref(fcst_table, fnx0)

    t1 = time.time()
    for k in np.arange(10000) :
        k0 = int(k%276)
        k1 = k0+276
        fnx0 = fnx[k0:k1,:]
        upd_fcst_days(fcst_table2, fnx0, ixn)

    t2 = time.time()

    assert np.max(np.abs(fcst_table-fcst_table2)) == 0
    print (t1-t0, t2-t1)
    return fcst_table, fcst_table2
