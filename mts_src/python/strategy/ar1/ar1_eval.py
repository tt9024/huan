import numpy as np
import mts_util
import dill
import Pop
import ar1_model
import ar1_sim
import datetime
import yaml
import copy
import Outliers as OT
import Pop3

def NullLogger(log_str):
    pass

########################
##  f,v,shp
########################

def get_lrf_nf(lr0, n, ixf, stdv=None) :
    """
    extend lr to space of ixf, so that lrf * f has form for sharp
    lr0: length nndays vector of log return
    return lrf, shape (nndays, nf)
    """
    nndays = len(lr0)
    nf = len(ixf)

    # use a smoothed lr
    lrd = lr0.reshape((nndays//n,n))
    if stdv is None :
        stdv = np.std(lrd,axis=0)  # std per each bar
    lr = OT.soft1(lrd,stdv,3,1).flatten()

    # construct a lr table to mul with f
    lr0 = np.r_[lr, np.zeros(ixf[-1]+1)]
    lrc=np.cumsum(lr0)
    ixc = np.r_[0,ixf+1]
    lrf = []
    lrix = np.arange(nndays).astype(int)

    for ix0, ix1 in zip(ixc[:-1],ixc[1:]):
        lrf.append(lrc[lrix+ix1]-lrc[lrix+ix0])

    return np.array(lrf).T

def get_sharp(lr0, f, n, ixf, stdv=None) :
    return get_lrf_nf(lr0, n, ixf, stdv=stdv) * f

def get_sharp_days(lr0, f, n, ixf, sday_ix=None, eday_ix=None, stdv=None):
    """
    shp = get_sharp_days(lr0, f, n, ixf, sday_ix=None, eday_ix=None, stdv=None)
        get a n-by-nf shp for given days, with outlier/infs removed
    input: 
        lr0: length nndays vector
        f:   shape nndays, nf 
        stdv: length n vector of stdv
    return:
        shp: n-by-nf sharp of f
    """
    nndays,nf=f.shape
    assert len(lr0)>=nndays
    assert len(ixf) == nf
    assert ixf[0] == 0
    assert ixf[-1]+1 <= n
    ndays = nndays//n
    if eday_ix is None:
        eday_ix = ndays
    if sday_ix is None:
        sday_ix = 0
    if eday_ix < 0:
        eday_ix +=ndays
    if sday_ix < 0:
        sday_ix +=ndays
    assert sday_ix < eday_ix and eday_ix <= ndays and sday_ix <= ndays

    shp=get_sharp(lr0, f, n, ixf, stdv=stdv).reshape((ndays,n,nf))[sday_ix:eday_ix,:,:]
    shp=np.mean(shp,axis=0)/np.std(shp,axis=0)
    # remove nan or inf
    ix = np.nonzero(np.isnan(shp))
    if len(ix[0]) > 0:
        print ('removing ', len(ix[0]), 'nans from shp')
        shp[ix]=0
    ix = np.nonzero(np.isinf(shp))
    if len(ix[0]) > 0:
        print ('removing ', len(ix[0]), 'inf from shp')
        shp[ix]=0
    return shp


def get_stdvh_days(model_fn, md_days, ni_ix=0, fm_in=None):
    if fm_in is None:
        fm = dill.load(open(model_fn, 'rb'))['fm']
    else :
        fm = copy.deepcopy(fm_in)
    symbols = list(md_days.keys())
    assert len(symbols) > 0, "empty md_days"
    ndays, n, l = md_days[symbols[0]].shape
    assert n == fm.fcst_obj.n, "%s dimention mismatch with fcst object!"%(symbols[0])
    assert l >= 4, "%s bar dimension less than 4!"%(symbols[0])
    for sym in symbols[1:] :
        ndays0, n0, l0 = md_days[sym].shape
        assert ndays0==ndays and n0 == n and l0 == l, "%s shape mismatch for md_days"%(sym)

    fm._init()  # this initializes the fcst_obj's stlr and input object
    stdvh=[fm.state_obj.std.v[ni_ix,:].copy()] # ni_ix, trading symbol's md idx
    for d in np.arange(ndays).astype(int) :
        # run for the day d
        for k in np.arange(n).astype(int) :
            # run for the day
            md_dict = {}
            for sym in md_days.keys() :
                md_dict[sym] = md_days[sym][d,k,:]
            fm.on_bar(k,md_dict)
        fm.on_eod()
        stdvh.append(fm.state_obj.std.v[ni_ix,:].copy())
    return np.array(stdvh)

def get_v_from_stdvh(stdvh, ixf):
    """
    stdvh: shape(ndays,n) from tshp_dict (ref ni_pick.py)
           note the first stdv is used with first f, but 
           last stdv is the future std est to 
           be used by future forecasts
    """
    import scipy.linalg as alg
    ndays,n=stdvh.shape 
    nx=ixf[-1]+1
    vd=stdvh**2
    vdh=[]
    for vd0 in vd:
        vdh.append(alg.hankel(np.r_[vd0[1:],vd0[0]],vd0))

    # get from stdv (nx) to v with ixf (nf)
    nfnx=Pop3.NfNx(n,ixf)
    return nfnx.fnf_days(np.vstack(vdh)[:,:nx])

def get_tcst_day_from_vector(tscale_vector,ixf):
    """
    tscale_vector: length n vector of tscale at each bar k=0,..,n-1
    return:
      tscale_day: 2D array shape(n,nf+1), where [k,:] to be used for pop at k
    """
    n=len(tscale_vector)
    nx = ixf[-1]+1
    assert ixf[0]==0
    import scipy.linalg as alg
    tcd = alg.hankel(tscale_vector, np.r_[0,tscale_vector])
    nfnx=Pop3.NfNx(n+1,np.r_[ixf,nx])
    return nfnx.fnf_days(tcd[:,:nx+1])/np.r_[ixf-np.r_[-1,ixf[:-1]],1]

def get_stlrhh_days(fm, md_days) :
    stlrh0 = fm.fcst_obj.stlrh.copy() # all the statelr before the first day
    stdvh0 = fm.state_obj.std.v.copy()  # the stdv to be used for the first day

    stlrhh = []  # daily stlr
    stdvhh = []   # daily stdv

    # validate the md_days
    symbols = list(md_days.keys())
    assert len(symbols) > 0, "empty md_days"
    ndays, n, l = md_days[symbols[0]].shape
    assert n == fm.fcst_obj.n, "%s dimention mismatch with fcst object!"%(symbols[0])
    assert l >= 4, "%s bar dimension less than 4!"%(symbols[0])
    for sym in symbols[1:] :
        ndays0, n0, l0 = md_days[sym].shape
        assert ndays0==ndays and n0 == n and l0 == l, "%s shape mismatch for md_days"%(sym)

    fm._init()  # this initializes the fcst_obj's stlr and input object

    for d in np.arange(ndays).astype(int) :

        # run for the day d
        for k in np.arange(n).astype(int) :
            # run for the day
            md_dict = {}
            for sym in md_days.keys() :
                md_dict[sym] = md_days[sym][d,k,:]
            fm.on_bar(k,md_dict)
        fm.on_eod()
        stlrhh.append( fm.fcst_obj.stlrh[:,-n:].copy() )
        stdvhh.append( fm.state_obj.std.v.copy() )

    return stlrh0, stdvh0, stlrhh, stdvhh

def get_fni_stlrhh(c, stlrh0, stdvh0, stlrhh, stdvhh, ot_ym, ot_scl) :
    stlrh = np.hstack((stlrh0, np.hstack((stlrhh))))
    stdv = stdvh0

    days = len(stlrhh)
    ni, n = stlrhh[0].shape
    nh = stlrh0.shape[1]

    fni = []
    for ni0 in np.arange(ni) :
        fni.append([])

    ix0 = n
    ix1 = nh+n
    ixl = np.arange(-nh,0).astype(int)
    ixs = np.mod(ixl, n).astype(int)
    stdv = stdvh0[:, ixs]

    for d in np.arange(days) :
        stlr = stlrh[:,ix0:ix1] # one day worth of stlrh
        lrch = np.ravel(OT.soft1(stlr,stdv, ot_ym, ot_scl),order='F')

        nix = np.arange(nh)*ni
        for ni0 in np.arange(ni) :
            fni[ni0].append(np.dot(c[:,:,nix+ni0],lrch[nix+ni0]))
        ix0+=n
        ix1+=n
        stdv=stdvhh[d][:,ixs]

    for ni0 in np.arange(ni) :
        fni[ni0] = np.vstack((fni[ni0]))

    return np.array((fni))

def get_shp_fni(lr, fni, ixf, n, stdv=None) :
    shp = []
    ni, nndays, nf = fni.shape
    assert len(ixf) == nf
    ndays = nndays//n
    lrf=get_lrf_nf(lr,n,ixf,stdv=stdv)
    shp=np.tile(lrf,(ni,1,1))*fni
    return shp.reshape((ni,ndays,n,nf)), lrf
    """
    # slower but cleaner:
    for f in fni :
        shp.append(get_sharp(lr,f,n,ixf,stdv=stdv).reshape((ndays, n, nf)))
    return np.array(shp)
    """

def get_fni(fm, md_days, symbol='CL') :
    c = fm.fcst_obj.c
    n,nf,nh=c.shape
    lr = md_days[symbol][:,:,0].flatten()
    ixf = fm.fcst_obj.ixf
    ot_ym = fm.fcst_obj.ot_ym
    ot_scl = fm.fcst_obj.ot_scl

    stlrh0, stdvh0, stlrhh, stdvhh = get_stlrhh_days(fm, md_days)
    fni = get_fni_stlrhh(c, stlrh0, stdvh0, stlrhh, stdvhh, ot_ym, ot_scl)
    shp_ni, lrf = get_shp_fni(lr, fni, ixf, n)

    return fni, shp_ni, lrf

def plot_fni_shp(shp_ni, sday_ix, eday_ix, fig, vmin=-1e-3, vmax=1e-3, ixf=None) :
    """
    shp_ni: shape of (ni, ndays, n, nf) of f dot v, returned by get_shp_fni
    plot the n-by-nf shp graph for each ni from sday to eday
    """
    ni, ndays, n, nf = shp_ni.shape
    ax_arr=[]
    for ni0 in np.arange(ni) :
        ax_arr.append(fig.add_subplot(1,ni,ni0+1))

    for ni0, ax in zip(np.arange(ni), ax_arr) :
        v0 = shp_ni[ni0,sday_ix:eday_ix,:,:]
        shp0 = np.mean(v0,axis=0)/np.std(v0,axis=0)
        if ixf is not None:
            ixd = ixf-np.r_[-1,ixf[:-1]]
            shp0 /= np.sqrt(ixd)
        ax.pcolor(shp0, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(str(np.mean(shp0)))

def plot_fni_f(fni, n, day_ix, fig, md=None, vmin=-1e-4, vmax=1e-4) :
    """
    fni: shape (ni, nndays, nf)
    day_ix: index into ndays
    fig: the figure to be plotted on
    md: if not none, shape [ndays, n, >5], where last index has [lr,vol,vbs,lpx,utc], 
        i.e. md_dict['CL']
    plot the n-by-nf forecast for each ni on day 'day_ix'
    """
    if len(fni.shape) == 2 :
        nndays, nf = fni.shape
        fni = fni.reshape((1,nndays, nf))
    ni, nndays, nf = fni.shape
    assert nndays//n*n == nndays
    ax_arr=[]
    axcnt = ni+1
    if md is not None :
        axcnt+= 1
    ax_arr.append(fig.add_subplot(axcnt,1,1))
    for ni0 in np.arange(axcnt-1)+1 :
        ax_arr.append(fig.add_subplot(axcnt,1,ni0+1,sharex=ax_arr[0]))
    
    if day_ix < 0:
        day_ix += (nndays//n)

    if md is not None :
        ax_arr[-1].plot(md[day_ix,:,-2].flatten())

    f0 = np.zeros((nf,n))
    for ni0, ax in zip(np.arange(ni), ax_arr[:-2]) :
        v0 = fni[ni0, day_ix*n:(day_ix+1)*n, :].T
        ax.pcolor(v0, cmap='RdBu', vmin=vmin, vmax=vmax)
        f0 += v0

    # the model (sum of fni)
    ax_arr[-2].pcolor(f0, cmap='RdBu', vmin=vmin, vmax=vmax)

def plot_coef(fm, nix_arr, fig, vmin=-1e-6, vmax=1e-6) :
    """
    for each of the n-by-nh coef graph for each ni at ixf=0
    each plotted coef value is normalized w.r.t. the input magnitude.
    i.e. the coefs are expected to apply to inputs with std=1
    """
    c = fm.fcst_obj.c
    n, nf, ninh=c.shape
    sd = np.std(fm.fcst_obj.stlrh,axis=1)
    ni = len(sd)
    nh=ninh//ni

    assert ninh == ni*nh

    ax_arr=[]
    for i, ni0 in enumerate(nix_arr) :
        ax_arr.append(fig.add_subplot(len(nix_arr),1,i+1))

    ix0 = np.arange(nh)*ni
    for ni0, ax in zip(nix_arr, ax_arr) :
        c0 = c[:,0,ix0+ni0]
        ax.pcolor(c0*sd[ni0], cmap='RdBu', vmin=vmin, vmax=vmax)


########################
# fcst_table with Pop3
########################
def fcst_table_shp(fni_arr, v, n, ixf, lr, fni_arr_in=None):
    """
    ft, tshp, fnx, tshp_ni = fcst_table_shp(fnf_ni_arr, v, n, ixf, lr, fni_arr_in)
    return:
        ft: the Pop3 object after apply fnf, which is sum of fni
        tshp: the table shp of fnf (not fnf_ni)
        fnx: the bar-by-bar forecast of fnf (not fnf_ni)
        tshp_ni: the per-ni table shp
    """
    tshp_ni = []
    ni, nndays, nf = fni_arr.shape
    assert nf==len(ixf)
    assert v.shape == (nndays,nf)
    assert len(lr) == nndays
    ndays=nndays//n
    if fni_arr_in is None:
        fni_arr_in = [None]*ni
        f_in = None
    else :
        f_in = np.sum(fni_arr_in,axis=0)

    for fni, fni_in in zip(fni_arr,fni_arr_in):
        _,tshp0,_,_=Pop3.run_days(fni,v,n,ixf,ixf,lr=lr, f_in=fni_in)
        #tshp0 shape [ndays,n,n]
        tshp_ni.append(tshp0) 
    tshp_ni=np.array(tshp_ni) # shape [ni,ndays,n,n] 
    ft,tshp,fnf,fnx=Pop3.run_days(np.sum(fni_arr,axis=0),v,n,ixf,ixf,lr=lr,f_in=f_in)
    return ft, tshp, fnf, tshp_ni

def tshp_days(tshp, tshp_ni, sday_ix=0, eday_ix=-1):
    ndays,n,n=tshp.shape
    if sday_ix < 0:
        sday_ix+=ndays
    if eday_ix < 0:
        eday_ix+=ndays
    eday_ix+=1
    assert sday_ix <= ndays and eday_ix <= ndays
    tshp_d=tshp[sday_ix:eday_ix,:,:]
    tshp_d=np.mean(tshp_d,axis=0)/np.std(tshp_d,axis=-0)
    tshp_ni_d=tshp_ni[:,sday_ix:eday_ix,:,:]
    tshp_ni_d=np.mean(tshp_ni_d,axis=1)/np.std(tshp_ni_d,axis=1)
    return tshp_d, tshp_ni_d

def plot_tshp_dict(tshp_dict, model_idx, fig, sday_ix, eday_ix) :
    tshp_d = tshp_dict['tshp'][model_idx][sday_ix:eday_ix,:,:]
    tshp_d = np.mean(tshp_d,axis=0)/np.std(tshp_d,axis=0)
    tshp_ni_d = tshp_dict['tshp_ni'][model_idx][:,sday_ix:eday_ix,:,:]
    tshp_ni_d = np.mean(tshp_ni_d,axis=1)/np.std(tshp_ni_d,axis=1)

    model_name=tshp_dict['fn'][model_idx].split('/')[-1].split('.')[0]
    ind_spec = tshp_dict['spec']

    ni,n,_ = tshp_ni_d.shape
    assert tshp_d.shape == (n,n)

    for i, ts in enumerate(tshp_ni_d):
        ax=fig.add_subplot(1,ni+2,i+1)
        ax.pcolor(ts, cmap='RdBu', vmin=-1e-1, vmax=1e-1)
        if ind_spec is not None:
            tst = str(ind_spec[i%(ni//2)])
            if i>=ni//2:
                tst+='_mul'
            ax.set_title(tst)

    ax=fig.add_subplot(1,ni+2,ni+1)
    ax.pcolor(tshp_d, cmap='RdBu', vmin=-1e-1, vmax=1e-1)
    ax.set_title(model_name+' model')
    # get a max
    tshp_max=np.max(np.vstack((tshp_ni_d, tshp_d.reshape((1,n,n)))), axis=0)
    ax=fig.add_subplot(1,ni+2,ni+2)
    ax.pcolor(tshp_max,cmap='RdBu', vmin=-1e-1, vmax=1e-1)
    ax.set_title(model_name+' best')

def pop3_p0_pnl(tshp_dict, model_idx, ni_idx, sday_ix, eday_ix, md_dict_symbol, ixf3=None, fscale=None, tbl_shp=None,picks=None,fnf3_scale=1):
    """
    challenge the fnf from tshp_dict with Pop3, using a particular model and ni (model_idx and ni_idx(-1 for sum of all ni))
    * a tbl_shp filtering
    * a different ixf3

    fnf3_scale approximately scales the fnf from pop3, accounting for reduced 
         std from averaging

    return the result p0 and pnl using a basic pop execution
        baseline, test, ft
    where baseline and test are dicts with format of
        {\
         'shp_tbl':       sharp of the forecast table
         'fnf':           the forecast in ixf
         'shp_fnf':       sharp of forecast in ixf
         'pp':            position from a basic execution
         'pnl':           pnl from pp
         'shp_pnl_bar':   sharp of per-bar pnl
         'shp_pnl_day':   sharp of daily pnl
        }
    """

    # get the sharp table
    tshp_ni = tshp_dict['tshp_ni'][model_idx]
    ni,ndays,n,_ = tshp_ni.shape
    sday_ix = (sday_ix+ndays)%ndays
    eday_ix = (eday_ix+ndays)%ndays
    assert ni_idx < ni
    if ni_idx >0:
        tshp = tshp_ni[ni_idx,:,:,:]
        fnf = tshp_dict['fni'][model_idx][ni_idx,sday_ix*n:eday_ix*n,:]
    else :
        print('ni_idx negative, getting model shp')
        tshp = tshp_dict['tshp'][model_idx]
        fnf = np.sum(tshp_dict['fni'][model_idx][:,sday_ix*n:eday_ix*n,:],axis=0)

    ixf=tshp_dict['ixf'][model_idx]
    lr = md_dict_symbol[sday_ix:eday_ix,:,0].flatten()
    lpx= md_dict_symbol[sday_ix:eday_ix,:,3].flatten()

    # ignore the first day for comparing with Pop3's shp_tbl
    shp_tbl = np.mean(tshp[sday_ix+1:eday_ix,:,:],axis=0)/np.std(tshp[sday_ix:eday_ix,:,:],axis=0)
    shp_tbl[np.isnan(shp_tbl)]=0
    shp_tbl[np.isinf(shp_tbl)]=0
    shp_fnf = get_sharp_days(lr, fnf, n, ixf)

    stdvh = tshp_dict['stdvh'][model_idx][sday_ix:eday_ix,:]
    print('getting existing pp0/pnl')
    pp0, pnl0 = p0_from_fnx2(fnf, \
                 stdvh, \
                 #tshp_dict['stdvh'][model_idx], \
                 ixf, lpx, tcost=0.05, maxpos=100, \
                 fscale=None, vscale=(1,0,6), \
                 wpos=0, min_trade=2, p0=0, pnl_tcost=0.025, contract_size=1000, \
                 k0_k1_list=[], no_trade_k0_k1=None)

    pnl0 = pnl0.reshape((eday_ix-sday_ix,n))
    shp_pnl0_bar = np.mean(pnl0,axis=0)/np.std(pnl0,axis=0)
    shp_pnl0_bar[np.isinf(shp_pnl0_bar)]=0
    shp_pnl0_day = np.mean(np.sum(pnl0,axis=1))/np.std(np.sum(pnl0,axis=1))

    ft=None
    if ixf3 is not None or tbl_shp is not None:
        # getting a f from Pop3
        # need v from stdvh
        ixf_in = ixf
        if ixf3 is None:
            ixf3 = ixf

        # use stdvh from the model, needed if no vnf
        #stdvh = tshp_dict['stdvh'][model_idx][sday_ix:eday_ix-1,:] #stdvh has one more ix
        v, _,_,_ = gen_daily_vvtw(ixf_in,stdvh,(1,0,1), 0, [])
        ft, shp_tbl3, fnf3,_ = Pop3.run_days(fnf,v,n,ixf_in,ixf3,fscale=fscale, tbl_shp=tbl_shp, lr=lr)
        # shp_tbl3 shape [ndays, n, n], ignore the first day, for the Pop3's fcst table
        shp_tbl3 = np.mean(shp_tbl3[1:,:,:],axis=0)/np.std(shp_tbl3[1:,:,:],axis=0)
        shp_tbl3[np.isnan(shp_tbl3)]=0
        shp_tbl3[np.isinf(shp_tbl3)]=0

        # fnf is the Pop3 forecast, with the fscale and tbl_shp. 
        # if picks is set, then overwrite this fnf using the average of most recent picks from fcst_table
        # pick doesn't seem to work, as the recent prediction is usually weak
        if picks is not None:
            fni_pick = ft.fnx_hist_latest(np.arange((eday_ix-sday_ix)*n).astype(int), nx=ixf3[-1]+1, picks=picks)
            nfnx=Pop3.NfNx(n, ixf3)
            fnf3=nfnx.fnf_days(np.vstack(fni_pick))

        shp_fnf3 = get_sharp_days(lr, fnf3, n, ixf3)
        print('getting Pop3 pp0/pnl')
        # fscale in nx, due to avg of multiple f, reducing std
        # fnf3_scale, a scalar or a array length nf_out
        fnf3 *= fnf3_scale # scale the nf_out w.r.t. target risk
        pp3, pnl3 = p0_from_fnx2(fnf3, \
                     stdvh, \
                     #tshp_dict['stdvh'][model_idx], \
                     ixf3, lpx, tcost=0.05, maxpos=100, \
                     fscale=None, vscale=(1,0,6), \
                     wpos=0, min_trade=2, p0=0, pnl_tcost=0.025, contract_size=1000, \
                     k0_k1_list=[], no_trade_k0_k1=None)

        pnl3 = pnl3.reshape((eday_ix-sday_ix,n))
        shp_pnl3_bar = np.mean(pnl3,axis=0)/np.std(pnl3,axis=0)
        shp_pnl3_bar[np.isinf(shp_pnl3_bar)]=0
        shp_pnl3_day = np.mean(np.sum(pnl3,axis=1))/np.std(np.sum(pnl3,axis=1))

    else :
        print('Not Running Pop3!')
        ft, shp_tbl3, fnf3, shp_fnf3, pp3, pnl3, shp_pnl3_bar, shp_pnl3_day = [None]*8

    baseline = {'shp_tbl':shp_tbl, 'fnf':fnf, 'shp_fnf':shp_fnf, \
                'pp':pp0, 'pnl':pnl0, 'shp_pnl_bar':shp_pnl0_bar, 'shp_pnl_day':shp_pnl0_day}
    test =     {'shp_tbl':shp_tbl3, 'fnf':fnf3, 'shp_fnf':shp_fnf3, \
                'pp':pp3, 'pnl':pnl3, 'shp_pnl_bar':shp_pnl3_bar, 'shp_pnl_day':shp_pnl3_day}
    return  baseline, test, ft


def fscale_tblshp_from_tshp_dict(tshp_dict, model_idx, ni_list, shp_thres_list = None, shp_scale_list = None, sday_idx = 0, eday_idx = -1, shp_scale_func_list=None):
    """
    fni_nfs, tbl_shp_ni = fscale_tblshp_from_tshp_dict(tshp_dict, model_idx, ni_list, shp_thres_list = None, shp_scale_list = None, sday_idx = 0, eday_idx = -1)
    input:
       shp_thres_list: length nx list of shp_thres to be clipped from, i.e.
                       x = clip(np.sign(x-thres), 0, 1) * x
                       default to be 0
       shp_scale_list: list of shp_scale to be multiplied onto
                       x = x*shp_scale
                       default to be 1 (ignored if shp_scale_func provided)
       sday_idx and eday_idx are inclusive
       shp_scale_func_list: a list of functions to apply the tble_shp_ni
                        shp = func_i(shp0), where shp0 shape (n,n)
                        if not None, shp_scale_list ignored
    return:
        fni_nfs: list of running scaled fni_nf, according to the running vnf,
                 so that std(fni_nfs) = 100/3*vnf, i.e. 
                 the 3*std(fni_nfs/vnf) is approximately 100
        tbl_shp_ni: list of tbl_shp, shape n-by-n to be used in ni_dict
    """
    nx = len(ni_list)
    if shp_thres_list is not None:
        assert len(shp_thres_list) == nx
    else :
        shp_thres_list = [0.0]*nx
    shp_thres = np.array(shp_thres_list)
    if shp_scale_list is not None:
        assert len(shp_scale_list) == nx
    else :
        shp_scale_list = [1.0]*nx
    shp_scale = np.array(shp_scale_list)

    if shp_scale_func_list is not None:
        assert len(shp_scale_func_list) == nx

    fni = tshp_dict['fni'][model_idx]
    fni = np.vstack((fni, [np.sum(fni,axis=0)])) # allow -1 to be model fnf
    vnf = tshp_dict['vnf'][model_idx]
    tni = np.vstack((tshp_dict['tshp_ni'][model_idx], [tshp_dict['tshp'][model_idx]]))
    ni,nndays,nf=fni.shape
    ni,ndays,n,n=tni.shape
    assert vnf.shape == (nndays, nf)
    assert nndays == n*ndays
    assert abs(sday_idx) <= ndays and abs(eday_idx) <= ndays
    if sday_idx < 0:
        sday_idx += ndays
    if eday_idx < 0:
        eday_idx += ndays
    nd=eday_idx-sday_idx
    assert nd>=20

    ni_list = np.array(ni_list).astype(int)
    # getting the fnfs
    fnfl=[]
    fni = fni[ni_list,:,:].reshape((nx,ndays,n,nf))
    vnf = vnf.reshape((ndays,n,nf))
    for d in np.arange(sday_idx, eday_idx).astype(int):
        v = vnf[max(d-1,0)]*100.0/3.0  # make f/v roughly less than 100
        fstd = np.std(fni[:,:max(50,d),:,:],axis=1)  # trailing fni for fstd
        fnid = fni[:,d,:,:]/fstd*v # fscale is approximately v/fstd
        fnid[np.isnan(fnid)]=0
        fnid[np.isinf(fnid)]=0
        fnfl.append(fnid.reshape((nx,n*nf)))
    fnfl=np.hstack((fnfl)).reshape((nx,nd,n,nf))

    # getting the tbl_shp for each nx
    tni = tni[ni_list,sday_idx:eday_idx,:,:]
    tni = np.mean(tni,axis=1)/np.std(tni,axis=1) #median, same with ni_pick
    tni[np.isnan(tni)]=0
    tni[np.isinf(tni)]=0
    # tni shape (nx,n,n), shp_thres shape (nx)
    tni*=(np.clip(np.sign(tni.T-shp_thres),0,1).T)

    if shp_scale_func_list is not None:
        for nxi in np.arange(nx).astype(int):
            tni[nxi,:,:] = shp_scale_func_list[nxi](tni[nxi,:,:])
    else :
        tni=(tni.T*shp_scale).T

    return fnfl, tni

def tshp_dict_run_days_ni(tshp_dict, n, model_ni_dict, ixf_out,\
        sday_idx, eday_idx, md_dict_symbol) :
    """
    run tshp_dict with chosen model_ni_idx, with given shp_thresholds from
    ni_pick.
    input:
        model_ni_dict: { model_idx: { ni_idx: {'shp_thres', 'shp_scale'} } }

        where the shp_thres is the threshold that a signal in fcst_table is included
                  shp_scale is the scale to apply to the signal's original sharp

    return result of run days
        ft, fcst_table_shp, fnf, fnx 
    """
    # get the fnf, tni, stdvh, ixf to be ready to run_days_ni()
    ni,ndays,n0,_=tshp_dict['tshp_ni'][0].shape
    assert n0 == n, 'n mismatch!'
    if sday_idx < 0 :
        sday_idx += ndays
    if eday_idx < 0:
        eday_idx += ndays
    fnfl = []
    tni = []
    ixf = []
    stdvh = []
    for model_ix in model_ni_dict.keys():
        nx,nndays,nf = tshp_dict['fni'][model_ix].shape
        ni_idx = []
        shp_thres_list = []
        shp_scale_list = []
        for nix in model_ni_dict[model_ix].keys():
            ni_idx.append(nix)
            shp_thres_list.append(model_ni_dict[model_ix][nix]['shp_thres'])
            shp_scale_list.append(model_ni_dict[model_ix][nix]['shp_scale'])

        fnf0, tni0 = fscale_tblshp_from_tshp_dict(tshp_dict, model_ix, ni_idx, shp_thres_list = shp_thres_list, shp_scale_list = shp_scale_list, sday_idx = sday_idx, eday_idx = eday_idx)
        nx0, ndays0, n0, nf0 = fnf0.shape
        # TODO - entertain different n here?
        # quick hack to fill smaller n upto 276
        assert n0 == n, 'n mismatch for model % ni %d'%(model_ix, nix)
        fnf0 = fnf0.reshape((nx0, ndays0*n0, nf0))
        for f0, t0  in zip(fnf0, tni0):
            fnfl.append(f0)  # fnfl
            tni.append(t0)
            ixf.append(tshp_dict['ixf'][model_ix])

        stdvh.append(tshp_dict['stdvh'][model_ix][sday_idx:eday_idx,:])

    # run_days_ni
    stdvh = np.mean(np.array(stdvh),axis=0)
    ft, fcst_table_shp, fnf, fnx = Pop3.run_days_ni(fnfl, stdvh, n, ixf, ixf_out, tbl_shp_ni = tni, lr = md_dict_symbol[sday_idx:eday_idx,:,0].flatten())
    return ft, fcst_table_shp, fnf, fnx

def pack_result_dict(tshp_dict, n, fnf3, tshp3, ixf3,  \
        md_dict_symbol, sday_idx, eday_idx, \
        fnf3_scale = None, \
        tcost = 0.05, maxpos = 100, vscale = (1,0,6), pnl_tcost = 0.025, \
        wpos = 0, min_trade=2, p0=0, fscale = None, contract_size = 1000, \
        k0_k1_list = [], no_trade_k0_k1=None):

    """
    run position and pnl with the given fnf3 and ixf3

    pop3_p0_pnl() :
        {\
         'shp_tbl':       sharp of the forecast table
         'fnf':           the forecast in ixf
         'shp_fnf':       sharp of forecast in ixf
         'pp':            position from a basic execution
         'pnl':           pnl from pp
         'shp_pnl_bar':   sharp of per-bar pnl
         'shp_pnl_day':   sharp of daily pnl
        }

    pp0, pnl0 = p0_from_fnx2(fnf, stdvh, ixf, lpx, tcost=0.05, maxpos=100, \
                 fscale=None, vscale=(1,0,6), \
                 wpos=0, min_trade=2, p0=0, pnl_tcost=0.025, contract_size=1000, \
                 k0_k1_list=[], no_trade_k0_k1=None)
    """
    tshp_ni = tshp_dict['tshp_ni'][0]
    ni,ndays,n0,_ = tshp_ni.shape
    assert n0 == n, 'n mismatch!'
    if sday_idx < 0 :
        sday_idx += ndays
    if eday_idx < 0:
        eday_idx += ndays
    nf = len(ixf3)
    if fnf3_scale is None:
        fnf3_scale = np.ones(nf)
    #assert len(fnf3_scale) == nf, 'fnf3_scale mismatch with nf'

    lr = md_dict_symbol[sday_idx:eday_idx,:,0].flatten()
    lpx= md_dict_symbol[sday_idx:eday_idx,:,3].flatten()

    shp_fnf3 = get_sharp_days(lr, fnf3, n, ixf3)
    shp_tbl3 = None
    if tshp3 is not None:
        shp_tbl3 = np.mean(tshp3, axis=0)/np.std(tshp3,axis=0)
        shp_tbl3[np.isnan(shp_tbl3)]=0
        shp_tbl3[np.isinf(shp_tbl3)]=0

    # generate an average of stdvh
    stdvh = []
    for td in tshp_dict['stdvh']:
        stdvh.append(td[sday_idx:eday_idx,:])
    stdvh = np.mean(stdvh,axis=0)

    print('getting Pop3 pp0/pnl')
    # fscale in nx, due to avg of multiple f, reducing std
    # fnf3_scale, a scalar or a array length nf_out
    fnf3 *= fnf3_scale # scale the nf_out w.r.t. target risk
    pp3, pnl3 = p0_from_fnx2(fnf3, stdvh, ixf3, lpx, \
            tcost=tcost, maxpos=maxpos, \
            fscale=fscale, vscale=vscale, \
            wpos=wpos, min_trade=min_trade, p0=p0, pnl_tcost=pnl_tcost, \
            contract_size=contract_size, k0_k1_list=k0_k1_list, \
            no_trade_k0_k1=no_trade_k0_k1)

    pnl3 = pnl3.reshape((eday_idx-sday_idx,n))
    shp_pnl3_bar = np.mean(pnl3,axis=0)/np.std(pnl3,axis=0)
    shp_pnl3_bar[np.isinf(shp_pnl3_bar)]=0
    shp_pnl3_day = np.mean(np.sum(pnl3,axis=1))/np.std(np.sum(pnl3,axis=1))
    test =     {'shp_tbl':shp_tbl3, 'fnf':fnf3, 'shp_fnf':shp_fnf3, \
                'pp':pp3, 'pnl':pnl3, 'shp_pnl_bar':shp_pnl3_bar, 'shp_pnl_day':shp_pnl3_day}
    return test

def plot_tshp_pnl(fig, baseline, test, pnl_test=None):
    """
    plotting a daily pnl versus the shp for shp_tbl or shp_fnf
    shp: shape of (n,nf), could be a shp_tbl, where nf=n
    pnl: vector of length (nndays)
    pnl_test: same length pnl of the same period to be compared with, usually 
              pnl from production execution
              if pn_test is 2D, then each row is a pnl_test
    """

    shp_b = baseline['shp_fnf']
    shp_t = test['shp_fnf']
    pnld_b = baseline['pnl']
    pnld_t = test['pnl']

    n,nf=shp_b.shape
    ndays,n=pnld_b.shape
    nndays=ndays*n
    ax1=fig.add_subplot(3,1,1)
    ax1.pcolor(shp_b.T,cmap='RdBu',vmin=-0.1,vmax=0.1)

    ax2=fig.add_subplot(3,1,2,sharex=ax1)
    ax2.plot(np.cumsum(np.mean(pnld_b,axis=0)),label='pnl_baseline')
    ax2.plot(np.cumsum(np.mean(pnld_t,axis=0)),label='pnl_3')

    if pnl_test is not None:
        if len(pnl_test.shape) == 1:
            pnl_test = pnl_test.reshape((1, nndays))
        for i, pt in enumerate(pnl_test):
            pnld_test = pt.reshape((nndays//n,n))
            ax2.plot(np.mean(pnld_test,axis=0),label='pnl_test_'+str(i+1))
    ax2.grid()
    ax2.legend(loc='best')
    ax3=fig.add_subplot(3,1,3,sharex=ax1)
    ax3.pcolor(shp_t.T, cmap='RdBu', vmin=-0.1,vmax=0.1)

    return [ax1,ax2,ax3]

def plot_pos_pnl_fnf(fig, baseline, test, tshp_dict, sday_idx, eday_idx):
    symbol = tshp_dict['symbol']
    md_dict = tshp_dict['md_dict'][symbol][sday_idx:eday_idx,:,:]
    lpx = md_dict[:,:,-2].flatten()
    dt = []
    for t0 in md_dict[:,:,-1].flatten():
        dt.append(datetime.datetime.fromtimestamp(t0))
    dt=np.array(dt)
    pp_b = baseline['pp'][:-1].flatten()
    pp_t = test['pp'][:-1].flatten()
    pnl_b = baseline['pnl'].flatten()
    pnl_t = test['pnl'].flatten()
    fnf_b = baseline['fnf'].T[::-1,:]
    fnf_t = test['fnf'].T[::-1,:]
    axlist = [ fig.add_subplot(5,1,1) ]
    for i in np.arange(4):
        axlist.append(fig.add_subplot(5,1,i+2,sharex=axlist[0]))

    axlist[0].plot(dt, lpx, '.-')
    axlist[1].plot(dt, np.cumsum(pnl_b), '.-', label='baseline')
    axlist[1].plot(dt, np.cumsum(pnl_t), '.-', label='test')
    axlist[2].plot(dt, pp_b, '.-', label='baseline')
    axlist[2].plot(dt, pp_t, '.-', label='test')

    for ax in axlist[:3]:
        ax.grid() ; ax.legend(loc='best')

    dtd=dt[1]-dt[0]
    dt=np.r_[dt,dt[-1]+dtd]
    dt-=dtd/2
    Y = np.arange(fnf_b.shape[0]+1)-0.5
    import matplotlib.dates as mdates
    X = mdates.date2num(dt)

    axlist[3].pcolormesh(X,Y,fnf_b, cmap='RdBu', vmin=-1e-3, vmax=1e-3)
    axlist[4].pcolormesh(X,Y,fnf_t, cmap='RdBu', vmin=-1e-3, vmax=1e-3)

    return axlist

def tile_t_v(t,v,v0=None):
    #     tt,vv = tile_t_v(t,v,v0=None)
    # used to get the levels of pos/pnl over time, 
    # for pretty plotting
    # at time [t_i,t_{i+1}], the value is v, adding the starting/ending values
    tt = np.r_[t[0],np.tile(t[1:],(2,1)).T.flatten()]
    vv = np.r_[np.tile(v[:-1],(2,1)).T.flatten(),v[-1]]
    if v0 is not None:
        # add v0
        tt=np.r_[t[0],tt]
        vv=np.r_[v0,vv]
    return tt,vv

def plot_f_lpx_p_pnl(p,pnl,md_dict,label='',axlist=None, fig=None, plot_lpx=False):
    # p: length [nndays+1] vector, with p[0] is the pp0, the position at open time of bar0
    # pnl: length [nndays] vector, pnl[0] is achieved by p[0] and bar0 return
    # md_dict: shape[nday, n, 5], last col is ['lr','vol','vbs','lpx','utc']
    nndays = len(pnl)
    assert len(p) == nndays+1
    utc = md_dict[:,:,-1].flatten()
    lpx = md_dict[:,:,-2].flatten()
    assert len(utc)==nndays 
    dt = []
    for t in utc:
        dt.append(datetime.datetime.fromtimestamp(t))
    dt=np.array(dt)

    if axlist is None:
        ax1=fig.add_subplot(3,1,1)
        ax2=fig.add_subplot(3,1,2,sharex=ax1)
        ax3=fig.add_subplot(3,1,3,sharex=ax1)
        axlist=[ax1,ax2,ax3]
    ax1,ax2,ax3=axlist

    # plot lpx on top
    if plot_lpx:
        ax1.plot(dt, lpx, '.-', label='lpx')

    # plot position
    tt,pp=tile_t_v(dt,p[1:],v0=p[0])
    ax2.plot(tt,pp,'.-', label=label)

    # plot the pnl
    tt,pp=tile_t_v(dt,pnl)
    ax3.plot(tt,np.cumsum(pp),'.-', label=label)
    return axlist



def coef_scale(coef,ixf,stdv,ni,ix):
    n,nf,ninh=coef.shape
    nh=ninh//ni
    coef_ix=np.sum(np.abs(coef[:,:,np.arange(nh).astype(int)*ni+ix]),axis=2)

    nfnx=Pop3.NfNx(n,ixf,stdv=stdv)
    fnx=nfnx.fnx_days(coef_ix) # straight out the ixf
    ixn=Pop3.get_ixn(n,n)
    fcst_table=np.zeros((2*n,n))
    Pop3.upd_fcst_days(fcst_table,fnx,ixn)
    return fcst_table[:n,:]+fcst_table[-n:,:]

def coef_scale2(fm, ix):
    """ This shows the distance between magnitude of coefs leading
    to each elements of fcst_table.  Used as kernel for sharp_table
    """
    coef = fm.fcst_obj.c
    stdv = fm.state_obj.std.v[0,:]
    sd = np.std(fm.fcst_obj.stlrh,axis=1)
    ixf = fm.fcst_obj.ixf
    ni = len(sd)
    n,nf,ninh=coef.shape
    nh=ninh//ni
    assert ix < ni
    ixl = np.arange(-nh,0).astype(int)
    ixs = np.mod(ixl, n).astype(int)
    cix = np.arange(nh).astype(int)*ni+ix
    coef0=coef[:,0,cix]*stdv[ixs]*sd[ix] #scale with stdv and sd
    shp_dist_horizontal = np.zeros((n,n))
    shp_dist_vertical = np.zeros((n,n))

    # horizontally, (i,j) applies distance between (i,j) and (i,j+1)
    # just the coef backwards
    for k in np.arange(n).astype(int):
        # the last dist shouldn't be used for smoothing
        # dist[0] is the distance between 0 and 1
        shp_dist_horizontal[k,:]=np.abs(coef0[k,nh-n-(n-k)+1:nh-(n-k)+1])[::-1]
    # vertically, (i,j) applies to distance between (i,j) and (i+1,j)
    # between two lines plus the
    for k in np.arange(n).astype(int):
        k1=int((k+1)%n)
        dkcs=np.cumsum(np.abs(coef0[k,:nh-n+k+1]-coef0[k1,:nh-n+k+1]))
        shp_dist_vertical[k,:]=(dkcs[-n:]+coef0[k1,k1])[::-1]
    return shp_dist_horizontal, shp_dist_vertical


###############################
## position/pnl from f,v, plotting
###############################

def plot_pnl_from_p0(ax_list, p00, p0, md, tcost, contract_size = 1000, plot_md=False, marker = '.-', legend='') :
    """
    ax_list is obtained by:
        fig=figure() ; ax1=fig.add_subplot(3,1,1) ; ax2=fig.add_subplot(3,1,2,sharex=ax1) ; ax3 = fig.add_subplot(3,1,3,sharex=ax1)
    p00, p0, tcost, fee, contract_size: same as pnl_from_p0()
    md: i.e. md_dict['CL'], shape (ndays, n, >5)
    """
    lpx = md[:,:,-2].flatten()
    dt = []
    for t in md[:,:,-1].flatten():
        dt.append(datetime.datetime.fromtimestamp(t))
    dt=np.array(dt)
    if plot_md:
        ax_list[0].plot(dt,lpx)

    ax_list[1].plot(dt,p0,marker,label=legend)
    pnl, pnl_cst = ar1_sim.pnl_from_p0(p00,p0,lpx,tcost,contract_size=contract_size)
    ax_list[2].plot(dt,np.cumsum(pnl_cst), marker, label=legend)
    if legend != '' :
        for ax in ax_list[1:] :
            ax.legend(loc='best')

def p0_from_fnx2_run(n,f,v,lpx,k_arr,tcost,maxpos,vscale_day,tscale_day,wt_day,min_trade=5,p0=0,logfunc=NullLogger, no_trade_k0_k1=None):
    pp = [p0]
    vv=v[0,:]

    if no_trade_k0_k1 is None:
        nt0, nt1 = (1,-1)
    else:
        nt0, nt1 = no_trade_k0_k1
    for k, f0, v0, lpx0 in zip(k_arr,f,v,lpx):
        p0=pp[-1]

        if k>=nt0 and k<nt1:
            logfunc('no trade time, set position to 0')
            p0_new = 0
        else :
            vs,maxv,expv=vscale_day[k]
            tc = tscale_day[k]*np.log(1+tcost/lpx0)

            # fix tc[-1] to be a fraction of tc[0]
            frac = 0.2
            tc[-1] = tc[0]*frac

            wt = wt_day[k]
            #if maxpos-np.abs(p0)>=min_trade:
            vv= v0*(vs+maxv*((np.clip(np.abs(p0)/maxpos,0,0.99))**expv))

            #debug
            #if k==274:
            #    import pdb
            #    pdb.set_trace()
            #debug

            pp0 = Pop.opt(p0,f0,vv,tc,wt)
            p0_new=pp0[0]
            if np.abs(p0_new)<min_trade:
                p0_new=0
            elif np.abs(p0_new - p0)<=min_trade:
                p0_new=p0
            elif np.abs(p0_new) > maxpos:
                p0_new = maxpos*np.sign(p0_new)
            #logfunc('pop:\nf:(%s)\np0_in:(%s)\ntgt_pos:(%s)\nvs(%s)\nv0(%s)\nf0/vv(%s)\ntc(%s)\nwt(%s)\np0_new(%f)'%\
            #        (str(f0), str(p0), str(pp0), str(vs),str(v0),str(f0/vv),str(tc), str(wt), p0_new))
            logfunc('pop:\nf:(%s)\np0_in:(%s)\ntgt_pos:(%s)\nvs(%s)\np0_new(%f)'%\
                    (str(f0), str(p0), str(pp0), str(vs),p0_new))
        pp.append(p0_new)
    pp=np.array(pp)
    return pp

def gen_daily_vvtw(ixf,stdvh,vscale,wpos,k0_k1_list):
    """
       v, vscale_day, tscale_day, wt_day = gen_daily_vvtw(ixf,stdvh,vscale,wpos,k0_k1_list)
       k0_k1_list: [[k0,k1],(vs,maxv,expv),tscale,wpos], for [k0:k1], k1 is exclusive
    return:
        v: shape[ndays,nf] v from stdvh, scaled by the 'vs' in k0_k1_list, 
           but NOT by the global vscale
        vscale_day: [ndays, 3], for each bar's (vs,maxv,expv), where, vs is always the
                vscale[0]
        tscale_day, [ndays, nf+1]
        w_day, list of ndays, each with np array of nf of w_pos
    """
    nf=len(ixf)
    stdvh_scl=stdvh.copy()
    ndays,n=stdvh_scl.shape
    tscale_k = np.ones(n)
    vscale_day = np.tile(vscale,(n,1))
    wpos_day = np.array([wpos]*n).astype(int)
    for kl in k0_k1_list:
        (k01,vs,tscl,wp) = kl
        vs,maxv,expv=vs
        k0,k1=k01
        stdvh_scl[:,k0:k1]*=np.sqrt(vs)
        tscale_k[k0:k1]*=tscl
        wpos_day[k0:k1]=wp

        # fill vscale_k - (vs,maxv,expv) - note vs already in v
        # only put in k0_k1's maxv/expv 
        vscale_day[k0:k1,1] = maxv
        vscale_day[k0:k1,2] = expv

    # create v, tcost and wt
    v = get_v_from_stdvh(stdvh_scl, ixf)
    tscale_day = get_tcst_day_from_vector(tscale_k,ixf)
    wt_day = []
    for wp in wpos_day:
        w0 = np.ones(nf)
        if wp is not None and wp > 0:
            w0 = np.r_[np.linspace(1,5,wp), np.linspace(5, 0.1, nf-wp)]
        w0/=np.sum(w0)
        wt_day.append(w0)

    return v, vscale_day, tscale_day, wt_day

def gen_daily_vvtw_no_v(n,ixf,vscale,wpos,k0_k1_list):
    """
    """
    nf=len(ixf)
    tscale_k = np.ones(n)
    vscale_day = np.tile(vscale,(n,1))
    wpos_day = np.array([wpos]*n).astype(int)
    for kl in k0_k1_list:
        (k01,vs,tscl,wp) = kl
        vs,maxv,expv=vs
        k0,k1=k01
        tscale_k[k0:k1]*=tscl
        wpos_day[k0:k1]=wp

        # fill vscale_k - (vs,maxv,expv) - note vs already in v
        # only put in k0_k1's maxv/expv 
        vscale_day[k0:k1,1] = maxv
        vscale_day[k0:k1,2] = expv

    # create v, tcost and wt
    tscale_day = get_tcst_day_from_vector(tscale_k,ixf)
    wt_day = []
    for wp in wpos_day:
        w0 = np.ones(nf)
        if wp is not None and wp > 0:
            w0 = np.r_[np.linspace(1,5,wp), np.linspace(5, 0.1, nf-wp)]
        w0/=np.sum(w0)
        wt_day.append(w0)

    return vscale_day, tscale_day, wt_day

def p0_from_fv_param(n, f, v, lpx, param_dict=None, p0=0, pnl_tcost = 0.025, contract_size=1000, no_trade_k0_k1=None):
    """
    param_dict: if not none, can be given as from 
                param_dict = ar1_model.read_ar1_config('/tmp/LiveHO-....cfg','CL')
                if None, then default parameters apply, see pop3_p0_pnl()
    """
    if param_dict is None:
        # create default parameters
        param_dict = {}
        param_dict['ixf'] = [0, 1, 2, 5, 9, 15, 25, 41, 67, 107, 172, 275]
        param_dict['tcost'] = 0.1
        param_dict['maxpos'] = 100
        param_dict['fscale'] = None
        param_dict['vscale'] = (1.0,0,6)
        param_dict['wpos'] = 0
        param_dict['min_trade'] = 2
        param_dict['k0_k1_list'] = []

    ixf = param_dict['ixf']
    tcost = param_dict['tcost']
    maxpos = param_dict['maxpos']
    fscale = param_dict['fscale']
    vscale = param_dict['vscale']
    wpos = param_dict['wpos']
    min_trade = param_dict['min_trade']
    k0_k1_list = param_dict['k0_k1_list']

    pp, pnl = p0_from_fnx2(f, None, ixf, lpx, tcost=tcost, maxpos=maxpos, \
                           fscale=fscale, vscale=vscale, wpos = wpos, min_trade=min_trade, \
                           p0=p0, pnl_tcost=pnl_tcost, contract_size=contract_size, \
                           k0_k1_list = k0_k1_list, no_trade_k0_k1=no_trade_k0_k1, 
                           v = v, n=n)
    return pp, pnl

def p0_from_fnx2(f, stdvh, ixf, lpx, tcost=0.06, maxpos=100, \
                 fscale=[1,1,1,1,2,2,2,2,2,3,3,3], vscale=(0.4,0,6), \
                 wpos=0, min_trade=5, p0=0, pnl_tcost=None, contract_size=None, \
                 k0_k1_list=[[(8,23),(1,0,6),2.0,5], [(72,93),(1,0,6),3.0,5], [(144,165),(0.3,0,6),.2, 0], [(228,256),(4,0,6),1,5], [(266,276), (0.5,0,6), 2.0,5]],\
                 no_trade_k0_k1=(243,247), v=None, n=None, lr0=0):
    """
    input: 
        stdvh: shape[ndays,n] of std 
        lpx:   shape nndays of close price

    Same with above, except 
        1. using k0_k1_list to scale v, i.e. generate v from stdvh based on k0_k1_list
        2. using k0_k1_list to scale tcost
        3. vscale adding the first parameter as master scale
    Format of k0_k1_list:
        [(k0,k1),(vscale_master, maxv, expv), tcst_scale,wpos]
        applies to [k0:k1], k1 is exclusive

        the final is given by 
        v = v0*(vs+maxv*((np.clip(np.abs(p0)/maxpos,0,0.99))**expv))
        

    Note about vs - vscale in k0_k1_list applies to the global 'vscale', i.e. real
           scale is the multply of these two during [k0,k1]. In implementation,
           the vscale of k0_k1_list will be put in the returned 'v', 
           and so when running the pop, and so only the 'vs' in global vscale,
           the 'vscale_day's vs applies.

    param_dict: if not none, can be given as from 
                param_dict = ar1_model.read_ar1_config('/tmp/LiveHO-....cfg','CL')
    """
    if stdvh is not None:
        _,n=stdvh.shape
    assert n is not None
    nndays,nf=f.shape
    if fscale is None:
        fscale = np.ones(nf)
    assert len(ixf)==nf
    assert len(lpx)==nndays
    assert len(fscale)==nf

    fs=f*np.array(fscale)
    if stdvh is not None:
        v, vscale_day, tscale_day, wt_day = gen_daily_vvtw(ixf,stdvh,vscale,wpos,k0_k1_list)
    else :
        assert v is not None
        vscale_day, tscale_day, wt_day = gen_daily_vvtw_no_v(n,ixf,vscale,wpos,k0_k1_list)

    k_arr=np.mod(np.arange(nndays).astype(int),n)
    pp=p0_from_fnx2_run(n,fs,v,lpx,k_arr,tcost,maxpos,vscale_day,tscale_day,\
            wt_day,min_trade=min_trade,p0=p0,no_trade_k0_k1=no_trade_k0_k1)
    pnl = None
    if pnl_tcost is not None and contract_size is not None:
        pnl_nocost, pnl = ar1_sim.pnl_from_p0(pp[0], pp[1:],lpx,pnl_tcost,contract_size=contract_size,lr0=lr0)

    return pp,pnl


##########################
## Market Data Plotting
##########################

def plot_md(md_dict, key, fig, dt=None, sdayix=None, edayix=None, plot_cum=False, title_str=None) :
    ndays, n, indcnt = md_dict[key].shape
    if dt is None :
        dt = []
        for t in md_dict[key][:,:,-1].flatten():
            dt.append(datetime.datetime.fromtimestamp(t))
        dt = np.array(dt)

    if sdayix is None :
        sdayix = 0
    if edayix is None :
        edayix = ndays

    # lpx
    ax1=fig.add_subplot(3,1,1)
    ax1.plot(dt[sdayix*n:edayix*n], md_dict[key][sdayix:edayix,:,-2].flatten(), label='last price')

    # vol
    ax2=fig.add_subplot(3,1,2,sharex=ax1)
    d = md_dict[key][sdayix:edayix,:,1].flatten()
    lbl = 'daily volume'
    if plot_cum:
        d = np.cumsum(d)
        lbl= 'cumulative volume'
    ax2.plot(dt[sdayix*n:edayix*n], d, label=lbl)

    # vbs
    ax3=fig.add_subplot(3,1,3,sharex=ax1)
    d = md_dict[key][sdayix:edayix,:,2].flatten()
    lbl = 'daily volume imbalance'
    if plot_cum:
        d = np.cumsum(d)
        lbl= 'cumulative volume imbalance'
    ax3.plot(dt[sdayix*n:edayix*n], d, label=lbl)

    for ax in [ax1, ax2, ax3]:
        ax.legend(loc='best')
        ax.grid()

    if title_str is None :
        title_str = '%s from %s to %s'%(key, dt[0].strftime('%Y-%m-%d'), dt[-1].strftime('%Y-%m-%d'))
    ax1.set_title(title_str)

    return ax1,ax2,ax3

def plot_md_vol(md_dict, key, fig, dt=None, sdayix=None, edayix=None, title_str=None) :
    ndays, n, indcnt = md_dict[key].shape
    if dt is None :
        dt = []
        for t in md_dict[key][:,:,-1].flatten():
            dt.append(datetime.datetime.fromtimestamp(t))
        dt = np.array(dt)

    if sdayix is None :
        sdayix = 0
    if edayix is None :
        edayix = ndays

    # lpx
    ax1=fig.add_subplot(3,1,1)
    ax1.plot(dt[sdayix*n:edayix*n], md_dict[key][sdayix:edayix,:,-2].flatten(), label='last price')

    # vol
    ax2=fig.add_subplot(3,1,2,sharex=ax1)
    d = md_dict[key][sdayix:edayix,:,1].flatten()
    lbl = 'daily volume'
    ax2.plot(dt[sdayix*n:edayix*n], d, label=lbl)

    # cum vol
    ax3=fig.add_subplot(3,1,3,sharex=ax1)
    d = np.cumsum(d)
    lbl= 'cumulative volume'
    ax3.plot(dt[sdayix*n:edayix*n], d, label=lbl)

    for ax in [ax1, ax2, ax3]:
        ax.legend(loc='best')
        ax.grid()

    if title_str is None :
        title_str = '%s from %s to %s'%(key, dt[0].strftime('%Y-%m-%d'), dt[-1].strftime('%Y-%m-%d'))
    ax1.set_title(title_str)

    return ax1,ax2,ax3


#################################
# old code, not used anymore
#################################
def p0_from_fnx(n, f, v, lpx, tcost, maxpos=100, vscale=(0,6), wpos=None, min_trade=5, p0=0, pnl_tcost=None, contract_size=None):
    """
    lpx: length nndays vector of last price, used for tcst and pnl
    tcost in price, i.e. 0.05 for crude
    maxpos in number of contracts
    vscale: (maxv, exponent), actual v is multiplied with (1+maxv*(p**exp))
            where p is current position in [0,1]
            maxv >=0, maximum multplier to v when position is 1,
                   the lower the less retrictive
            exponent >= 2, quadratic shape, the higher the less restrictive
    """
    nf = f.shape[1]
    w0 = np.ones(nf)
    if wpos is not None:
        w0 = np.r_[np.linspace(0.5,2,wpos), np.linspace(2, 0.1, nf-wpos)]
    w0/=np.sum(w0)

    t0 = np.ones(nf+1)
    pp = [p0]
    maxv, expv = vscale
    maxv0, expv0 = vscale
    for i, (f0, v0, lpx0) in enumerate(zip(f,v,lpx)):
        p0=pp[-1]
        if maxpos-np.abs(p0)>=min_trade:
            vv= v0*(1+maxv*((np.clip(np.abs(p0)/maxpos,0,0.95))**expv))
        tt = t0* np.log(1+tcost/lpx0)
        pp0 = Pop.opt(p0,f0,vv,tt,w0)
        p0_new=pp0[0]
        if np.abs(p0_new - p0)<=min_trade:
            p0_new=p0
        elif np.abs(p0_new) > maxpos:
            p0_new = maxpos*np.sign(p0_new)
        """
            maxv*=1.2
        elif np.abs(p0_new) < maxpos/2 and maxv>maxv0:
            maxv=np.clip(maxv/1.2,maxv0,1e+10)
        """
        pp.append(p0_new)

    pp=np.array(pp)
    pnl = None
    if pnl_tcost is not None and contract_size is not None:
        pnl_nocost, pnl = ar1_sim.pnl_from_p0(pp[0], pp[1:],lpx,pnl_tcost,contract_size=contract_size)

    return pp,pnl

####################################
### Legacy code for n4 execution
####################################
def plot_pnl_from_fv(f, v, param, md, tcost, farr=None, varr=None, contract_size=1000, fig=None, ax_list=None, legend='') :
    plot_md = True
    if ax_list is None :
        ax1=fig.add_subplot(3,1,1) ; ax2=fig.add_subplot(3,1,2,sharex=ax1) ; ax3 = fig.add_subplot(3,1,3,sharex=ax1)
        ax_list = [ax1,ax2,ax3]
        plot_md = False

    if farr is None :
        farr, varr, pop_obj = ar1_sim.farr_from_fv(f,v,param)

    p00, f, f_st, v_st, f_ss, v_ss = ar1_sim.p0_from_farr(f,v,farr,varr,param)
    plot_pnl_from_p0(ax_list, p00[0], p00[1:], md, tcost, contract_size=contract_size, plot_md=True, marker = '-', label=legend)
    for ax in ax_list:
        ax.grid()
    return ax_list, p00

default_param = \
    {'trd_day': '20211129', \
     'strat_code': '370', \
     'mpos': 100, \
     'min_trade': 33, \
     'trading_hour': [-6.0, 17.0], \
     'n': 276, \
     'barsec': 300, \
     'trigger_cnt': 1200, \

# for old get_pos2() 
     'trade_cost_scale': 1.8, \
     'max_pos_scale': 0.1, \
# for new get_pos3() 
     'tscale': 0.2, \
     'mpscale': 1, \
#
     'w_pos': 5, \
     'clp_ratio': 0.7, \
     'ixf': [  0,   1,   2,   5,   9,  15,  25,  41,  67, 107, 172, 275], \
     'ixf_st': [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], \
     'ixf_lt': [  0,   1,   2,   5,   9,  15,  25,  41,  67, 107, 172, 275], \
     'pick': [47, 113], \
     'execution':  [\
      {'k0': 6, 'k1': 11, 'fscale_st': 1.0, 'tscale_st': 1.0}, \
      {'k0': 12, 'k1': 66, 'fscale_st': 0.5, 'tscale_st': 0.15}, \
      {'k0': 67, 'k1': 100, 'fscale_st': 1.0, 'tscale_st': 1.0}, \
      {'k0': 114, 'k1': 132, 'fscale_st': 1.0, 'tscale_st': 0.5},\
      {'k0': 144, 'k1': 160, 'fscale_st': 1.0, 'tscale_st': 1.0},\
      {'k0': 174, 'k1': 209, 'fscale_st': 1.0, 'tscale_st': 0.9} \
      ],\
    }

def plot_model(model, md_dict, lpx_symbol='CL', pnl_tcost=0.015, nf_cnt = -1, ni_shp_threshold = -1e+10, plot_old_exec=False, fig=None, ax_list=None, legend='', param0=None, f0=None, v0=None, fni0=None, shp_ni0=None, farr0=None, varr0=None):

    """
    model: n4_rt = dill.load(open('python/strategy/ar1/n4_rt.dill','rb'));
    md_dict: dill.load(open(python/strategy/ar1/md_dict_20210719_20211231_all.dill','rb'))
             md_dict['CL'] has shape [days, n, >5] where first 5 are [lr,vol,vbs,lpx,utc]
             days controls how many days to run simulation on
    lpx_symbol: the symbol in md_dict.keys() for pnl, i.e. 'CL'
    pnl_tcost:  Used for pnl_cst, usually 1.5 times spread, i.e. 0.015 for CL
                Note: 1.5 means for each trade, aggressive plus 1 spread of slippage on average
    nf_cnt:  number of ixf to use in pop. For example, nf_cnt=12 for nf=36 will use first 12 forecasts for pop
    plot_old_exec: if true, will generate farr/varr and use p0_from_farr to generate
                   p00 from farr and execution.
                   param needs to agree with nf of f
    if ax_list is None, then generate 3 ax from fig and plot the md
    ni_shp_threshold: the threshold for a ni to be picked into fni_s, setting it to infinity 
                      ensures no ni picked, and therefore position from fni not plotted
                      setting to -infinity includes all ni
                      setting to 0 includes only positive shp ni. 
                      the shp of ni is a mean for all day/bar/ixf, i.e. 
                      np.mean(shp_ni, axis=(1,2,3)), where shp_ni shape [ni,day,n,nf]

    param0: if None, use default_parameter.  Typically from ar1_model.read_ar1_config('python/strategy/ar1/param_nf24.cfg')
           change the execution parameters, i.e. tscale/mpscale.  
           param['ixf'] is set from model ixf ixf_st/ixf_lt can be set as desired, if not set, use default
    f0/v0/fni0/vni0/farr0/varr0: can be previous returned, to save the computation
    """
    mn = model['name']
    if legend != '' :
        mn = mn + '_' + legend

    if param0 is None:
        param0 = default_param
    param = copy.deepcopy(param0)
    ixf = copy.deepcopy(model['fm'].fcst_obj.ixf)
    if nf_cnt == -1 :
        nf_cnt = len(ixf)
    else :
        ixf = ixf[:nf_cnt]
    param['ixf'] = ixf
    for k, v in zip([ 'ixf_st', 'ixf_lt'], [np.arange(nf_cnt).astype(int), ixf]) :
        param[k] = v[:nf_cnt]

    plot_md = False
    if ax_list is None :
        ax1=fig.add_subplot(3,1,1) ; ax2=fig.add_subplot(3,1,2,sharex=ax1) ; ax3 = fig.add_subplot(3,1,3,sharex=ax1)
        ax_list = [ax1,ax2,ax3]
        plot_md = True

    # plot the position and 
    if f0 is None:
        fm = copy.deepcopy(model['fm'])
        f0,v0,stlrh = fm.run_days(md_dict)
    f = f0[:,:nf_cnt].copy()
    v = v0[:,:nf_cnt].copy()

    if fni0 is None :
        fm = copy.deepcopy(model['fm'])
        fni0, shp_ni0, lrf = get_fni(fm, md_dict)
    fni = fni0[:,:,:nf_cnt].copy()
    shp_ni = shp_ni0[:,:,:,:nf_cnt].copy()
    shpm = np.mean(shp_ni, axis=(1,2,3))
    ix = np.nonzero(shpm>=ni_shp_threshold)[0]
    if len(ix) == 0 :
        print('no ni has better shp than ', ni_shp_threshold)
    else :
        print('shpm ', shpm, ' picked ', ix)
        m = len(model['ind']['spec'])
        [print(model['ind']['spec'][i%m]) for i in ix]
        fni_s = np.sum(fni[ix,:,:],axis=0)
        p00 = ar1_model.get_pos3(fni_s, v, 0, param)
        plot_pnl_from_p0(ax_list, p00[0], p00[1:], md_dict[lpx_symbol], pnl_tcost, plot_md=plot_md, marker='.-', legend=mn)
        plot_md = False
        # f is modified to be fni_s here
        f = fni_s

    if plot_old_exec:
        if farr0 is None :
            farr0, varr0, pop2_obj = ar1_sim.farr_from_fv(f,v,param)
        p00, f, f_st, v_st, f_ss, v_ss = ar1_sim.p0_from_farr(f,v,farr0[:,:,:,:nf_cnt],varr0[:,:,:,:nf_cnt],param)
        plot_pnl_from_p0(ax_list, p00[0], p00[1:], md_dict[lpx_symbol], pnl_tcost, plot_md=plot_md, marker='-', legend=mn)

        ## try a run from modified f (first 4 ix using f_st) and entirely f_st?
        p00 = ar1_model.get_pos3(f, v, 0, param)
        plot_pnl_from_p0(ax_list, p00[0], p00[1:], md_dict[lpx_symbol], pnl_tcost, plot_md=plot_md, marker='-.', legend='fmod_'+mn)
        p00 = ar1_model.get_pos3(f_st, v, 0, param)
        plot_pnl_from_p0(ax_list, p00[0], p00[1:], md_dict[lpx_symbol], pnl_tcost, plot_md=plot_md, marker='-.', legend='fst_'+mn)

    else :
        farr = None; varr=None

    return ax_list, f0, v0, fni0, shp_ni0, farr0, varr0

def plot_model_days(model, md_dict, lpx_symbol='CL', pnl_tcost=0.015, nf_cnt = -1, ni_shp_threshold = -1e+10, plot_old_exec=False, fig=None, ax_list=None, legend='', param0=None, f0=None, v0=None, fni0=None, shp_ni0=None, farr0=None, varr0=None, sdix=0, edix=-1):
    """
    same as plot_model(), with choice of day.  
    Note: if shp_ni is given, the choice is still given based 
          on shp of original days, averaged over the given days range.

    sdix, edix: the index into the md_dict, f0/v0, etc day index. edix inclusive
    """
    ndays,n,ncol = md_dict[lpx_symbol].shape
    if sdix < 0:
        sdix += n
    if edix < 0 :
        edix += (n+1)
    md_dict0 = {}
    for k in md_dict.keys() :
        md_dict0[k] = md_dict[k][sdix:edix,:,:]
    k0 = sdix*n
    k1 = edix*n

    f = f0
    v = v0
    fni = fni0
    shp_ni = shp_ni0
    farr = farr0
    varr = varr0
    if f0 is not None:
        f = f0[k0:k1,:]
        v = v0[k0:k1,:]
    if fni is not None :
        fni = fni0[:,k0:k1,:]
        shp_ni = shp_ni0[:,sdix:edix,:,:]
    if farr is not None :
        farr = farr0[k0:k1,:,:,:]
        varr = varr0[k0:k1,:,:,:]

    return plot_model(model, md_dict0, lpx_symbol = lpx_symbol, pnl_tcost=pnl_tcost, \
            nf_cnt = nf_cnt, ni_shp_threshold = ni_shp_threshold, plot_old_exec=plot_old_exec, fig=fig, ax_list=ax_list, legend=legend, \
            param0=param0, f0=f, v0=v, fni0=fni, shp_ni0=shp_ni, farr0=farr, varr0=varr)


# ===

def vbs_pack(md_dict, vbs_col, utc_col, vbs_type=np.int16):
    """
    return: 
    vbs_pack: {'utc': shape[ ndays,2] start/end utc daily, 
               'vbs': shape[ nndays, nbars], nbars is number of bars per day}
    note, the vbs array has type short
    """
    p = {}
    ua = np.array([0,-1]).astype(int)
    for k in md_dict.keys():
        p[k] = {'utc': md_dict[k][:,ua,utc_col], 'vbs':md_dict[k][:,:,vbs_col].astype(vbs_type)}

    return p

