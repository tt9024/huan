import numpy as np
import ar1_eval
import copy
import Pop3
import os
import dill

"""
utils for going from cpa to a pop module

pop module: 
    input - ni_pick/f/v
    output - f/v
"""
def pack_exec_param(contract_size, tcost, pnl_tcost, maxpos):
    return {'contract_size':contract_size, \
            'tcost':tcost,\
            'pnl_tcost':pnl_tcost,\
            'maxpos':maxpos}

def pack_smooth_param(smooth_width=3, smooth_poly=3, threshold=0.0, min_group_bars=4, min_group_mean=1.0):
    smooth_param = {}
    smooth_param['smooth_width'] = smooth_width
    smooth_param['smooth_poly'] = smooth_poly
    smooth_param['threshold'] = threshold
    smooth_param['min_group_bars'] = min_group_bars
    smooth_param['min_group_mean'] = min_group_mean
    return smooth_param

def p0_from_fv(n, f, v, lpx, ixf, tcost_px, contract_size, pnl_tcost, maxpos=100, p0=0, fscale=None, vscale=(1,0,6), no_trade_k0_k1=None, lr0=0):
    """
    lpx: length nndays of vector of close price
    return:
       pp:  length nndays+1 vectors,  pp[0] = p0, pp[1] is the target position at close of bar[0], i.e. after getting f[0] and v[0], 
       pnl: length nndays vector, pnl[0] is the pnl achieved by pp[0] using lr0, 
            pnl[1] is the pnl achieved by pp[1] using lpx[1]-lpx[0]. 
    Note: 
            pp[-1] not included in pnl
    """
    wpos = 0
    min_trade = 1
    k0_k1_list = []
    pp, pnl = ar1_eval.p0_from_fnx2(f, None, ixf, lpx.flatten(), tcost=tcost_px, maxpos=maxpos, \
                           fscale=fscale, vscale=vscale, wpos = wpos, min_trade=min_trade, \
                           p0=p0, pnl_tcost=pnl_tcost, contract_size=contract_size, \
                           k0_k1_list = k0_k1_list, no_trade_k0_k1=no_trade_k0_k1, 
                           v = v, n=n, lr0=0)
    return pp, pnl

def get_lpx_from_lr(lpx, lr):
    """
    input:
        lpx,lr: shape [ndays, n]
        roll-adjust the lpx from the lr
    return:
        lpx_adj: shape[ndays,n] roll adjusted lpx
    """
    ndays, n = lpx.shape
    assert ndays==lr.shape[0] and n==lr.shape[1]

    # adjust each each day's lpx overnight using lr
    lpx_adj = lpx.flatten().copy()
    for d in np.arange(ndays-1).astype(int)+1:
        lr0 = lr[d,0]
        lpx0 = lpx[d-1,-1]
        lpx0_lr = lpx[d,0]*np.exp(-lr0)
        ra = lpx0_lr-lpx0
        #if np.abs(ra) > 1e-3:
        #    print(d, ra)
        lpx_adj[:d*n] += ra
    return lpx_adj.reshape((ndays,n))

def local_regression_gaussian(y, width=3, poly=3):
    """ local regression using a gaussian kernel
    """
    n = len(y)
    wt = np.exp(-np.arange(n)**2/(2*width**2))/(np.sqrt(2*np.pi)*width)
    wt = np.r_[wt[::-1], wt[1:]]
    nwt = len(wt)
    X = [np.ones(n)]
    for p in np.arange(poly) + 1 :
        X.append(np.arange(n)**p)
    X = np.array(X).T
    xs = []
    for i in np.arange(n):
        wn = wt[n-1-i:2*n-1-i]
        xtw =  X.T * wn
        wy = wn*y
        xs.append(np.dot(X[i,:], np.dot(np.dot(np.linalg.inv(np.dot(xtw,X)),X.T),wy)))
    return np.array(xs)

def pick_bars(shp_bar, smooth_width=3, smooth_poly=3, threshold=0.0, min_group_bars=3, min_group_mean=1):
    """
    pick_array=smooth_pnl_bar(shp_bar, smooth_width, smooth_poly, min_bars, min_bar_mean)
    Input: 
      shp_bar: length n vector of per-bar sharpe ratio, etstimated from previous history
      smooth_width/smooth_poly: local regression parameters, width (i.e. 1,2,3) increases smoothness,
                              poly (i.e. 1,3) the shape
      threshold: minimum smoothed pnl to be selected, as a fraction of mean positive per-bar pnl,
               0.0 means any positive bar, 1.0 mean any bar > 1.0 * mean positive per-bar pnl
      min_group_bars: minimum consecutive bars to be included as a group
      min_group_mean: minimum mean per-bar pnl to be included as a group\
    Output:
      pick_array: [[(k0,k1), group_mean_shp, group_mean_percent]]
      where (k0, k1) is the first/last bar_index to be included, both inclusive.

    Selects positive pnl groups and return a pick_array
    """
    n = len(shp_bar)
    ys = local_regression_gaussian(shp_bar, width=smooth_width, poly=smooth_poly)
    pick_array = []
    ixp = np.nonzero(ys>0)[0]
    if len(ixp) == 0:
        return pick_array
    mpv = np.sum(ys[ixp])/len(ixp)
    th0 = mpv * threshold
    th1 = mpv * min_group_mean
    th2 = mpv * min_group_bars

    ix0 = np.nonzero(ys>th0)[0]
    if len(ix0) == 0:
        return pick_array
    
    ixy = np.zeros(n)
    ixy[ix0] = 1
    
    ixs = ix0[0]
    ix = min(ixs + 1, n-1)
    while ix < n:
        while ix<n and ixy[ix] == 1:
            ix += 1
        # ix points to 1 + ixe, where ixe is the last positive index
        if ix - ixs >= min_group_bars or ixs==0:
            gs = np.sum(ys[ixs:ix])
            if gs > th2 or ixs==0:
            #gm = np.mean(ys[ixs:ix])
            #if gm > th1:
                pick_array.append([(ixs, ix-1), gs, gs/th2])
        while ix<n and ixy[ix] ==0:
            ix += 1
        if ix >= n:
            break
        ixs = ix
        ix = min(ixs+1,n-1)
    return pick_array

def pick_bars_param(shp_bar, smooth_param):
    return pick_bars(shp_bar, 
            smooth_width = smooth_param['smooth_width'], \
            smooth_poly = smooth_param['smooth_poly'], \
            threshold =  smooth_param['threshold'],\
            min_group_bars = smooth_param['min_group_bars'],\
            min_group_mean = smooth_param['min_group_mean'])

def gen_group_shp(pick_array, pnl, n):
    nndays = len(pnl)
    ndays = nndays//n
    assert ndays*n == nndays
    pnld = pnl.reshape((ndays,n))


    shp_group = []
    for pick in pick_array:
        (k0, k1) = pick[0]
        pnlg = np.sum(pnld[:,k0:k1+1],axis=1)
        shp_group.append(np.mean(pnlg)/np.std(pnlg))
    shp_group = np.array(shp_group)
    shp_group[np.isnan(shp_group)] = 0
    shp_group[np.isinf(shp_group)] = 0
    return shp_group

def fv_2_model_dict(model_name, n, fnf, vnf, ixf, lpx, tcost_test_array,\
                    exec_param, smooth_param, pnl_tcost_bar):
    """
    return a dict to store all the necessary information on how to add this model

    model_dict = {'name': model_name, 'n': n, 'ixf': ixf, \
                  'pick': pick_array,         # array of [(k0,k1), smoothed_group_shp, ratio_over_min_group_shp]
                  'tcost':best_tcost,         # tcost used to get pos from f/v, picked from tcost_test_array
                  'vol_scale': wt,            # the annual vol adjusted for the model
                  'pos': pp, 'pnl':pnl,       # the unadjusted pos and pnl from f/v, see NOTE
                  'shp_bar':shp_n,            # daily sharp of pnl
                  'group_scale': shp_group,   # group specific weight, from group shp
                  ''pos_scale':               # pos scaled with group_scale and vol_scale,

    NOTE - position in mdl_array is 1 position shifted
    i.e. pos[day, bar] is the position entering into this bar
    it is written this way for the corresponding k0k1 picking.
    The pnl[day, bar] is the pnl achieved at close of the bar.

    NOTE2 - position in 'pos_scale' is applied with first the 
    global annual vol 'wt', and then apply a per-group sharpe
    scale, as the "annual" sharpe of the group. This is so
    that picking can compared directly
     
    """
    contract_size = exec_param['contract_size']
    pnl_tcost = exec_param['pnl_tcost']
    maxpos = exec_param['maxpos']
    nndays, nf = fnf.shape
    ndays = nndays//n
    assert len(ixf) == nf
    if (lpx.shape) == 2:
        # expect ndays, n
        assert lpx.shape[0] == ndays
        assert lpx.shape[1] == n
        lpx = lpx.flatten()
    else:
        assert len(lpx) == nndays

    # find a best tcost 
    best_shp = 0
    for tcost in tcost_test_array:
        pp0, _ = p0_from_fv(n, fnf, vnf, lpx, ixf, tcost, contract_size, pnl_tcost, maxpos=maxpos, p0=0, fscale=None, vscale=(1,0,6), no_trade_k0_k1=None)
        _, pnl0, _ = pnl_from_pos_tcost(lpx.reshape((ndays,n)), pp0, pnl_tcost_bar, maxpos, contract_size)
        pnld0 = np.sum(pnl0.reshape((ndays,n)),axis=1)
        shp0 = np.mean(pnld0)/np.std(pnld0)
        if np.isnan(shp0) or np.isinf(shp0):
            shp0 = 0
        print('running for tcost', tcost, ' got shp ', shp0)
        if shp0 > best_shp:
            best_shp = shp0
            best_tcost = tcost
            pp = pp0.copy()
            pnl = pnl0.copy()

    if best_shp <= 0:
        return {'name': model_name, 'n': n, 'ixf': ixf, \
                  'pick': []}

    # getting the shp and pick
    pnl_day = pnl.reshape((ndays, n))
    shp_n = np.mean(pnl_day,axis=0)/np.std(pnl_day,axis=0)
    shp_n[np.isnan(shp_n)]=0
    shp_n[np.isinf(shp_n)]=0
    pick_array = pick_bars_param(shp_n, smooth_param)

    # given the pick, figure out each group's shp based on pnl
    shp_group = []
    for pick in pick_array:
        (k0, k1) = pick[0]
        pnlg = np.sum(pnl_day[:,k0:k1+1],axis=1)
        shp_group.append(np.mean(pnlg)/np.std(pnlg)/np.sqrt((k1-k0)+1)*np.sqrt(252*n))
    shp_group = np.array(shp_group)
    shp_group[np.isnan(shp_group)] = 0
    shp_group[np.isinf(shp_group)] = 0

    # figure out the position scale to be added
    pnl_d = np.sum(pnl_day, axis=1)
    annual_vol = np.std(pnl_d)*np.sqrt(252)
    wt = 1e+6/annual_vol

    # scale position based on shp_gropup and wt
    # shp_group index with pick_array
    # NOTE - position in mdl_array is 1 position shifted
    # i.e. pos[day, bar] is the position entering into this bar
    # it is written this way for the corresponding k0k1 picking
    pos_day = pp[:-1].reshape((ndays,n))
    pos_day_scale = np.zeros((ndays,n))
    for pick, gw in zip(pick_array, shp_group):
        (k0, k1) = pick[0]
        pos_day_scale[:,k0:k1+1] = pos_day[:,k0:k1+1]*gw*wt

    model_dict = {'name': model_name, 'n': n, 'ixf': ixf, \
                  'pick': pick_array, 'tcost':best_tcost, \
                  'vol_scale': wt, 'pos': pp, 'pnl':pnl, 'shp_bar':shp_n, 
                  'group_scale': shp_group, 'pos_scale': pos_day_scale}

    return model_dict

def gen_model_pnl(model_dict, pos, pnl):
    n = model_dict['n']
    scl = model_dict['pos_scale']
    pick_array = model_dict['pick']

    ndays = len(pnl)//n
    pnl_d = pnl.reshape((ndays, n))
    pos_d = pos.reshape((ndays, n))
    pnl_d_out = np.zeros((ndays,n))
    pos_d_out = np.zeros((ndays,n))
    for pick in pick_array:
        (k0, k1) = pick[0]
        pnl_d_out[:,k0:k1+1] = pnl_d[:,k0:k1+1]
        pos_d_out[:,k0:k1+1] = pos_d[:,k0:k1+1]
    return pnl_d_out*scl, pos_d_out*scl

def run_ensemble(model_dict_array, maxpos = 400):
    """
    run each model with its position pick and output
    a simple pnl and a sharp.  Assuming each model has a same days.

    1. generate each model's daily position and pnl
    2. find maxpos by comparing the shp
    3. set the scale
    4. pack a ensemble dict
    """

    pnl_d, pos_d = gen_model_pnl(model_dict_array[0], \
                  model_dict_array[0]['pos'], \
                  model_dict_array[0]['pnl'])

    for mdl in model_dict_array[1:]:
        pnl0, pos0 = gen_model_pnl(mdl, mdl['pos'], mdl['pnl'])
        pnl_d += pnl0
        pos_d += pos0

    # scl with the maxpos
    ix = np.nonzero(np.abs(pos_d)>maxpos)[0]
    if len(ix)>0:
        scl = maxpos/np.abs(pos_d[ix])
        pos_d[ix] *= scl
        pnl_d[ix] *= scl

    pnl = np.sum(pnl_d,axis=1)
    shp = np.mean(pnl)/np.std(pnl)
    scl = 1e+6/np.std(pnl)
    return pnl_d, pos_d, shp, scl

def get_fv_from_tshp(tshp_dict, model_ix, sday_ix, eday_ix, md_dict_extension=None):
    """
        n, f, v, lpx, utc, ixf = get_fv_from_tshp(tshp_dict, mix, sday_ix, eday_ix)
        f,v: shape [nndays,n]
        lpx,utc: shape[ndays,n]
    """
    f = np.sum(tshp_dict['fni'][model_ix],axis=0)
    nndays, nf = f.shape
    v = tshp_dict['vnf'][model_ix]
    ixf = tshp_dict['ixf'][model_ix]
    ndays, n, _ = tshp_dict['tshp'][model_ix].shape
    assert n == ixf[-1]+1
    assert nf == len(ixf)
    assert ndays == nndays//n
    name = tshp_dict['fn'][model_ix].split('/')[-1].split('.')[0]
    symbol = tshp_dict['symbol']
    md = tshp_dict['md_dict'][symbol]

    if md_dict_extension is not None:
        fm = copy.deepcopy(tshp_dict['fm_dict'][name])
        f0,v0,_ = fm.run_days(md_dict_extension)
        # extend f,v,md
        f = np.vstack((f,f0))
        v = np.vstack((v,v0))
        md = np.vstack((md, md_dict_extension[symbol]))

    # get lpx from lr - better for roll adjustment
    lpxcol=3
    lrcol =0
    tcol = 4
    lpx = get_lpx_from_lr(md[:,:,lpxcol],md[:,:,lrcol])
    utc = md[:,:,tcol]
    # apply sday_ix and eday_ix
    f = f[sday_ix*n:eday_ix*n,:]
    v = v[sday_ix*n:eday_ix*n,:]
    lpx = lpx[sday_ix:eday_ix,:]
    utc = utc[sday_ix:eday_ix,:]
    return n, f, v, lpx, utc, ixf

def get_fv_from_fm(fm_in, md_dict, symbol, sday_ix, eday_ix):
    n = fm_in.fcst_obj.n
    ixf = fm_in.fcst_obj.ixf
    md = md_dict[symbol]
    ndays, n0, k0 = md.shape
    assert n == n0
    assert k0 >= 5

    lpxcol=3
    lrcol =0
    tcol = 4
    lpx = get_lpx_from_lr(md[:,:,lpxcol],md[:,:,lrcol])
    utc = md[:,:,tcol]
    fm = copy.deepcopy(fm_in)
    f,v,_ = fm.run_days(md_dict)

    # apply sday_ix and eday_ix
    f = f[sday_ix*n:eday_ix*n,:]
    v = v[sday_ix*n:eday_ix*n,:]
    lpx = lpx[sday_ix:eday_ix,:]
    utc = utc[sday_ix:eday_ix,:]
    stdv = fm_in.state_obj.std.v.copy() # last day's stdv
    return fm, n, f, v, lpx, utc, ixf, stdv

def tshp_2_model_dict(tshp_dict, model_ix, sday_ix, eday_ix, tcost_test_array, exec_param, smooth_param, md_dict_extension=None):
    n, f, v, lpx, utc, ixf = get_fv_from_tshp(tshp_dict, model_ix, sday_ix, eday_ix, md_dict_extension=md_dict_extension)
    name = tshp_dict['fn'][model_ix].split('/')[-1].split('.')[0]
    stdv_ref = tshp_dict['stdvh'][0][sday_ix] # use the first days' stdv
    pnl_tcost_bar = tcost_from_std(stdv_ref, exec_param['pnl_tcost'])
    return fv_2_model_dict(name, n, f, v, ixf, lpx.flatten(), tcost_test_array, \
            exec_param, smooth_param, pnl_tcost_bar)

def collect_tshp_2_ensemble(tshp_dict, sday_ix, eday_ix, tcost_test_array, exec_param, smooth_param, exclude_model_ix=[], md_dict_extension=None):
    """
    pick a set up models to be included in ensemble
    """
    mdl_cnt = len(tshp_dict['fn'])
    mdl_array = []
    for i in np.arange(mdl_cnt).astype(int):
        if i in exclude_model_ix:
            mdl_array.append({})
            continue
        mdict = tshp_2_model_dict(tshp_dict, i, sday_ix, eday_ix, tcost_test_array, exec_param, smooth_param, md_dict_extension=md_dict_extension)
        mdl_array.append(mdict)
    return mdl_array

def tcost_from_std(stdv, pnl_tcost):
    tcost = local_regression_gaussian(1.0/stdv, width=6, poly=3)
    tcost = np.clip(tcost/tcost.mean() * pnl_tcost, pnl_tcost, 1e+10)
    return tcost

def pnl_from_pos_tcost(lpx, pos, tcost_bar, maxpos, contract_size, lr0=0, fee=2.0, clip_maxpos=True):
    """
    input:
        lpx: shape [ndays, n] of close price of each bar
        pos: length nndays+1 vector of position, pos[0] is the incoming position, pos[1]
             put at the close of first bar
        tcost_bar: length n vector of tcost at each bar
        lr0: log return of the first bar
        clip_maxpos: if true, simple clip pos to be within [-maxpos, maxpos], otherwise
                     scale pos to be within the maxpos
    return:    pos, pnl, shp
        pos: length nndays+1 vector of position, clipped by maxpos
        pnl: length nndays per-bar pnl after tcost and fee, the first pnl is achieved by pos[0] at
             the close of first bar
        shp: daily sharpe ratio of pnl
    """
    ndays, n = lpx.shape
    if clip_maxpos:
        pos = np.clip(pos, -maxpos, maxpos)
    else:
        pos*=(maxpos/np.max(np.abs(pos)))

    p00 = pos[0]
    p0  = pos[1:]
    assert len(p0) == ndays*n
    assert len(tcost_bar) == n
    lpx=lpx.flatten()
    tcost = np.tile(tcost_bar,(ndays,1)).flatten()

    p0d = p0-np.r_[p00,p0[:-1]]
    trd_cst = np.abs(p0d)*tcost*contract_size+np.abs(np.sign(p0d))*fee
    lpx0 = np.r_[lpx[0]/np.exp(lr0),lpx]
    pnl = (lpx0[1:]-lpx0[:-1])*np.r_[p00,p0[:-1]]* contract_size
    #pnl = np.r_[0,(lpx[1:]-lpx[:-1])*p0[:-1]]* contract_size
    pnl_cst = pnl-trd_cst

    pnld = pnl_cst.reshape((ndays,n))
    shp = np.mean(np.sum(pnld,axis=1))/np.std(np.sum(pnld,axis=1))
    if np.isnan(shp) or np.isinf(shp):
        shp=0
    return pos, pnl_cst, shp


def find_best_group(mdl_array, lpx, tcost_bar, contract_size, max_group_pos=20, exclude_gp=None, clip_maxpos=True, best_pos=None, best_shp=0):
    """
    iteratively find a best group based on previous pos/pnl among the given mdl_array
    return: best_gp, best_pos, best_pnl, best_shp
    best_gp: group id: (model_id, group_id),
    best_pos, pnl, shp: unscaled pos/pnl/shp, only the picked group has non-zero values
    """
    ndays, n = lpx.shape
    mcnt = len(mdl_array)
    best_gp = None
    if best_pos is None:
        best_pos = np.zeros((ndays,n))

    best_shp0 = best_shp
    for m, mdl_dict in enumerate(mdl_array):
        pick_array = mdl_dict['pick']
        for p, pick in enumerate(pick_array):
            if  exclude_gp is not None and (m,p) in exclude_gp:
                continue
            (k0,k1) = pick[0]
            pos0 = best_pos.copy()
            pos0[:,k0:k1+1]+=mdl_dict['pos_scale'][:,k0:k1+1]
            pos00, pnl0, shp0 = pnl_from_pos_tcost(lpx, np.r_[pos0.flatten(),0], tcost_bar, max_group_pos, contract_size, clip_maxpos=clip_maxpos)

            # recover the original k0k1 position. i.e. from model_dict['pos_scale']
            pos0-=best_pos
            if shp0>best_shp0:
                print('  best shp update - ', m, p, '(%d,%d) shp:%f pnl:%f'%(k0,k1,shp0,np.sum(pnl0)))
                best_shp0 = shp0
                best_gp0 = (m,p)
                best_pos0 = pos0.copy()
                best_pnl = pnl0.copy()
    dshp = best_shp0-best_shp
    if dshp == 0:
        print('Nothing beats the given, fold')
        return None, None, None, None

    print('PICK: ', best_gp0, best_shp0, np.sum(best_pnl))
    return best_gp0, best_pos0, best_pnl, best_shp0

def ens_fit_param(exec_param, smooth_param, tcost_test_array, barsec=300):
    return {'exec_param': exec_param, \
            'smooth_param':smooth_param,\
            'tcost_test_array':tcost_test_array,\
            #'max_group_pos':max_group_pos, \
            'clip_maxpos':True, \
            'barsec': barsec
           }
    md_dict_extension, tcostexclude_model_ix

def get_negative_group(tshp_dict, mdl_array, test_sday, test_eday):
    exclude_gp = []
    for i, mdl_dict in enumerate(mdl_array):
        n = tshp_dict['stdvh'][i].shape[1]
        if 'tcost' not in mdl_dict.keys():
            continue
        pnl = mdl_dict['pnl']
        nndays = pnl.shape[0]
        pnl = pnl.reshape((nndays//n,n))

        for k, pick in enumerate(mdl_dict['pick']):
            k0,k1=pick[0]
            if np.sum(pnl[test_sday:test_eday,k0:k1+1]) < 0:
                print('removing model ', i, ' k0k1 ', k0,k1, ' negative pnl in test days')
                exclude_gp.append((i,k))
    return exclude_gp

def fit_ensemble(tshp_dict, sday_ix, eday_ix, fit_param, exclude_model_ix=[], md_dict_extension=None, mdl_array=None, exclude_gp=[]):
    """find gropus of consecutive bars with good sharpe for each model and aggregate the groups into several ensembles.
       Each ensemble includes a group of consecutive bars with proper scaling.  The final position is the
       aggregation of group positions subject to a maximum group position and an ensemble scale.
    return: 
         ens_array: list of ens_dict, fitted ensembles of groups, with format of 
             {'mk0k1_array': ['model_ix', 'k0k1', 'pos_scl'], 'max_gp_pos', 'ens_scl'}
         mdl_array: list of model_dict, defining groups of consecutive bars to be aggregated in ensemble.
             {'name', 'n', 'ixf', 'pick', 'tcost', 'vol_scale', 'pos', 'pnl', 'shp_bar', 'group_scale', 'pos_scale'}
    """
    if mdl_array is None:
        print('mdl_array is None, getting it')
        mdl_array = collect_tshp_2_ensemble(tshp_dict, sday_ix, eday_ix, \
                                   fit_param['tcost_test_array'], \
                                   fit_param['exec_param'],\
                                   fit_param['smooth_param'], \
                                   exclude_model_ix=exclude_model_ix, md_dict_extension=md_dict_extension)

    n, _,_, lpx,_,ixf = get_fv_from_tshp(tshp_dict, 0, sday_ix, eday_ix, md_dict_extension=md_dict_extension)
    stdv_ref = tshp_dict['stdvh'][0][eday_ix]
    exec_param = fit_param['exec_param']
    tcost_bar = tcost_from_std(stdv_ref, exec_param['pnl_tcost'])
    contract_size = exec_param['contract_size']
    maxpos = exec_param['maxpos']
    #max_group_pos = fit_param['max_group_pos']
    clip_maxpos=fit_param['clip_maxpos']

    # gp:(k0,k1), pos:shape(ndays,n) including p00 as the first, pnl:shape(nndasy)
    gp, pos, pnl, shp = find_best_group(mdl_array, lpx, tcost_bar, contract_size, max_group_pos=maxpos, clip_maxpos=clip_maxpos, exclude_gp=exclude_gp)
    (m,p)=gp
    shp_g = mdl_array[m]['group_scale'][p]
    # pnl is from the group's position, with tcost_bar
    _,pnl_g,_=pnl_from_pos_tcost(lpx, np.r_[pos.flatten(),0], tcost_bar, maxpos, contract_size)

    print('* adding (initial):', gp, ' shp', shp)
    ens_gp = [copy.deepcopy(gp)]
    ens_pnl = [pnl_g.copy()]
    ens_pos = [pos.copy()]
    ens_shp = [shp_g]
    ens_wt = np.array([1.0])

    run_cnt = 100

    exclude_gp = copy.deepcopy(exclude_gp) + [gp]

    mkt_shp = shp # the sharp including all portfolios in the ensemble
    while run_cnt>0:
        run_cnt -= 1
        gp, pos, pnl, shp = find_best_group(mdl_array, lpx, tcost_bar, contract_size, max_group_pos=maxpos, exclude_gp=exclude_gp,clip_maxpos=clip_maxpos, best_pos=np.sum(ens_pos,axis=0))

        if gp is None:
            # cannot find any more groups
            break
        exclude_gp = exclude_gp + [copy.deepcopy(gp)]

        # calculate the new position's pnl, figure out a weight
        # and apply maxpos to the ensemble as the next iteration
        (m,p)=gp
        shp_g = mdl_array[m]['group_scale'][p]
        # pnl is from the group's position, with tcost_bar
        _,pnl_g,_=pnl_from_pos_tcost(lpx, np.r_[pos.flatten(),0], tcost_bar, maxpos, contract_size)

        # save the initial conditions
        ens_pnl0 = copy.deepcopy(ens_pnl)
        ens_pnl0.append(pnl_g)
        ens_gp0 = copy.deepcopy(ens_gp)
        ens_gp0.append(copy.deepcopy(gp))
        ens_pos0 = copy.deepcopy(ens_pos)
        ens_pos0.append(pos.copy())
        ens_shp0 = copy.deepcopy(ens_shp)
        ens_shp0.append(shp_g)

        while True:
            # position already scaled with shp weighted
            # here just to remove the correlations
            v=np.dot(np.array(ens_pnl0),np.array(ens_pnl0).T)
            vd=1.0/np.sqrt(v.diagonal())
            v*=np.outer(vd,vd)
            try:
                wt=np.dot(np.linalg.inv(v),np.array(ens_shp0))
                # remove negative weights if any
                ixw = np.argsort(wt)
                if wt[ixw[0]] < 1e-3:
                    # remove the most negative and try again
                    print('negative weight detected ', wt, ' removing ', ens_gp0[ixw[0]])
                    ens_pnl0.pop(ixw[0])
                    ens_gp0.pop(ixw[0])
                    ens_pos0.pop(ixw[0])
                    ens_shp0.pop(ixw[0])
                else:
                    break
            except:
                import traceback
                traceback.print_exc()
                return None

        wt/=np.sum(wt)
        posa,pnla,shpa=pnl_from_pos_tcost(lpx, np.r_[np.dot(np.array(ens_pos0).T,wt).T.flatten(),0], tcost_bar, maxpos, contract_size)
        print('* iterate %d\nens_gp:%s\nens_shp:%s\nens_wt:%s'%(run_cnt, str(ens_gp0), str(ens_shp0), str(wt)))
        if shpa > mkt_shp:
            # include it!
            print('SHP Improved: %f>%f'%(shpa, mkt_shp))
            ens_gp = copy.deepcopy(ens_gp0)
            ens_pos = copy.deepcopy(ens_pos0)
            ens_pnl = copy.deepcopy(ens_pnl0)
            ens_shp = copy.deepcopy(ens_shp0)
            ens_wt = copy.deepcopy(wt)
            mkt_shp=shpa
            continue
        print('NOT improved, try more')

    print('done with ', len(ens_gp), ' groups', 'ensemble_shp', mkt_shp, 'pick:', ens_gp, 'wt', ens_wt)
    mk0k1 = []
    for gp, wt in zip(ens_gp, ens_wt):
        (mdl_ix, pick_ix) = gp
        mdl_dict = mdl_array[mdl_ix]
        (k0, k1) = mdl_dict['pick'][pick_ix][0]
        group_wt = mdl_dict['vol_scale']*mdl_dict['group_scale'][pick_ix]*wt
        mk0k1.append( [mdl_ix, k0, k1, group_wt] )
    ens_dict={'mk0k1_array':copy.deepcopy(mk0k1), 'max_gp_pos':maxpos, 'ins_shp': mkt_shp}
    return ens_dict, mdl_array

def collect_test(tshp_dict, model_array, sday_ix, eday_ix, exec_param, md_dict_extension=None):
    """
    model_pos, model_f, model_pnl, model_ixf, vnf, lpx, utc, tcost_bar = collect_test()

    Note - model_pos a list of per-model (ndays,n) array.  element (d,k) is the
           STARTING position (NOT ending position), i.e. [0,0] is the first
           entering position, and [-1,-1] is the starting position of last bar.
           This setup is due to the ens_dict[k0,k1] refers to the pnl achieved, and
           therefore the starting position. 
           Live aggregation has to adjust k accordingly
    """
    nmdl = len(tshp_dict['fn'])
    assert len(model_array) == nmdl, 'model_array has to index with tshp_dict'
    _,n,_ = tshp_dict['tshp'][0].shape
    stdv_ref = tshp_dict['stdvh'][0][sday_ix] # use the first days' stdv
    pnl_tcost = exec_param['pnl_tcost']
    tcost_bar = tcost_from_std(stdv_ref, pnl_tcost)
    contract_size = exec_param['contract_size']
    maxpos = exec_param['maxpos']

    # for each model, get fnf,vnf,pos for each model on the give days
    model_pos = []
    model_f = []
    model_ixf = []
    model_pnl = []

    print('getting model positions')
    for model_ix in np.arange(nmdl).astype(int):
        n0, fnf,vnf,lpx,utc,ixf = get_fv_from_tshp(tshp_dict, model_ix, sday_ix, eday_ix, md_dict_extension=md_dict_extension)
        assert n0 == n
        nndays,nf=fnf.shape
        ndays=nndays//n
        assert ndays*n == nndays

        # using regular way to get position from fnf,vnf
        # pp length nndays+1, pnl length nndays
        if 'tcost' not in model_array[model_ix].keys():
            # tcost is the best tcost used in fitting. If not 
            # found, this model is not used. 
            # append nan so it index with tshp_dict and mdl_array
            model_pos.append(np.nan)
            model_f.append(np.nan)
            model_pnl.append(np.nan)
            model_ixf.append(np.nan)
        else :
            tcost = model_array[model_ix]['tcost']
            pp, pnl = p0_from_fv(n, fnf, vnf, lpx, ixf, tcost, contract_size, pnl_tcost, maxpos=maxpos, p0=0, fscale=None, vscale=(1,0,6), no_trade_k0_k1=None, lr0=0)
            model_pos.append(pp[:-1].reshape((ndays,n)).copy())
            model_pnl.append(pnl.reshape((ndays,n)).copy())
            model_f.append(fnf.reshape((ndays,n,nf)).copy())
            model_ixf.append(Pop3.NfNx(n,ixf,stdv_ref))
    return model_pos, model_f, model_pnl, model_ixf, vnf, lpx, utc, tcost_bar

def test_ens_dict(ens_dict, model_pos, lpx, tcost_bar, exec_param, lr0=0):
    """
    ens_dict: returned by fit_ensemble()
    model_pos, lpx, tcost_bar: returned by previous collect
    """
    pos = None
    mk0k1_array = ens_dict['mk0k1_array']
    for mk0k1 in mk0k1_array:
        m,k0,k1,wt=mk0k1
        if pos is None:
            ndays,n=model_pos[m].shape
            pos = np.zeros((ndays,n))
        pos[:,k0:k1+1] += (model_pos[m][:,k0:k1+1] * wt)
    pos_agg_pos, pnl_agg_pos, shp_agg_pos = pnl_from_pos_tcost(lpx, np.r_[pos.flatten(),0], tcost_bar, exec_param['maxpos'], exec_param['contract_size'], lr0=lr0)
    return pos_agg_pos, pnl_agg_pos, shp_agg_pos

### pnl plotting made easy
# n,f,v,lpx,utc,ixf = pop4.get_fv_from_tshp(tshp_dict, model_ix, sday_ix, eday_ix, md_dict_extension=md_dict)
# pp, pnl = p0_from_fv(n,f,v,lpx, ixf, tcost, contract_size, pnl_tcost, maxpos=maxpos, p0=0, fscale=None, vscale=(1,0,6), no_trade_k0_k1=None, lr0=0)
# ar1_eval.plot_pnl_from_p0(ax_list, p00, p0, md, tcost, contract_size = 1000, plot_md=False, marker = '.-', legend='')

def plot_lpx_pos_pnl(axlist,lpx,utc,pos,pnl,label,marker='-',f=None,plot_lpx=False):
    import datetime
    dt = []
    if f is not None:
        fnf=f.T[:,:]
        assert len(axlist)==4
    if len(utc.shape)==2:
        utc=utc.flatten()
    if len(lpx.shape)==2:
        lpx=lpx.flatten()
    for t in utc:
        dt.append(datetime.datetime.fromtimestamp(t))
    dt = np.array(dt)

    if plot_lpx:
        axlist[-3].plot(dt, lpx,marker)
    axlist[-2].plot(dt,pos,marker,label=label)
    axlist[-1].plot(dt,np.cumsum(pnl),marker,label=label)
    if f is not None:
        dtd=dt[1]-dt[0]
        dt=np.r_[dt,dt[-1]+dtd]
        dt-=dtd/2
        Y = np.arange(fnf.shape[0]+1)-0.5
        import matplotlib.dates as mdates
        X = mdates.date2num(dt)
        axlist[0].pcolormesh(X,Y,fnf, cmap='RdBu', vmin=-1e-3, vmax=1e-3)
        k = 1

    return axlist

def shp_from_f(n,f,lpx,ixf,sday_ix=None,eday_ix=None,stdv=None):
    lr0 = np.r_[0, np.log(lpx.flatten()[1:])-np.log(lpx.flatten()[:-1])]
    shp = ar1_eval.get_sharp_days(lr0,f,n,ixf,sday_ix=sday_ix,eday_ix=eday_ix,stdv=stdv)
    return shp

####################################################################
# procedure to fit the ens_dict:
#
# to fit in insample from days [50, -50]
#
# 1. update md since the last day of tshp
#    md_dict = ar1_md.get_md_days(ar1_md.mts_repo_mapping.keys(), '20220721','20220801', 300)
# 2. mdl_array = collect_tshp_2_ensemble(tshp_dict, 50, -50, [0.05,0.1,0.15,0.25,0.5], exec_param, smooth_param, exclude_model_ix=[0], md_dict_extension=md_dict)
# 3. ens_dict, mdl_array = fit_ensemble(tshp_dict, 50, -50, fit_param, md_dict_extension=md_dict, mdl_array=mdl_array, exclude_gp=[])
#
# to test (out sample) from days [-50, -1]
#
# 4. n,fnf,vnf,lpx,utc,ixf=pop4.get_fv_from_tshp(tshp_dict, 1, -50, -1, md_dict_extension=md_dict)
# 5. posa, pnla, shpa = test_ens_dict(ens_dict, model_pos, lpx, tcost_bar, exec_param)
#
# to release ens_dict: 
# 
# 6. mdl_array = collect_tshp_2_ensemble(tshp_dict, 50, -1, [0.05,0.1,0.15,0.25,0.5], exec_param, smooth_param, exclude_model_ix=[0], md_dict_extension=md_dict)
#
# 7. ens_dict, mdl_array = fit_ensemble(tshp_dict, 50, -1, fit_param, md_dict_extension=md_dict, mdl_array=mdl_array, exclude_gp=[(6,0)])
# 
# Note - be very careful about the mdl with tcost high - such as (6,0)
#  
##################################################################

def fit_ens(tshp_dict, md_extension_sday_eday, fit_param, fit_sday_eday = (50, -50), test_sday_eday = (-50, -1), exclude_model_ix=[], exclude_gp=[], md_dict_extension=None, mdl_array_in=None):
    """
    md_extension_sday_eday: i.e. ('20220721','20220801')
    fit_sday_eday: day index (indexed from the entire days, i.e. the tshp_dict days PLUS the extension days), to be used for fitting
    test_sday_eday: the day index (same as fit_sday_eday) to be used for testing negative pnl group to be excluded
    exclude_model_ix: models index from tshp_dict (and model_array) to be excluded in ensemble
    exclude_gp: (model_idx, group_idx), to be excluded in ensemble
    """
    basrec = fit_param['barsec']
    tcost_test_array = fit_param['tcost_test_array']
    exec_param = fit_param['exec_param']
    smooth_param = fit_param['smooth_param']
    md_dict = md_dict_extension
    if md_dict is None:
        sday, eday = md_extension_sday_eday
        md_dict = ar1_md.get_md_days(ar1_md.mts_repo_mapping.keys(), sday, eday, fit_param['barsec'])
    mdl_array = mdl_array_in
    fit_sday, fit_eday = fit_sday_eday

    if mdl_array is None:
        mdl_array = collect_tshp_2_ensemble(tshp_dict, fit_sday, fit_eday, tcost_test_array, exec_param, smooth_param, exclude_model_ix=exclude_model_ix, md_dict_extension=md_dict)

        # save the data in case fitting changes are expected
        # mdl_array takes a long time to collect
        return md_dict, mdl_array

    # include the negative pnl groups in the exclude_gp
    exclude_gp += get_negative_group(tshp_dict, mdl_array, -50, -1)
    ens_dict, mdl_array = fit_ensemble(tshp_dict, fit_sday, fit_eday, fit_param, md_dict_extension=md_dict, mdl_array=mdl_array, exclude_gp=exclude_gp)

    if test_sday_eday is None:
        # return the ens_dict
        return ens_dict, mdl_array

    # test the pos/pnl/shp in the test
    test_sday, test_eday = test_sday_eday

    # remove groups with negative pnl in the last test days
    model_pos, model_f, model_pnl, model_ixf, vnf, lpx, utc, tcost_bar = collect_test(tshp_dict, mdl_array, test_sday, test_eday, exec_param, md_dict_extension=md_dict)
    posa, pnla, shpa = test_ens_dict(ens_dict, model_pos, lpx, tcost_bar, exec_param)
    ndays, n = lpx.shape
    pnla_d = pnla.reshape((ndays,n))
    annual_vol = np.std(np.sum(pnla_d,axis=1))*np.sqrt(252)
    return md_dict, mdl_array, ens_dict, posa, pnla, shpa, annual_vol


def release_ens_dict(md_dict_all_time, persist_path='/tmp'):
    #fn={'fm4':'n4_upd_rt.dill','fm6':'n6_st.dill', 'fm7':'n7_zf_ng_release.dill','fm9':'n9_fx_release.dill', 'fm10':'n10_st_release.dill', 'fm11':'n11_lr_release.dill', 'fm12':'n12_zn_release.dill'}
    fn={'fm4':'n4_upd_rt.dill', 'fm7':'n7_zf_ng_release.dill','fm9':'n9_fx_release.dill', 'fm10':'n10_st_release.dill', 'fm11':'n11_lr_release.dill', 'fm12':'n12_zn_release.dill'}
    for k in fn.keys():
        fm = dill.load(open(fn[k],'rb'))['fm']
        f,v,_ = fm.run_days(md_dict_all_time)
        fp=open(os.path.join(persist_path, k), 'wb')
        dill.dump(fm,fp)
        fp.close()

#################
# code not used #
#################

def test_ensemble_tshp(tshp_dict, ens_array, mdl_array, sday_ix, eday_ix, fit_param, md_dict_extension=None, model_pos=None, model_f=None, model_ixf=None):
    nmdl = len(tshp_dict['fn'])
    _,n,_ = tshp_dict['tshp'][0].shape
    stdv_ref = tshp_dict['stdvh'][0][sday_ix] # use the first days' stdv
    exec_param = fit_param['exec_param']
    pnl_tcost = exec_param['pnl_tcost']
    tcost_bar = tcost_from_std(stdv_ref, pnl_tcost)
    contract_size = exec_param['contract_size']
    maxpos = exec_param['maxpos']
    max_group_pos = fit_param['max_group_pos']

    # for each model, get fnf,vnf,pos for each model on the give days
    if model_pos is None or model_f is None or model_ixf is None:
        model_pos = []
        model_f = []
        model_ixf = []
        model_pnl = []

        print('getting model positions')
        for model_ix in np.arange(nmdl).astype(int):
            n0, fnf,vnf,lpx,_,ixf = get_fv_from_tshp(tshp_dict, model_ix, sday_ix, eday_ix, md_dict_extension=md_dict_extension)
            assert n0 == n
            nndays,nf=fnf.shape
            ndays=nndays//n
            assert ndays*n == nndays

            # using regular way to get position from fnf,vnf
            # pp length nndays+1, pnl length nndays
            if 'tcost' not in mdl_array[model_ix].keys():
                # tcost is the best tcost used in fitting. If not 
                # found, this model is not used. 
                # append nan so it index with tshp_dict and mdl_array
                model_pos.append(np.nan)
                model_f.append(np.nan)
                model_pnl.append(np.nan)
                model_ixf.append(np.nan)
            else :
                tcost = mdl_array[model_ix]['tcost']
                pp, pnl = p0_from_fv(n, fnf, vnf, lpx, ixf, tcost, contract_size, pnl_tcost, maxpos=maxpos, p0=0, fscale=None, vscale=(1,0,6), no_trade_k0_k1=None, lr0=0)
                model_pos.append(pp[:-1].reshape((ndays,n)).copy())
                model_pnl.append(pnl.reshape((ndays,n)).copy())
                model_f.append(fnf.reshape((ndays,n,nf)).copy())
                model_ixf.append(Pop3.NfNx(n,ixf,stdv_ref))
        return model_pos, model_f, model_pnl, model_ixf
    else:
        # want vnf, lpx and ixf
        n0, fnf,vnf,lpx,_,ixf = get_fv_from_tshp(tshp_dict, 0, sday_ix, eday_ix, md_dict_extension=md_dict_extension)
        assert n0 == n
        nndays,nf=fnf.shape
        ndays=nndays//n
        assert ndays*n == nndays

    # creating ensemble position and forecasts
    mkt_pos = []
    mkt_f = []

    print('calculating ensembles positions')
    for ens_dict in ens_array:
        mk0k1_array = ens_dict['mk0k1_array']
        max_gp_pos = ens_dict['max_gp_pos']
        ens_scl = ens_dict['ens_scl']

        ens_pos = np.zeros((ndays,n))
        ens_f = np.zeros((ndays,n,n))
        for (m, k0, k1, wt) in mk0k1_array:
            ens_pos[:,k0:k1+1] += (model_pos[m][:,k0:k1+1]*wt)

            # the k0,k1 refers to the pnl at ending bar
            # due to the forecast from previous bar
            # get the previous forecasts here
            ndays0,n0,nf0 = model_f[m].shape
            assert ndays0 == ndays and n0==n

            mfnf = np.r_[np.zeros(nf0),model_f[m].flatten()][:-nf0].reshape((ndays,n,nf0))
            mf_1 = np.zeros((ndays,n,nf0))
            mf_1[:,k0:k1+1,:] = (mfnf[:,k0:k1+1,:]*wt)
            mfnf = np.r_[mf_1.flatten()[nf0:], np.zeros(nf0)].reshape((ndays*n,nf0))
            mfnx = model_ixf[m].fnx_days(mfnf)
            nndays0,nx0 = mfnx.shape
            ens_f[:,:,:nx0] += mfnx.reshape((ndays,n,nx0))

        # figure out the group_maxpos_scale
        grp_maxpos_scale = 1.0/np.clip(np.abs(ens_pos)/(np.ones((ndays,n))*max_gp_pos),1,1e+10)
        ens_pos*=(grp_maxpos_scale*ens_scl)
        #ens_f*=(grp_maxpos_scale*ens_scl) #third index in

        # scale f similarly with pos would be subject to maxpos
        ens_f = np.r_[np.zeros(n), ens_f.flatten()[:-n]].reshape((ndays,n,n))
        ens_f=(ens_f.T*(grp_maxpos_scale*ens_scl).T).T
        ens_f = np.r_[ens_f.flatten()[n:],np.zeros(n)].reshape((ndays,n,n))

        mkt_pos.append(ens_pos.copy())
        mkt_f.append(ens_f.copy())

    # TODO - use a size-dependant tcost
    # calculate the aggregation
    print('calculating the aggregated position')
    mkt_pos = np.sum(mkt_pos,axis=0)
    pos_agg_pos, pnl_agg_pos, shp_agg_pos = pnl_from_pos_tcost(lpx, np.r_[mkt_pos.flatten(),0], tcost_bar, maxpos, contract_size, lr0=0, fee=2.0)

    # create pos from fnx
    print('calculating the aggregated position from f')
    mkt_f = np.sum(mkt_f,axis=0)
    nfnx=Pop3.NfNx(n,ixf,stdv_ref)
    vnx=nfnx.fnx_days(vnf)
    ixf_nx = np.arange(n).astype(int)
    tcost = pnl_tcost * 2  # set as a reference 
    pos_nx, pnl_nx = p0_from_fv(n, mkt_f.reshape((ndays*n,n)), vnx, lpx, ixf_nx, tcost, contract_size, pnl_tcost, maxpos=maxpos, p0=0, fscale=None, vscale=(1,0,6), no_trade_k0_k1=None, lr0=0)
    pos_agg_f, pnl_agg_f, shp_agg_f = pnl_from_pos_tcost(lpx, pos_nx, tcost_bar, maxpos, contract_size, lr0=0, fee=2.0)

    return pos_agg_pos, pnl_agg_pos, shp_agg_pos, pos_agg_f, pnl_agg_f, shp_agg_f

