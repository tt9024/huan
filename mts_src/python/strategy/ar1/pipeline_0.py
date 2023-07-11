import numpy as np
import Outliers as OT
import copy

# haven't tested yet
def clean_md(lr,vol,vbs,lpx,utc,aspd=None, bdif=None, adif=None):
    """lr = smooth_md(lr,vol,vbs,lpx,utc[-1,:])
    removes the inf/nan and 
    checks and warns the potential outliers, 
    lr - OT 30std
    vol - warning on 30std volume - mean
    lpx - warning on 30std simple return

    input:
        lr,vol,vbs,lpx,utc: shape [m,n] mdays, nbar
    return:
        lr: shape[m,n] log returns cleaned up
            make sure no warning for others
    
    """
    import Outliers as OT

    m,n=lr.shape
    lr0 = lr.copy()
    vol0 = vol.copy()
    vbs0 = vbs.copy()
    lpx0 = lpx.copy()

    aspd0 = aspd.copy() if aspd is not None else np.zeros((m,n))
    bdif0 = bdif.copy() if bdif is not None else np.zeros((m,n))
    adif0 = adif.copy() if adif is not None else np.zeros((m,n))
    val = [lr0, vbs0, vol0, lpx0, aspd0, bdif0, adif0]

    # remove nan/inf with 0
    for i, v in enumerate (val):
        ix = np.nonzero(np.isinf(v))
        if len(ix[0])>0:
            print('replacing %d inf for (%d)'%( len(ix[0]), i))
            v = 0
        ix = np.nonzero(np.isnan(v))
        if len(ix[0])>0:
            print('replacing %d nan for (%d)'%( len(ix[0]), i))
            v = 0
        val[i] = v

    # remove 30*sd as "wrong" value
    cnt = 2
    while cnt > 0:
        for i, v in enumerate(val[:1]):
            vm = np.mean(v,axis=0)
            vs = np.clip(np.std(v,axis=0),1e-10,1e+10)
            val[i] = OT.soft1(v-vm, vs, 30, 1)+vm
        cnt -= 1

    # need to regulate the overnight
    lr0 = val[0]
    lr00 = lr0[:,0]
    cnt = 2
    while cnt > 0:
        lr00m = np.mean(lr00)
        lr00 = OT.soft1(lr00-lr00m, np.std(lr00), 15, 1)+lr00m
        cnt -= 1
    val[0][:,0] = lr00

    # detect 100*sd vol
    vol0 = val[2]
    vm = np.mean(vol0,axis=0)
    vs = np.clip(np.std(vol0,axis=0),1,1e+10)
    ix = np.nonzero(np.abs(vol0-vm)-100*vs>0)
    if len(ix[0])>0:
        print('(%d) vol outliers (please check vbs also) at '%(len(ix[0])) + str(ix) + '\n' + str(vol0[ix]))
    else:
        print('vol/vbs good')

    # vbs check is included in the vol, so skipped here
    lpx0 = val[3]
    lpx0_ = lpx0.flatten()
    ixz = np.nonzero(np.abs(lpx0_) < 1e-10)[0]
    assert len(ixz) == 0, "zero price detected!"

    # simple return
    rtx0_ = np.r_[0, (lpx0_[1:]-lpx0_[:-1])/lpx0_[:-1]].reshape((m,n))
    vm = np.mean(rtx0_,axis=0)
    vs = np.std(rtx0_,axis=0)
    ix = np.nonzero(np.abs(rtx0_-vm)-100*vs>0)
    if len(ix[0])>0:
        print('(%d) lpx outliers at '%(len(ix[0])) + str(ix) + '\n' + str(lpx0[ix]))
        return ix
    else:
        print('lpx good')

    # check on the avg_spread and bid/ask diff
    for vx,name in zip(val[4:7], ['avg_spd','bid_diff','ask_diff']):
        vm = np.mean(vx,axis=0)
        vs = np.clip(np.std(vx,axis=0),1e-10,1e+10)
        ix = np.nonzero(np.abs(vx-vm)-30*vs>0)
        if len(ix[0])>0:
            print('(%d) %s outliers at '%(len(ix[0]),name) + str(ix) + '\n' + str(vol0[ix]))
        else:
            print('spd/bid_ask_diff good')

    return val[0]

def bar_agg_ix(bar, col_dict, ix) :
    """
    bar: shape [nday, nbar, ncol]
    col_dict: dict with key {'utc','lr','open','high','low','close','vol','vbs','lpx','bsz','asz','spd','bqd','aqd','opt_v1','opt_v2'}
    ix: the index at which the each aggegated bar should end at, (including)
        i.e. returned by vb_vol() 
        ix=0: just use the first bar
    """
    ix = np.array(ix).astype(int)
    assert np.min(ix[1:]-ix[:-1])>=1

    m,n,cols=bar.shape
    assert cols == len(col_dict.keys())
    arr=[]

    for kn in col_dict.keys(): 
        # aggregate: lr, vol, vbs
        if kn in ['lr','vol','vbs','bqd','aqd','opt_v1','opt_v2']:
            k = col_dict[kn]
            v = bar[:,:,k]
            vc = np.vstack((np.zeros(m),np.cumsum(v,axis=1)[:,ix].T)).T
            arr.append((vc[:,1:]-vc[:,:-1]).flatten())
            continue

        # just pick the latest
        if kn in ['utc','close','lpx']:
            k = col_dict[kn]
            arr.append(bar[:,ix,k].flatten())
            continue

        # just get an average
        ixd = np.r_[ix[0]+1, ix[1:]-ix[:-1]]
        assert (np.min(ixd)>0)
        if kn in ['spd', 'bsz','asz']:
            k = col_dict[kn]
            v = bar[:,:,k]
            vc = np.vstack((np.zeros(m),np.cumsum(v,axis=1)[:,ix].T)).T
            arr.append(((vc[:,1:]-vc[:,:-1])/ixd).flatten())
            continue

        # just get the first
        if kn in ['open']:
            k = col_dict[kn]
            ix0=np.r_[0,ix[:-1]+1]
            arr.append(bar[:,ix0,k].flatten())
            continue

        # just get a max/min
        if kn in ['high','low']:
            k = col_dict[kn]
            fn=np.max if kn == 'high' else np.min
            v = bar[:,:,k]
            arr0=[]
            for ix0, ix1 in zip(np.r_[0,ix[:-1]+1], ix+1):
                arr0.append(fn(v[:,ix0:ix1],axis=1))
            arr.append(np.array(arr0).T.flatten())
            continue

        assert 0==1, '%s not found!'%(kn)

    # enforce the same order as input
    col_ix=np.argsort([col_dict[k] for k in col_dict.keys()])
    return np.array(arr).T.reshape((m,len(ix),cols))[:,:,col_ix]

##############################################################
# pyres md_dict procedure
# 
# get raw md_dict from ar1_md, say 30 second bar
# 1. clean the data
# 2. adjust the columns to be [utc, lr, vol, vbs, lpx]
# 3. save to the md_pyres as a cleaned format of a single md
# 
# with the cleaned up 30 second bar, do a bucket using a chosen
# indicator, could be the primary lr, or the vbs of other symbol.
# 1. run:    ix, lr = def vb_vol(lr, m_out, min_bars=2)
# 2. repeat 1, so that number of bars is total_sec/barsec 
#    i.e. 276 for barsec = 300, 184 for barsec=450, 
#    138 for barsec=600, etc
# 3. aggregate all symbols in md_dict using the given bucket ix
#
# Finally call xobj_vd with md_dict as md_dict_in
###############################################################

def get_raw_md_dict(mts_symbol, start_day, end_day, barsec=30, extended_cols=True):
    """
    copied from ar1_md.md_dict_from_mts() with slight change:
    1. return cols including spd+bdif+adif
    2. default parameters
    return:
        bar: shape(ndays,n,8)) for [utc, lr, vol, vbs, lpx, aspd, bdif, adif]
        Note both the lr and lpx are roll adjusted
    """
    import ar1_md
    import mts_repo

    #sym = ar1_md.mts_repo_mapping[symbol]
    sym = mts_symbol
    if extended_cols:
        cols=['open', 'close', 'vol','vbs','lpx','utc','aspd','bdif','adif']
    else:
        cols=['open', 'close', 'vol','vbs','lpx','utc']
    open_col = np.nonzero(np.array(cols) == 'open')[0][0]
    close_col = np.nonzero(np.array(cols) == 'close')[0][0]
    vol_col = np.nonzero(np.array(cols) == 'vol')[0][0]
    vbs_col = np.nonzero(np.array(cols) == 'vbs')[0][0]
    lpx_col = np.nonzero(np.array(cols) == 'lpx')[0][0]
    utc_col = np.nonzero(np.array(cols) == 'utc')[0][0]
    if extended_cols:
        aspd_col = np.nonzero(np.array(cols) == 'aspd')[0][0]
        bdif_col = np.nonzero(np.array(cols) == 'bdif')[0][0]
        adif_col = np.nonzero(np.array(cols) == 'adif')[0][0]

    bars, roll_adj_dict = ar1_md.md_dict_from_mts_col(sym, start_day, end_day, barsec, cols=cols, use_live_repo=False, get_roll_adj=True)
    bars=mts_repo.MTS_REPO.roll_adj(bars, utc_col, [lpx_col], roll_adj_dict)

    # construct md_days
    if extended_cols:
        cols_ret = ['utc','lr','vol','vbs','lpx','aspd','bdif','adif']
    else:
        cols_ret = ['utc','lr','vol','vbs','lpx']

    ndays, n, cnt = bars.shape
    bars=bars.reshape((ndays*n,cnt))

    # use the close/open as lr
    # for over-night lr, repo adjusts first bar's open as previous day's close
    lr = np.log(bars[:,close_col]/bars[:,open_col])

    if extended_cols:
        bars = np.vstack((bars[:,utc_col], lr, \
            bars[:,vol_col], bars[:,vbs_col], bars[:,lpx_col],
            bars[:,aspd_col], bars[:,bdif_col], bars[:,adif_col])).T.reshape((ndays,n,8))
    else:
        bars = np.vstack((bars[:,utc_col], lr, \
            bars[:,vol_col], bars[:,vbs_col], bars[:,lpx_col])).T.reshape((ndays,n,8))

    return bars, cols_ret

############
# this may take some time
# check the data carefully to remove missing/wrong/abnormal data
#############
def clean_bars(bars):
    m,n,col_cnt = bars.shape
    utc = bars[:,:,0]
    lr = bars[:,:,1]
    vol = bars[:,:,2]
    vbs = bars[:,:,3]
    lpx = bars[:,:,4]
    if col_cnt > 5:
        aspd = bars[:,:,5]
        bdif = bars[:,:,6]
        adif = bars[:,:,7]
    else:
        aspd = None
        bdif = None
        adif = None

    # check for the warning message
    lr0 = clean_md(lr,vol,vbs,lpx,utc,aspd=aspd, bdif=bdif, adif=adif)

    return lr0

###################
# play with the vb_vol using the lr 
# lr0 = clean_bars(cl)
# ix, lrn = vb.vb_vol(lr, 400, 3) # i.e. could get 276 bars
# 
# md_dict = def md_agg_ix(md_dict, ix)
#
# build it into xobj
####################

