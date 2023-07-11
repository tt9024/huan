import numpy as np
import pandas
import datetime
import os

#############
# utilities 
#############

COL_Base = {'utc':0,'open':1,'high':2,'low':3,'close':4,'vol':5,'lpx':6,'ltm':7,'vbs':8}
COL_Ext = {'bsz':9, 'asz':10, 'spd':11, 'bqd':12, 'aqd':13}
COL_Opt = {'opt_v1':14, 'opt_v2':15}
MTS_BAR_COL = dict(**COL_Base, **COL_Ext, **COL_Opt) # adding extended fileds
def get_default_cols(m):
    if m == 9:
        return COL_Base
    if m == 14:
        return dict(**COL_Base, **COL_Ext)
    if m == 16:
        return dict(**COL_Base, **COL_Ext, **COL_Opt)
    raise RuntimeError('unknown default colume number %d'%(m))
def get_col_ix(col_name_array) :
    cols = []
    for c in col_name_array:
        cols.append(MTS_BAR_COL[c])
    return np.array(cols).astype(int)

def mergeBar(MtsBar, barsec) :
    """
    MtsBar shape supposed to be 1-second bars with shape [nsecs, mcols]
    base_columes: {'BarTime':0, 'Open':1, 'High':2, 'Low':3, 'Close':4, 'TotalVolume':5, 'LastPx':6, 'LastPxTime':7, 'VolumImbalance':8}
    ext_colums: 'avg_bsz', 'avg_asz', 'avg_spd', 'tot_bsz_dif', 'tot_asz_diff'
    opt_colums: 'opt_v1', 'opt_v2'
    """
    n,m = MtsBar.shape
    assert(n//barsec*barsec==n), 'barsec not a multiple of total bars'
    assert(MtsBar[-1,0]-MtsBar[0,0]==n-1), 'MtsBar not in 1-second period'
    ix = np.arange(0,n,barsec)
    ## getting the ohlc
    oix = np.arange(0,n,barsec)
    cix = np.r_[oix[1:]-1,n-1]
    bt=MtsBar[cix,0]
    o=MtsBar[oix,1]
    h=np.max(MtsBar[:,2].reshape(n//barsec,barsec),axis=1)
    l=np.min(MtsBar[:,3].reshape(n//barsec,barsec),axis=1)
    c=MtsBar[cix,4]

    tvc=np.cumsum(MtsBar[:,5])
    tv=tvc[cix]-np.r_[0,tvc[cix[:-1]]]
    lpx=MtsBar[cix,6]
    lpxt=MtsBar[cix,7]
    tvbc=np.cumsum(MtsBar[:,8])
    tvb=tvbc[cix]-np.r_[0,tvbc[cix[:-1]]]

    flds = np.vstack((bt,o,h,l,c,tv,lpx,lpxt,tvb)).T
    m_base = flds.shape[1]
    if m > m_base :
        # extended
        # 'avg_bsz', 'avg_asz', 'avg_spd', 'tot_bsz_dif', 'tot_asz_diff'
        m_ext = 5
        assert m >= m_base + m_ext, 'wrong extended field number'
        eflds = MtsBar[:,m_base:m_base+m_ext].T.reshape((m_ext,n//barsec,barsec))
        # get avg for the first 3 fields
        eflds0 = np.mean(eflds[:3,:,:],axis=2)
        # get sum for the tot_diff
        eflds1 = np.sum(eflds[3:,:,:],axis=2)
        flds = np.hstack((flds,eflds0.T,eflds1.T))
        if m > m_base + m_ext:
            m_opt = 2
            assert m == m_base + m_ext + m_opt, 'wrong optional field number'
            optflds = MtsBar[:,m_base+m_ext:m_base+m_ext+m_opt].T.reshape((m_opt,n//barsec,barsec))
            # get sum for them
            optflds1 = np.sum(optflds[:,:,:],axis=2)
            flds = np.hstack((flds,optflds1.T))

    return flds

def daily1s(bar_1sec, sutc, eutc, cols=None, backward_fill=False, allow_non_increasing=False, min_bars = 2):
    """
    normalize a day worth of bar_1sec into (eutc-sutc) bars, each close at utc as 
    'np.arange(sutc,eutc)+1', invalid removed, forward (backward) filled. 
    input: 
        bar_1sec: shape (n,m) 1-second bars, with m columes defined in cols
        stuc, eutc: start and end time, first bar close at sutc+1, last close at eutc
        cols: a list of column names defined in MTS_BAR_COL
              len(cols)==m, and cols[k] is for bar[:,k]
        backward_fill:  shouldn't be true, unless for the first old day of a 20 years bar
                        put there for remove 0 prices
        allow_non_increasing: bar0 has non-increasing time columne, possible
                              for live data
        min_bars: minimum number of bars to take as valid input
    return: 
        1 second bar, first bar at sutc+1, last bar at eutc
        forward (and backward_fill) filled on any missing

    usage cases:
        1. for writing bars from mts live csv file: may have gap/invalid/non-increasing
        2. for reading repo bars, normalize hours, remove invalid, and forward fill gaps.
           The first bar's px should set properly before this call.
        
    Note: 
    1. requires a first bar be on or earlier than sutc+1 and is valid
       this is true for getting from mts_live csv files,
       but may not be true for tickdata. repo's get_bars() should 
       fill the first bar before this call. backward_fill is allowed for
       rare cases on non-price columes, such as bbo size/spd.
    2. the valid bar:
       * 'bsz','asz','spd' > 0 if exists
       * ohlc 0
       * lpx,ltm only forward fill
    3. first bars can be invalid, and not filled if backward_fill
       is false.  This can be fixed at data level

    Most of the complexity comes from the freedom of cols. It normalizes any
    combinations of cols, including stateful cols such as ohlc, and their forward
    and backward fills.  This is intended to be used by both repo and by model
    """
    n,m=bar_1sec.shape
    if cols is None:
        cols = get_default_cols(m)
    else:
        cdict = {}
        for i, c in enumerate(cols):
            cdict[c]=i
        cols=cdict
    for k in cols.keys():
        assert k in MTS_BAR_COL.keys(), "normalize_ref: unknown col %s"%(k)
    assert 'utc' in cols.keys(), "normalize_ref: utc not in bar's colume"
    ref_utc = np.arange(sutc, eutc).astype(int)+1
    bar = bar_1sec.copy()

    # butc could be really messy:
    # * have duplicates and goes back
    # * later than stuc,
    # * have gaps or invalid bars (0 px, -sz)
    # * too few bars within ref_utc
    non_increase_cnt = 0
    while True:
        butc = bar[:,cols['utc']].astype(int)
        butc_diff = butc[1:]-butc[:-1]
        ix = np.nonzero(butc_diff<=0)[0]
        if len(ix) > 0:
            non_increase_cnt += len(ix)
            bar = np.delete(bar, ix, axis=0)
        else:
            break
    if non_increase_cnt > 0:
        print('%d Time Loopback!'%(non_increase_cnt))
        if not allow_non_increasing:
            raise RuntimeError('%d non-increaing bars found!'%(non_increase_cnt))

    # check min bars
    ix0 = np.searchsorted(butc, sutc+0.5) # ix on or before ref_utc[0]
    ix1 = np.searchsorted(butc, eutc+0.5) # ix after the last bar less/equal to eutc
    assert ix1-ix0>=min_bars, "No enough bars recorded in Live, " + str(ix1-ix0)

    # populate the dbar with existing bar at bidx
    # note this works across day
    butc = butc[ix0:ix1]
    bidx = np.searchsorted(ref_utc, butc) # this should match
    dbar = np.empty((len(ref_utc),m))
    dbar[:,:] = np.nan
    dbar[bidx,:] = bar[ix0:ix1,:].copy()

    # check some invalid values and set them to nan to
    # 1. the zero price fields
    for c in ['lpx', 'ltm', 'open', 'high', 'low', 'close']:
        if c in cols.keys():
            c0 = cols[c]
            ix = np.nonzero(dbar[:,c0] == 0)[0]
            dbar[ix,c0] = np.nan

    # 2. the zero or negative bsz/asz/spd
    for c in ['bsz', 'asz', 'spd']:
        if c in cols.keys():
            c0 = cols[c]
            ix = np.nonzero(dbar[:,c0] <= 0)[0]
            dbar[ix,c0] = np.nan

    # start forward filling
    ixnan = np.nonzero(np.isnan(dbar))[0]
    if len(ixnan) > 0 :
        #print ("found " + str(len(ixnan)) + " missing/bad values")
        # in case the first bar has nan and there are previous bars, use it
        ixnan0 = np.nonzero(np.isnan(dbar[0,:]))[0]
        if len(ixnan0) > 0 :
            #print ("missing first bar on %s"%(str(datetime.datetime.fromtimestamp(sutc))))
            if ix0>0:
                # forward fill lpx/sizes in bar_1s upto ix0
                ff_cols=[]
                for c0 in ['lpx','ltm','bsz','asz','spd']:
                    if c0 in cols.keys():
                        ff_cols.append(c0)
                if len(ff_cols)>0:
                    ff_cols=get_col_ix(ff_cols)
                    dbar1=bar[:ix0,ff_cols].copy()
                    zix=np.nonzero(dbar1==0)
                    dbar1[zix]=np.nan
                    # forward fill bar[:ix0+1
                    df=pandas.DataFrame(dbar1)
                    df.fillna(method='ffill',axis=1,inplace=True)
                    dbar[0,ff_cols]=dbar1[-1,:]

                # forward fill ohlc upto ix0-1
                last_bar=bar[ix0-1,:].copy()
                ff_cols=[]
                for c0 in ['open','high','low','close','lpx']:
                    if c0 in cols.keys():
                        ff_cols.append(c0)
                if len(ff_cols)>0:
                    ff_cols=get_col_ix(ff_cols)
                    dbar1=bar[:ix0,ff_cols].copy()
                    zix=np.nonzero(dbar1==0)
                    dbar1[zix]=np.nan
                    df=pandas.DataFrame(dbar1)
                    df.fillna(method='ffill',axis=1,inplace=True)
                    last_bar[ff_cols]=dbar1[-1,:]

                prev_px=np.nan
                if 'open' in cols.keys() and not np.isnan(dbar[0,cols['open']]):
                    prev_px=dbar[0,cols['open']]
                else:
                    for c0 in ['close','lpx','open','high','low']:
                        if c0 in cols.keys():
                            px0=last_bar[cols[c0]]
                            if not np.isnan(px0):
                                prev_px=px0
                                break
                if not np.isnan(prev_px):
                    for c0 in ['open','close','lpx']:
                        if c0 in cols.keys():
                            if np.isnan(dbar[0,cols[c0]]):
                                dbar[0,cols[c0]] = prev_px
                #else:
                #    print('cannot get previous price when first bar has nan')
                """
                for c0 in ['vol','ltm','vbs','bqd','aqd','opt_v1','opt_v2']:
                    if c0 in cols.keys():
                        dbar[0,cols[c0]] = 0
                """
            else:
                # try to fix the lpx
                if 'lpx' in cols.keys() and np.isnan(dbar[0,cols['lpx']]):
                    # try to handle first lpx 0
                    for c in ['open','close']:
                        if c in cols.keys() and not np.isnan(dbar[0,cols[c]]):
                            dbar[0,cols['lpx']] = dbar[0,cols[c]]
                            if 'ltm' in cols.keys():
                                dbar[0,cols['ltm']] = 0
                            break
                print ("first bars has nan on %s"%(str(ixnan0)))

        # forward fill missing bars
        # 1. fix the ohlc by cross reference close/open, then derive high/low
        if 'close' in cols.keys():
            cc = cols['close']
            ixnan = np.nonzero(np.isnan(dbar[:,cc]))[0]
            if len(ixnan) > 0:
                # check if next open is valid
                if 'open' in cols.keys():
                    co = cols['open']
                    dbar[ixnan,cc] = np.r_[dbar[1:,co], dbar[-1,cc]][ixnan]
                # forward fill close
                df = pandas.DataFrame(dbar[:,cc])
                df.fillna(method='ffill',axis=0,inplace=True)
        if 'open' in cols.keys():
            co = cols['open']
            if 'close' in cols.keys():
                cc = cols['close']
                dbar[1:,co] = dbar[:-1,cc]
            ixnan = np.nonzero(np.isnan(dbar[:,co]))[0]
            if len(ixnan) > 0:
                # forward fill open
                df = pandas.DataFrame(dbar[:,cols['open']])
                df.fillna(method='ffill',axis=0,inplace=True)

        # 2. with open/close being populated, if exist, set high/low
        # trick is to allow for any or all those cols could not be included
        for c, func in zip(['high','low'],[np.max,np.min]):
            if c in cols.keys():
                da = np.tile(dbar[:,cols[c]].copy(),(3,1))
                for c0,ix0 in zip(['open','close'],[0,1]):
                    if c0 in cols.keys():
                        da[ix0,:] = dbar[:,cols[c0]].copy()
                df = pandas.DataFrame(da)
                df.fillna(method='ffill',axis=0,inplace=True)
                #df.fillna(method='ffill',axis=1,inplace=True)
                df.fillna(method='bfill',axis=0,inplace=True) # just remove nan for min()/max()
                dbar[:,cols[c]] = func(da,axis=0)

        for fcol in ['lpx','ltm', 'bsz', 'asz', 'spd']:
            if fcol in cols.keys():
                df = pandas.DataFrame(dbar[:,cols[fcol]])
                df.fillna(method='ffill',axis=0,inplace=True)

        # backward fill bsz/asz/spd
        for c0 in ['bsz', 'asz', 'spd']:
            if c0 in cols.keys():
                df = pandas.DataFrame(dbar[:,cols[c0]])
                df.fillna(method='bfill',axis=0,inplace=True)

        # backward fill price if allowed to
        ixnan = np.nonzero(np.isnan(dbar[0,:]))[0]
        if len(ixnan)>0:
            if not backward_fill:
                # check price has nan
                for c0 in ['open','high','low','close','lpx']:
                    if c0 in cols.keys() and np.isnan(dbar[0,cols[c0]]):
                        raise RuntimeError('initial invalid bars not backward filled!')

            # do the backward fill as asked to - lpx/ltm first
            for c0 in ['lpx','ltm']:
                if c0 in cols.keys():
                    df = pandas.DataFrame(dbar[:,cols[c0]])
                    df.fillna(method='bfill',axis=0,inplace=True)

            # use 'open' if exist
            for fill_col in ['open','lpx','close','high','low','no_fill_cols']:
                if fill_col in cols.keys(): 
                    break
            if fill_col in cols.keys():
                cf = cols[fill_col]
                df = pandas.DataFrame(dbar[:,cf])
                df.fillna(method='bfill',axis=0,inplace=True)
                for c0 in ['open','high','low','close']:
                    if c0 not in cols.keys():
                        continue
                    ixnan = np.nonzero(np.isnan(dbar[:,cols[c0]]))[0]
                    dbar[ixnan,cols[c0]] = dbar[ixnan,cf]

        # 3. fill in zeros for any nan
        dbar[np.isnan(dbar)]=0

    # 5. normalize the utc
    dbar[:,cols['utc']] = ref_utc
    return dbar

def crop_bars(bar, cols, barsec, fail_check=False) :
    """ make all daily bars in 'bar' same length, taking a minimum of all days
        no fill is performed, just crop.  see normalize() for backfill/forward_fill
        This is mainly for last check on in case the daily bars from repo has different
        shape, i.e. LCO has more bars on Sunday, and the ref_utc was not given.
        input:
            bar: shape [nday, nbar, mcol] multiday bars
            cols: list of string as column names
    """
    bcnts = []
    for b0 in bar:
        bcnts.append(b0.shape[0])
    bset = set(bcnts)
    if len(bset) > 1 :
        if fail_check:
            raise 'different bar counts on different days!'
        print('different bar counts on different days, taking a smallest one %s'%(str(bset)))
        bc = min(bset)
        bix = np.nonzero(np.array(bcnts).astype(int)==bc)[0][0]
        if cols is None:
            cols = get_default_cols().keys()
        utc_col = np.nonzero(np.array(cols) == 'utc')[0]
        if len(utc_col) != 1:
            raise RuntimeError('cannot crop bars with different daily counts: utc not found cols')
        ucol = utc_col[0]
        utc = bar[bix][:, ucol]

        # crop bars with minimum barcounts, and no later than 5pm 
        # LCO had old bars from 18 - 17, then the new format has either 
        # 19-18 (sunday open) or 20-18.  So have 3 bar counts to deal with
        # all bars then use a minimum of 20 - 17. 
        # 
        # We enforce a utc_start and utc_end on each day, assuming
        # barsec are strictly enforced, i.e. no skipped bars
        day = datetime.datetime.fromtimestamp(utc[-1]).strftime('%Y%m%d')
        utc0 = int(datetime.datetime.strptime(day+'000000','%Y%m%d%H%M%S').strftime('%s'))
        utc1 = int(datetime.datetime.strptime(day+'170000','%Y%m%d%H%M%S').strftime('%s'))
        soffset = utc[0]-utc0
        eoffset = min(utc1,utc[-1])-utc0
        bc = (eoffset-soffset)//barsec+1
        assert (bc-1)*barsec+soffset==eoffset
        utc=np.arange(utc0+soffset,utc0+eoffset+barsec,barsec).astype(int)

        bars = []
        for b0 in bar:
            if b0.shape[0] != bc:
                utc_b = b0[:,ucol]
                day = datetime.datetime.fromtimestamp(utc_b[-1]).strftime('%Y%m%d')
                utc0 = int(datetime.datetime.strptime(day+'000000','%Y%m%d%H%M%S').strftime('%s'))
                utc1 = np.arange(utc0+soffset,utc0+eoffset+barsec,barsec).astype(int)
                uix = np.searchsorted(utc_b, utc1)
                if np.max(np.abs(utc_b[uix]-utc1)) > 0:
                    raise RuntimeError('cannot crop bars with different daily counts: incompatible bars')
                b0 = b0[uix,:]
            bars.append(b0)
        bar = bars
    return np.array(bar)

def saveCSV(bar, file_name, do_gzip = True):
    assert (len(bar.shape)==2), 'bar needs to be a 2-dimensional array'
    assert bar.shape[1] in [9, 14, 16], 'unknown bar shape!'
    fmt = ['%d','%.8g','%.8g','%.8g','%.8g','%d','%.8g','%d','%d']
    if bar.shape[1] >= 14:
        fmt += ['%.5g', '%.5g', '%.8f', '%d','%d']
        if bar.shape[1] > 14:
            fmt += ['%d','%d']
    np.savetxt(file_name, bar, delimiter = ',', fmt = fmt)
    if do_gzip :
        os.system('gzip -f ' + file_name)


############
# plotting
############

def plot_bar_multiday(bar, fig):
    """
    bar: shape [ndays, nbars, ncols]
         assuming bar has full columns of MTS_BAR_COL
    """

    d,n,m=bar.shape
    assert m==len(MTS_BAR_COL.keys()), 'bar has to have all MTS_BAR_COL'
    bar0 = bar.reshape((d*n,m))
    utc,close,vbs,bsz,asz,spd,bqd,aqd,swp,ibg = get_col_ix(['utc','close','vbs','bsz','asz','spd','bqd','aqd','opt_v1','opt_v2'])
    dt=[]
    for t in bar[:,:,utc].flatten():
        dt.append(datetime.datetime.fromtimestamp(t))
    dt=np.array(dt)

    # ax1: the close price and the spread
    ax1=fig.add_subplot(4,1,1)
    ax1.plot(dt, bar0[:,close], label='close')
    ax1t=ax1.twinx()
    ax1t.plot(dt,bar0[:,spd],'y-.',label='avg_spd')

    #ax2: the vbs and qbs
    ax2=fig.add_subplot(4,1,2,sharex=ax1)
    ax2.plot(dt,np.cumsum(bar0[:,vbs]),label='vbs')
    ax2t=ax2.twinx()
    ax2t.plot(dt,np.cumsum(bar0[:,bqd]-bar0[:,aqd]),'y-',label='qbs')

    #ax3: the bqd and aqd and bsz, asz
    ax3=fig.add_subplot(4,1,3,sharex=ax1)
    ax3.plot(dt, bar0[:,bqd],label='bqd')
    ax3.plot(dt, bar0[:,aqd],label='aqd')
    ax3t=ax3.twinx()
    ax3t.plot(dt,bar0[:,bsz],'r-',label='bsz')
    ax3t.plot(dt,bar0[:,asz],'g-',label='asz')

    #ax4: the swp and ibg
    ax4=fig.add_subplot(4,1,4,sharex=ax1)
    ax4.plot(dt, bar0[:,swp],label='swp')
    ax4.plot(dt, bar0[:,ibg],label='ibg')
    ax4t=ax4.twinx()
    ax4t.plot(dt,np.cumsum(bar0[:,swp]),'r-',label='cs_swp')
    ax4t.plot(dt,np.cumsum(bar0[:,ibg]),'g-',label='cs_ibg')

    axs = [ax1,ax2,ax3,ax4,ax1t,ax2t,ax3t,ax4t]
    for ax in axs[:4]:
        ax.grid()
        ax.legend(loc='upper left')
    for ax in axs[4:]:
        ax.legend(loc='upper right')
    return axs

