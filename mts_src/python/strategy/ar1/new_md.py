import mts_repo
import mts_liquidity as ml
import Outliers as OT
import os
import subprocess
import copy
import dill
import gc
import numpy as np
import datetime
import td2mts
import pandas

"""
Energy_Symbols = ['WTI','Brent','NG','RBOB', 'Gasoil', 'HO']
Metal_Symbols = ['Gold', 'Silver', 'Platinum', 'Palladium', 'HGCopper']
Agri_Symbols = ['Corn', 'Wheat', 'Soybeans', 'SoybeanMeal', 'SoybeanOil']
LiveStock_Symbols = ['LiveCattle', 'LeanHogs', 'FeederCattle']
Softs_Symbols = ['Sugar', 'Cotton', 'Cocoa']

EquityUS_Symbols = ['SPX', 'NDX', 'Russell', 'DJIA']
#EquityEU_Symbols = ['EuroStoxx', 'EuroStoxx600', 'FTSE', 'DAX', 'CAC']
EquityEU_Symbols = ['EuroStoxx', 'EuroStoxx600', 'DAX', 'CAC', 'FTSE'] # FTSE needs to be fixed.

RatesUS_Symbols = ['TU',     'FV',    'TY',   'US']
RatesEU_Symbols = ['Schatz', 'BOBL',  'Bund', 'BUXL', 'OAT', 'Gilt']

FXFuture_Symbols = ['EUR', 'JPY', 'AUD', 'GBP', 'NZD', 'CAD', 'CHF', 'ZAR', 'MXN', 'BRL']
VIX_Symbols = ['VIX', 'V2X']

Comm_Symbols = Energy_Symbols + Metal_Symbols + Agri_Symbols + LiveStock_Symbols + Softs_Symbols
Eqt_Symbols = EquityUS_Symbols + EquityEU_Symbols 
Rate_Symbols = RatesUS_Symbols + RatesEU_Symbols
FX_Symbols = FXFuture_Symbols
"""

###
# ingesting td_dev
###
class TD_Dev:
    def __init__(self, symbol_map_obj=None):
        self.live_path = mts_repo.MTSRepoPath
        self.td_dev_path=os.path.join('/'.join(self.live_path.split('/')[:-1]), 'td_dev')
        self.repo = \
            mts_repo.MTS_REPO(self.td_dev_path,\
                              symbol_map_obj=symbol_map_obj,\
                              backup_repo_path=self.live_path)

    @staticmethod
    def run_td_dev(yymm, overwrite_repo=False, start_end_days=(), symbol_list=None):
        """
        yymm: can be a year, meaning all months
        start_end_days: given as (sday, eday), each in YYYYMMDD,
                        when yymm is a month, the same month of yymm
        symbol_list: if not None, only run the given symbols
        """
        # adjust the yyyymm
        ymstr = str(yymm)
        if len(ymstr) == 6:
            ym = [ymstr]
        elif len(ymstr) == 4:
            assert len(start_end_days) == 0, 'no start_end_days on a year'
            ym = []
            for i in np.arange(12).astype(int)+1:
                ym.append('%s%02d'%(ymstr,i))
        else:
            raise "unknown yymm " + ymstr

        # eqt
        eqty = {'sym': ml.EquityUS_Symbols + ml.EquityEU_Symbols, \
                'param': {'maxn':2, 'extra_mon':[12]}}
        # agr
        agr = {'sym': ml.Agri_Symbols + ml.LiveStock_Symbols + ml.Softs_Symbols, \
               'param': {'maxn':6, 'extra_mon':[6,8,10,12]}}

        # vix
        vix = {'sym': ml.VIX_Symbols, \
               'param': {'maxn':6, 'extra_mon':[7,9,12]}}

        # comm
        comm = {'sym': ml.Energy_Symbols + ml.Metal_Symbols, \
                'param': {'maxn':3, 'extra_mon':[6,12]}}

        # rates
        rates = {'sym': ml.RatesUS_Symbols + ml.RatesEU_Symbols + ml.FXFuture_Symbols, \
                 'param': {'maxn':2, 'extra_mon':[12]}}

        for sd in [eqty, agr, vix, comm, rates]:
        #Ifor sd in [comm]:
            s = sd['sym']
            p = sd['param']
            for s0 in s:
                if symbol_list is not None and s0 not in symbol_list:
                    continue
                for ym0 in ym:
                    print('running %s for %s'%(ym0, s0))
                    td2mts.run_month(s0, ym0, p['maxn'], extended_fields=True, overwrite_repo=overwrite_repo, extra_N=p['extra_mon'], write_optional=True, start_end_days=start_end_days)

########################
# be aware of the size
# usually 2 year 1S maximum.
########################
def collect_raw(dump_path = '/tmp/md_pyres', \
                nlist = [0,1,2,3,6], \
                symbols = ['WTI'], \
                sday = '20160101', eday = '20230223', barsec = 10, \
                base_cols = ['utc','open','close','vol','vbs','lpx', 'bqd','aqd','opt_v1','opt_v2','bsz','asz','spd']):
    """
    this just gets some representative symbols to study the vbs/qbs/opt_v/avg_sz/spd,
    also the contributions from spreads

    dump a dict with key {'md': {n: bar[ndays,nbar,ncol]}, 'ra': {n: holiroll}, 
                          'col_dict', 'barsec'}

    Note 1: the barsec is 10, years is 7, so a lot of data, it just dump to dill.sz for each symbol
    Note 2: gets upto 3 and plus 6, 12
    """
    td = TD_Dev()
    rd = td.repo
    dump_path0 = os.path.join(dump_path, 'collect_raw')
    try:
        os.system('mkdir -p '+ dump_path0)
    except Exception as e:
        print(str(e)+'\ncannot create dump path ' + dump_path0)
        return

    col_dict={}
    for i,c in enumerate(base_cols):
        col_dict[c]=i

    for s in symbols:
        md = {}
        ra = {}
        """
        try:
            bar_n1, holiroll = rd.get_bars(s+'_N1',       sday, eday, barsec=barsec, get_roll_adj=True, cols=base_cols, is_mts_symbol=True)
            md[1] = bar_n1.copy()
            ra[1] = copy.deepcopy(holiroll)
        except Exception as e:
            print(str(e))
            continue
        """
        for n in nlist:
            try:
                bar_nn, holiroll = rd.get_bars(s+'_N'+str(n), sday, eday, barsec=barsec, get_roll_adj=True, cols = base_cols, is_mts_symbol=True)
                md[n] = bar_nn.copy()
                ra[n] = copy.deepcopy(holiroll)
            except Exception as e:
                print(str(e))
        try:
            nstr = ''.join(np.array(nlist).astype(str))
            cstr = (''.join(base_cols)).replace('_','')
            fname = os.path.join(dump_path0, '%s_%s_%s_%sS_N%s_C%s.dill'%(s,sday,eday,str(barsec),nstr,cstr))
            with open(fname, 'wb') as fp:
                dill.dump({'md':md, 'ra': ra,'cols':col_dict,'barsec':barsec}, fp)
            os.system('gzip -f ' + fname)
        except Exception as e:
            print('problem saving to dump ' + fname)
            break

        md = None ; ra = None
        gc.collect()

    # return the last file name
    return fname

def vstack_md(_N, md_list=None, md_fn_list=None, col_list=None, bar_ix01=None):
    """
        bar, cols, ra_dict=vstack_md(1, [md0, md1, md2], col_list=['utc','close','vol','vbs','spd'])

    vstack huge, i.e. 7 years of 1S bar, from 3 md files, using _N contract.
    return bar from vstack md0['md'][_N], md1['md'][_N] and md2['md'][_N], and
    ra_dict from md0['ra'][_N], etc

    md_fn_list: if given, will dill load, save some memory
    col_list: if given, only return selected columes, save some memory
    bar_ix01: pair of (ix0, ix1), if given, get bars [ix0:ix1), ix1 exclusive, save memory

    Note the time in md_list assumed to be strictly increase, 
    ra_adj days has to be strictly increasing as well.
    """
    def merge_ra_dict(rd_list):
        days=[]
        contracts=[]
        roll_adj=[]
        for rd in rd_list:
            days+= rd['days']
            contracts+=rd['contracts']
            roll_adj+=rd['roll_adj']
        ix=np.argsort(days)
        days=np.array(days)[ix]
        ixz=np.nonzero(days.astype(int)[1:]-days.astype(int)[:-1]<=0)[0]
        assert len(ixz)==0, 'ra_list days non-increasing: '+ str(days[ixz])
        contracts=np.array(contracts)[ix]
        roll_adj=np.array(roll_adj)[ix]
        ra_dict={'days':list(days), 'contracts':list(contracts),'roll_adj':list(roll_adj)}
        return ra_dict

    def get_cols(cols_dict, col_list):
        c=[]
        for c0 in col_list:
            c.append(cols_dict[c0])
        return np.array(c).astype(int)

    mdn=[]
    rdn=[]
    if md_fn_list is not None:
        assert md_list is None
        md_list = [None]*len(md_fn_list)
    else:
        assert md_fn_list is None
        md_fn_list=[None]*len(md_list)
    nbars=None
    mdays=[]
    for i, (md, fn) in enumerate(zip(md_list, md_fn_list)):
        if fn is not None:
            md=dill.load(open(fn,'rb'))
        try:
            if col_list is None:
                col_list=md['cols']
            cix=get_cols(md['cols'],col_list)
        except:
            print('problem getting col_list %s from md index %d'%(str(col_list), i))
            return None, None, None
        nd,nb,nc=md['md'][_N].shape
        if nbars is None:
            nbars=nb
        else:
            assert nb==nbars
        if bar_ix01 is None:
            bar_ix01=(0,nbars)
        ix0,ix1=bar_ix01
        mdn.append(md['md'][_N][:,ix0:ix1,cix])
        rdn.append(md['ra'][_N])
        for t in md['md'][_N][:,-1,md['cols']['utc']]:
            mdays.append(datetime.datetime.fromtimestamp(t).strftime('%Y%m%d'))
    cols = {}
    for i,c in enumerate(col_list):
        cols[c]=i

    md=None #save memory
    mdn=np.vstack(mdn)
    nd,nb,nc=mdn.shape
    utc=mdn[:,:,cols['utc']].flatten()
    ixz=np.nonzero(utc[1:]-utc[:-1]<=0)[0]
    assert len(ixz)==0, 'non-increasing utc detected: '+str(utc[ixz])
    ra_dict=merge_ra_dict(rdn)
    if len(ra_dict['days'])!=nd:
        mdays=np.array(mdays)
        rdays=np.array(ra_dict['days'])
        ix=np.clip(np.searchsorted(mdays, rdays),0,nd-1)
        ixnz=np.nonzero(mdays[ix]!=rdays)[0]
        if len(ixnz)>0:
            print('days in md and ra_dict disagree: utc_days:%s, ra_dict_days:%s, ra_adj:%s'%(\
                    str(mdays[ix][ixnz]), str(rdays[ixnz]), \
                    str(np.array(ra_dict['roll_adj'])[ixnz])))
    return mdn, cols, ra_dict

def merge_n(bar_in, cols_dict, bar_n_list, cols_dict_n, in_place=True):
    """
    merge the md_n onto bar_in(usually n1), return bar_out
    take avg: 'bsz','asz','spd'
    take agg: 'vol','vbs','bqd','aqd','opt_v1','opt_v2'

    assuming bar_n has equal or less days than bar_in, i.e., N1
    all having same cols_dict, as returned by collect_raw()

    bar_in: shape[nday,nbar,mcol] to be merged onto, i.e. _N1
    cols_dict: {'col_name': col_ix}
    bar_n_list: list of bar_n, shape[nday,nbar,mcol], i.e. _N2/3

    return bar_out, same shape with bar_in, and cols avg/agg'ed
           note columes such as utc/ohlc/lpx, etc are not merged
    """
    avg_cols=['bsz','asz','spd']
    agg_cols=['vol','vbs','bqd','aqd','opt_v1','opt_v2']

    d,n,m=bar_in.shape
    assert m == len(cols_dict.keys())
    bc = {};
    for c in avg_cols:
        bc[c]=0
    bar_out = bar_in if in_pace else bar_in.copy()

    for k, bar_n in enumerate(bar_n_list):
        d0,n0,m0=bar_n.shape
        assert d0<=d and n==n0, 'bar shape mismatch!'
        assert m0 == len(cols_dict_n.keys()), 'bar_n(%d) cols mismatche with cols_dict_n'%(k)

        dix = np.searchsorted(\
                bar_in[:,-1,cols_dict['utc']], \
                bar_n[:,-1,cols_dict_n['utc']])
        assert np.max(np.abs(bar_in[dix,-1,cols_dict['utc']]-bar_n[:,-1,cols_dict_n['utc']]))==0,'%d day mismatch!'%(k)

        for c in agg_cols + avg_cols:
            if c in cols_dict.keys() and c in cols_dict_n.keys():
                c1 = cols_dict[c]
                cn = cols_dict_n[c]
                bar_out[dix,:,c1]+=bar_n[:,:,cn]
                if c in avg_cols:
                    bc[c]+=1

    for c in avg_cols:
        if bc[c]>0:
            bar_out[:,:,cols_dict[c]]/=bc[c]
    
    return bar_out

def plot_data_vbs_qbs_opt(bar, fig, cols_dict) :
    """
    plot relationship between close px with vbs,qbs,bp,opt
    bar: shape [nday,nbar,ncols]
    """
    d,n,m=bar.shape
    assert d*m<100*8280, 'too many data to visualize'
    bar0 = bar.reshape((d*n,m))

    utc,close,vbs,bsz,asz,spd,bqd,aqd,swp,ibg = [cols_dict[c] for c in ['utc','close','vbs','bsz','asz','spd','bqd','aqd','opt_v1','opt_v2']]

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

def remove_half_days(utc, lr=None, px_close=None, max_consecutive_0min=2*60, hours=(), mts_symbol=None):
    """
    enforce active open/close for the symbol,
        utc: shape [nday,nbar], used to find start/stop ix for the market open/close
        if lr or px_close is given, remove days with more than 'max_consecutive_0min',
           i.e. consecutive minutes with 0 price change during active market hours
        hours: if given, (start_hour, start_min, end_hour, end_min), same as mts_repo's get_bars()
        mts_symbol: if hours not given, use mts_symbol and the first day in utc to decide hours
    return 
       ix0, ix1: first and last index +1 into column of utc,
                 i.e. utc_crop=utc[:,ix0:ix1]
       half_days: list of index into row of utc
    """
    if len(hours) > 0:
        st='%02d:%02d:00'%(hours[0]%24,hours[1])
        et='%02d:%02d:00'%(hours[2]%24,hours[3])
    else:
        import symbol_map
        sm=symbol_map.SymbolMap()
        tinfo=sm.get_tinfo(mts_symbol.split('_')[0]+'_N1',ymd,is_mts_symbol=True,add_prev_day=True)
        st=tinfo['start_time']
        et=tinfo['end_time']

    barsec=utc[0,1]-utc[0,0]
    ymd=datetime.datetime.fromtimestamp(utc[0,-1]).strftime('%Y%m%d')
    utc0=int(datetime.datetime.strptime(ymd+st,'%Y%m%d%H:%M:%S').strftime('%s'))
    utc1=int(datetime.datetime.strptime(ymd+et,'%Y%m%d%H:%M:%S').strftime('%s'))
    if utc0>utc1:
        utc0-=(3600*24)
    ix0=np.searchsorted(utc[0,:],utc0)
    assert utc[0,ix0]==utc0+barsec
    ix1=np.searchsorted(utc[0,:],utc1)
    assert utc[0,ix1]==utc1
    ix1+=1

    # remove days with more than 1 hour of 0 lr
    # during active hours
    max0bars=int((max_consecutive_0min*60)//barsec)
    half_days=[]
    if lr is not None or px_close is not None:
        nd,nb=utc.shape
        if lr is None:
            pxf=px_close.flatten()
            lr = np.r_[0,np.log(pxf[1:]/pxf[:-1])]
        lr=lr.reshape((nd,nb))
        lr[np.isnan(lr)]=0;
        lr0=lr[:,ix0:ix1].flatten()
        x0=np.zeros(len(lr0))
        x0[np.nonzero(lr0==0)[0]]=1
        x0cs=np.cumsum(x0)
        ixzz=np.nonzero((x0cs[max0bars:]-x0cs[:-max0bars])==max0bars)[0]
        half_days=list(set(ixzz//(ix1-ix0)))
    return ix0,ix1,list(np.sort(half_days))

def weekly_data(utc, x, zero_day_fill):
    """
    get the weekly index to use for each day
    utc: shape nd,nb, only last bar used as the trading day
    x:   shape [nd, nb] of daily data
    """
    def weekday_ix(utc):
        ud = utc[:,-1].copy()
        wd = []
        wk = []
        wd0 = []
        wk0 = []
        k0=0
        d0=-1
        for t in utc[:,-1]:
            d=datetime.datetime.fromtimestamp(t).weekday()
            if d<=d0:
                while d0<4:
                    d0+=1
                    wk0.append(k0)
                    wd0.append(d0)
                k0+=1
                d0=-1
            while d0+1<d:
                d0+=1
                wk0.append(k0)
                wd0.append(d0)
            wd.append(d)
            wk.append(k0)
            d0=d
        assert wd[0] == 0,'first day not monday'
        utc_ix = np.array(wk)*5+np.array(wd)
        utc_ix0 = np.array(wk0)*5+np.array(wd0)
        return k0+1, utc_ix, utc_ix0

    nd,nb=x.shape
    k,ix,ix0=weekday_ix(utc)
    wv=np.empty((k*5,nb))
    wv[ix0,:]=zero_day_fill.copy()
    wv[-1,:]=zero_day_fill.copy()
    wv[ix,:]=x.copy()
    return wv.reshape((k,5,nb))

def smooth_lr(utc, px_close, ra_dict=None, px_open=None, std_mul=5, bad_day_sd=10):
    """
        lra, lrs, sdlr, px_close_ra, bad_days = smooth_lr(utc, px_close, ra_dict, px_open, std_mul, bad_day_sd)

    input:
        utc, px_close, shape [ndays, nbar], not roll adjusted, typically
             from a bar with from [ndays, nbars, [utc, close, vol, vbs, lpx]
        ra_dict: the md['ra'][n] roll_adj dict in md. If None, use px_open to calculate lr
        px_open: in case ra_dict is None, px_open is used to calculate lr
        std_mul: this multiplies the time-of-day sd curve to generate sdlr and lrs
        bad_day_sd: sd used in detecting bad days, when daily sd is more than 
                    daily_sd > avg_daily_sd + bad_day_sd*sd_of_daily_sd
        start_end_offset: diff from 18:00 and 17:00, in seconds
    return:
        lra: shape[ndays, nbar] the roll adjusted raw lr, bad_days NOT removed
        lrs: shape[ndays, nbar[ smoothed lra, bad_days NOT removed
        sdlr: length nbar std in-sample to Outlier(lra), with bad_days removed
              this can be used for online filtering
        px_close_ra: the roll adjusted quote close px, bad_days NOT removed
        bad_days: array of bad days index into ndays, dayix is just 
                  dayix = np.delete(np.arange(ndays),bad_days).astype(int)

    In general, procedure of data cleanning is:
        1. get a smoothed lra, bad days
        2. based on the lra, obtain a vol profile
        3. from vol, estimate a rough qr structure
        4. from the qr structure, attempt a vol_bucketing
    """
    def filter_days(lr, bad_day_sd=10):
        """
        remove days with extremely high vol, due to, say war
        Note1: lr is already roll adjusted
        """

        # filter days with extremely high vol, say war
        ndays, nbars = lr.shape
        day_ix = np.arange(ndays).astype(int)
        bad_days = []
        while len(day_ix)>2:
            lr0 = lr[day_ix, :]
            sd_day = np.std(lr0, axis=1)
            ix = np.nonzero(sd_day>(np.mean(sd_day)+np.std(sd_day)*bad_day_sd))[0]
            if len(ix) == 0:
                return np.array(bad_days).astype(int)
            bad_days = np.r_[bad_days, day_ix[ix]]
            day_ix = np.delete(day_ix, ix)
        raise "error in lr!"

    ndays, nbars = utc.shape
    utc0=utc.copy().flatten()
    px_close0=px_close.copy().flatten()
    bar0 = np.array([utc0, px_close0]).T.reshape((ndays, nbars,2))

    if ra_dict is not None:
        bar0 = mts_repo.MTS_REPO.roll_adj(bar0, 0, [1], ra_dict)
        px_close_ra = bar0[:,:,1]
        # in case the ra adjust the price to negative, adjust it up
        if np.min(px_close_ra) < 1e-10:
            print('got negative price after adjust!')
            px_close_ra += (np.abs(np.min(px_close_ra))+1e-10)
        px_open_ra  = np.r_[px_close_ra[0,0], px_close_ra.flatten()][:-1].reshape((ndays,nbars))
        lra = np.log(px_close_ra/px_open_ra)
        if px_open is not None:
            print('px_open not used!')
    else:
        print('no ra_dict, using px_open')
        px_close_ra=None
        lra=np.log(px_close/px_open)
    lra[np.isnan(lra)]=0

    # filter out bad days
    bad_days = filter_days(lra, bad_day_sd=bad_day_sd)
    dayix = np.delete(np.arange(ndays),bad_days).astype(int)

    # iteratively finds sd and smooth the outliers
    import vol_bucket as vb
    lrs = lra[dayix,:].copy()
    sdlr=np.std(lrs,axis=0)
    tol=3e-6; e=1e+10; iter_cnt=20
    while e>tol and iter_cnt>0:
        lrs=OT.soft1(lra[dayix,:],sdlr,std_mul,1)
        sd0=np.clip(vb.lrg1d(np.std(lrs,axis=0),width=1,poly=3),1e-8,1e+10)
        e=np.max(np.abs(sd0-sdlr))
        sdlr=sd0 ; iter_cnt-=1
        print('OT.soft err:%f/%f'%(e,tol))

    lrs = OT.soft1(lra,sdlr,std_mul,1)
    return lra, lrs, sdlr, px_close_ra, bad_days

def summary_plotting(fig, mts_symbol, md):
    """
    md: {'md':{'0','1','2'...:{shape[nday,nbar,ncol]} },
         'ra':{'0','1','2'...: ra},
         'cols':{'col_name':colix}
         }

    plots basic 10S bar structures using lr's std and bp correlation
    
    ###################
    ### setup data
    ###################
    # given md
    wti1=md['md'][1]
    cols=md['cols'] or cols:{collect_raw()::base_cols}
    ra=md['ra'][1]

    # in case vstack from multiple 1S files:
    fn=glob.glob('/tmp/md_pyres/collect_raw/WTI_20*.dill')
    wti1,cols,ra=vstack_md(1, md_fn_list=fn, col_list=['utc','close','vol','vbs','bsz','asz','spd','bqd','aqd'])

    utc=wti1[:,:,cols['utc']]
    px_close=wti1[:,:,cols['close']]
    ix0,ix1,half_days=remove_half_days(utc, px_close=px_cloase, max_consecutive_0min=2*60, hours=(-6,0,17,0))
    wti1=wti1[:,ix0:ix1,:]
    utc=wti1[:,:,cols['utc']]
    px_close=wti1[:,:,cols['close']]

    lra,lrs,sdlr,close_ra,bad_days=new_md.smooth_lr(utc,px_close,rastd_mul=5, bad_day_sd=10)
    bad_days=list(set(llist(bad_days)+list(half_days)))

    dayix=np.delete(np.arange(ndays),bad_days).astype(int)

    ###########################
    ### weekly analysis
    ###########################
    lrsw=weekly_data(utc,lrs,np.zeros(nb))
    # plot std for each of the 5 weekdays

    ############################
    ### time of day volatility
    ############################
    # plotting of the volatility curve

    # this checks the cycle of vol, where are most of volatilities go?
    # usually spike at 30 minutes, 5 minutes, 1 minutes, at open/settle
    # certain busy time could found 5-second spikes
    lb='base per-bar vol'
    fig=figure() ; ax1=fig.add_subplot(2,1,1);
    ax1.plot(dt, np.std(lrs[dayix,:],axis=0),label=lb)
    ax2=fig.add_subplot(2,1,1,sharex=ax1)
    lb='1S lr vs 1M'
    bs = 60 #60 seconds
    bs0 = 1 #first 1 seconds
    bs1 = 9 #next 9 seconds 
    assert bs0+bs1<=bs
    lb='%sS versus rest %sS in %sS bar'%(bs0,bs1,bs)
    nd,nb=lrs.shape
    lrsb=lrs.reshape((nd,nb//bs,bs))
    cb=[]
    for k in np.arange(nb//bs)).astype(int):
        cb0=np.corrcoef(np.sum(lrsb[:,k,:bs0],axis=1), np.sum(lrsb[:,k,bs0:bs0+bs1],axis=1))[0,1]
        cb.append(cb0)
    ax2.plot(dt[::bs], np.array(cb), label=lb)

    # create a 2D 1S in 5M dissemination chart
    # the corr of first 1S to the 2nd, third, etc seconds within 5M bar
    cb=[]
    bs = 300 #5M bar
    lrsb=lrs.reshape((nd,nb//bs,bs))
    for x in np.arange(bs-1).astype(int)+1:
        cb_=[]
        for k in np.arange(nb//bs).astype(int):
            cb0=np.corrcoef(np.sum(lrsb[dayix,k,:1],axis=1), np.sum(lrsb[dayix,k,x:x+1],axis=1))[0,1]
            cb_.append(cb0)
        cb.append(cb_)
    cb=np.array(cb)
    cbs=vb.lrg2d(cbs,(1,1))
    figure(); pcolor(cbs,cmap='RdBu',vmin=-0.05, vmax=0.05)
    xlabel('5m bars') ; ylabel('lr corr with first 1S bar');

    # some of the intraday patterns - impluse response
    # for example, WTI: the 10:30am (bar=198), use a 
    # surprise shock of second versus first plus previous 5m
    id1=np.sign(np.clip(lrs[dayix,300*198]*lrs[dayix,300*198+1],-1e+10,0))*lrs[dayix,300*198+1]*np.abs(np.sum(lrs[dayix,300*197:300*198+1],axis=1))
    id1=np.sign(np.clip(lrs[dayix,300*198]*lrs[dayix,300*198+1],-1e+10,0))*lrs[dayix,300*198+1]*np.abs(np.sum(lrs[dayix,300*191:300*198][:,::2],axis=1)+lrs[dayix,300*198])

    ix1=60; id0=id1 
    id0/=np.sqrt(np.sum(id0**2))
    x=(lrs[dayix,300*198+2:300*198+ix1].T*id0).T
    plot(np.cumsum(np.sum(x,axis=0)),label=lb)
    xs=np.sum(x,axis=1) ; print(xs.mean()/xs.std()) #shp 0.046

    # for 10am - first tick is the king
    ix1=16; id0=id1 ; lb='id1:1*2<0;2*|sum(-300:1)|' ; id0/=np.sqrt(np.sum(id0**2)); x=(lrs[dayix,300*192+1:300*192+ix1].T*id0).T ; plot(np.cumsum(np.sum(x, axis=0)),label=lb); xs=np.sum(x,axis=1) ; print(xs.mean()/xs.std()) 
    # shp is 0.1

    # first seconds on Sunday open (6pm)
    rt = (lrsw[:,0,0]+lrsw[:,0,2]+lrsw[:,0,3])*np.sum(lrsw[:,0,4:781],axis=1)
    print(rt.mean()/rt.std()*np.sqrt(52)) 
    # shp is 0.145 * sqrt(52) (since it's weekly)
    # assuming there are more such weekly signals to be extracted, also using ind

    ####################
    # try aggregate to see how trendy/mean-reversion at each time of day
    ####################
    lb0='std of lr from start to t, as agg of iid bars'
    figure() ; plot(dt, np.sqrt(np.cumsum(np.std(lrs[dayix,:],axis=0)**2)), label=lb0)

    lb1='std of lr from start to t, realized'
    plot(dt, np.std(np.cumsum(lrs[dayix,:],axis=1),axis=0),label=lb1)

    # check on a different start time
    stix=3600*4  # starting from 4 hours into open
    lrs0 = np.r_[lrs.flatten()[stix:],np.zeros(stix)].reshape(lrs.shape)
    dt0 = np.r_[dt[stix:], dt[:stix]+datetime.timedelta(1)]

    ix=np.argsort(np.std(lrs0,axis=1))
    dayix0=ix[:len(dayix)]
    plot(dt0, np.sqrt(np.cumsum(np.std(lrs0[dayix0,:],axis=0)**2)), label=lb0)
    plot(dt0, np.std(np.cumsum(lrs0[dayix0,:],axis=1),axis=0),label=lb1)

    ############################
    # getting all the indicators
    ############################
    bp1=wti1[dayix,:,cols['bsz']]/(wti1[dayix,:,cols['bsz']]+wti1[dayix,:,cols['asz']])-0.5
    spd1z=spd_zscore(wti1[dayix,:,cols['spd']], wti1[dayix,:,cols['lpx']], spd_hist_days=50)
    vbs1=wti1[dayix,:,cols['vbs']]
    qbs1=wti1[dayix,:,cols['bqd']]-wti1[dayix,:,cols['aqd']]

    # some derived from vbs/qbs
    vqbs1=vbs1+qbs1
    vqbs1t=np.sign(vbs1)*np.clip(vbs1*qbs1,0,1e+10)
    vqbs1r=np.sign(vbs1)*np.clip(vbs1*qbs1,-1e+10,0)
    ov1=wti1[dayix,:,cols['opt_v1']]  # swipe, volume beyond bbo, not accounted by qbs
    ov2=wti1[dayix,:,cols['opt_v2']]  # iceberg, volume at bbo, not accounted by qbs

    # for all n0123
    wtia=new_md.merge_n(md['md'][1][:,ix0:ix1,:],\
            [md['md'][0][:,ix0:ix1,:],\
             md['md'][2][:,ix0:ix1,:],\
             md['md'][3][:,ix0:ix1,:]],cols)
    bpa=wtia[dayix,:,cols['bsz']]/(wtia[dayix,:,cols['bsz']]+wtia[dayix,:,cols['asz']])-0.5
    vbsa=wtia[dayix,:,cols['vbs']]
    qbsa=wtia[dayix,:,cols['bqd']]-wtia[dayix,:,cols['aqd']]
    vqbsa=vbsa+qbsa
    vqbsat=np.sign(vbsa)*np.clip(vba*qbsa,0,1e+10)
    vqbsar=np.sign(vbsa)*np.clip(vbsa*qbsa,-1e+10,0)

    # target lr
    lrts   = get_lr_tgt(lrs[dayix,:], next_ix=None, lf=1, smooth_sd=2, return_nz_target=False)
    lrtsnz = get_lr_tgt(lrs[dayix,:], next_ix=None, lf=1, smooth_sd=2, return_nz_target=True)

    # qbs/vbs stuff

    """
    nd,nb=bp1.shape
    assert len(lrts)==nd*nb
    assert len(lrtsnz)==nd*nb

    cc=[]
    for i in np.arange(nb).astype(int):
        cc.append(np.corrcoef(lrts.reshape((nd,nb))[:,i],lrtsnz.reshape((nd,nb))[:,i])[0,1])
    cc=np.arrray(cc)

    ax1=fig.add_subplot(3,1,1)
    ax1.plot(dt, np.array(cc), label='prob of non-zero 10S bar')

    # plot sd iid vs cumsum
    ax2=fig.add_subplot(3,1,2,sharex=ax1)
    ax2.plot(dt, np.cumsum(np.std(np.r_[0,lrts[:-1]].reshape((nd,nb)),axis=0)**2), label='cumsum 10S bar variance')
    ax2.plot(dt, np.std(np.cumsum(np.r_[0,lrts[:-1]].reshape((nd,nb)),axis=1),axis=0)**2, label='variance of cumsum')

    c,cr,cs
    ax3=fig.add_subplot(3,1,3,sharex=ax1)
    ax3.plot(dt,c)


#####################################
# ind visualization 
#####################################
def zscore(v, z_hist_days=50):
    """
    get a zscore of v, based on previous z_hist_days
    input:
       v: shape[ndays, nbars] daily per-bar ind
       z_hist_days: prev days to get mean/std for normalization
    output:
       vz: the normalized v

    the first Z_hist_days are used to prime the pump
    """
    ndays,nbars=v.shape
    assert z_hist_days<=ndays/2,'too few days for hist'
    zd=z_hist_days  #lazy notation
    v0=np.vstack((np.zeros((1,nbars)),np.vstack((v[:zd,:],v))))
    vs=np.cumsum(v0,axis=0)
    vm=(vs[zd:,]-vs[:-zd,:])[:-1,:]/zd

    # create ix for zd to apply m
    ixv=(np.tile(np.arange(zd),(ndays,1)).T+np.arange(ndays)).T.astype(int).flatten()
    ixm=np.ravel(np.tile(np.arange(ndays),(zd,1)),order='F').astype(int)
    vm2=np.mean((v0[1+ixv,:]-vm[ixm,:]).reshape((ndays,zd,nbars))**2,axis=1)
    return (v-vm)/np.sqrt(vm2)

def ind_derived(v, bar_ix, lb):
    """
    v: shape [ndays,mbars] mean 0 indicator
    bar_ix: [ix0,ix1) the bars to be included
    lb: the lookback during the day

    output:
       next_ix: length nx vector of index onto nbars, as the first
                bar as response. nx=(ix1-ix0)//lb
       avg, diff, sss1, sss2: length(nd*nx) ind vectors,
                sss1, sss2: same sign shock, sss1: upon the lb, 
                sss2, upon v and agg on lb
    """
    ix0,ix1=bar_ix
    v=v[:,ix0:ix1].copy()
    n,m=v.shape
    m=m//lb*lb
    v=v[:,:m]

    #ix for immediate next of v[ix], for prediction
    next_ix=np.arange(0,m,lb)+lb+ix0 
    v=v.reshape((n,m//lb,lb))
    v_avg=np.mean(v,axis=2)
    #v_dif=v[:,:,-1]-v[:,:,0]
    # get the slope
    if lb>1:
        x0=np.arange(lb)-(lb-1)/2
        v_dif=np.mean(v*x0,axis=2)/(np.std(x0)**2)
    else:
        v_dif=v[:,:,-1]-v[:,:,0]

    # same-sign shock1
    vp=v_avg*v_dif
    ix=np.nonzero(vp>0)
    v_sss1=np.zeros((n,m//lb))
    v_sss1[ix]=np.sign(v_avg[ix])*vp[ix]

    # same-sign shock2
    v=v.reshape((n,m))
    vd=np.hstack((np.zeros((n,1)), (v[:,1:]-v[:,:-1])))
    vp=vd*v
    ix=np.nonzero(vp>0)
    v_sss2=np.zeros((n,m))
    v_sss2[ix]=np.sign(vd[ix])*vp[ix]
    v_sss2=np.sum(v_sss2.reshape((n,m//lb,lb)),axis=2)

    return next_ix, v_avg, v_dif, v_sss1, v_sss2

def get_lr_tgt(lr, next_ix=None, lf=1, smooth_sd=2, return_nz_target=False):
    """
    constrct lr target for each day of next_ix
    input:
       lr: shape[ndays,nbars]
       next_ix: length nx vector of index onto nbars, as the first
                bar as response. next_ix is same for each day.
                if none, then taken as np.arange(nbars)+1
       lf: look forward >=1
       smooth_sd: OT's ym
       return_nz_target: if true, return target lr as vector of [nd*nx] non-zero lr
                in this case, lf must be 1
    return:
        lr_tgt: shape[nd*nx, lf], each row is for the response of lf
    """

    nd,m=lr.shape
    if next_ix is None:
        next_ix=np.arange(m).astype(int)+1
    nx=len(next_ix)
    assert lf>=1
    assert smooth_sd>0
    if return_nz_target:
        assert lf==1

    lrs=lr.copy()
    for ym in np.arange(10,smooth_sd-1,-1).astype(int):
        sd = np.std(lrs,axis=0)
        lrs = OT.soft1(lrs, sd, ym, 1)

    # backward fill 0 in lrs
    if return_nz_target:
        lrs=np.r_[lrs.flatten(),1e-10]
        ixz=np.nonzero(lrs==0)[0]
        lrs[ixz]=np.nan
        df=pandas.DataFrame(lrs)
        df.fillna(method='bfill',inplace=True)

    ix = (np.tile(next_ix,(nd,1)).T+np.arange(nd)*m).T.flatten().astype(int)
    lrt=[]
    lrf=np.r_[lrs.flatten(),np.zeros(lf)]
    for x in np.arange(lf).astype(int):
        lrt.append(lrf[ix+x])
    return np.array(lrt).T

def ind_tgt(v, bar_ix, lb, lr, lf):
    next_ix, va, vd, s1, s2=ind_derived(v,bar_ix,lb)
    lrt=get_lr_tgt(lr,next_ix,lf)
    return lrt, va, vd, s1, s2

def in_out(ind0, lr_tgt0, ax=None, ax_x=None, ax_label='', percentile=(0,1)):
    """
       lr_resp = in_out(ind, lr_tgt)
    input:
        ind: vector of nxnd, with mean 0
        lr_tgt:  shape [nxnd,lf],
        percentile: the picking of ind, applying to all nxnd inds
    return:
        lr_resp: np.mean((lr_tgt.T*ind).T,axis=0)
    """

    nxnd,lf=lr_tgt0.shape

    # apply percentile
    ix0,ix1=(nxnd*np.array(percentile)).astype(int)
    ind_ix=np.argsort(ind0)[ix0:ix1]
    print('pick ind: ', ind0[ind_ix[0]],ind0[ind_ix[-1]], len(ind_ix))
    ind=ind0[ind_ix]
    lr_tgt=lr_tgt0[ind_ix,:]
    nxnd,lf=lr_tgt.shape

    assert nxnd==len(ind)
    scl=np.sqrt(np.sum(ind**2))
    resp=(lr_tgt.T*ind).T
    resp_m = np.sum(resp,axis=0)/scl
    resp_s=np.sum(resp,axis=1)
    resp_shp=np.mean(resp_s)/np.std(resp_s)
    resp_s_m = np.sum(resp_s)/scl

    if ax is not None:
        if ax_x is None:
            ax_x=np.arange(lf)
        lb='%s:sum(%.5f)shp(%.3f)'%(ax_label,resp_s_m,resp_shp)
        ax.plot(ax_x, resp_m, label=lb)
        ax.legend(loc='best')
    else:
        return resp, resp_m, resp_shp

######
# book pressure
# 1. raw on 10sec bar:
#    * 
#    * bp: immediate lb1-lf1: best on hours[0,8] and [20,23]
#    * slope: may have longer term lb150:lf:300, percent(0.99,1), only hours 18-20 - 
# 2. state with spread
#    if spread increase, trend weaken
#    if spread decreases, trend strengthen
def bp_timing(lrt, bp1, bpa, bp1z, bpaz):
    """
    check the 5-minute effect, special time of day effect
    together with n1 and merged n0123
    return
       for each bp, at each bar
       c0: correlation
       cr: the mean of bp*lr (not scaled, z already normalized)
       cs: the shp 
    results:
        bp1 good in most of time, better during
            asian/europen/us close, us hours needs faster signal
        bpa is only good at certain instance, like
             3:00, 9:15, 9:20, 10:30, 14:30, etc
    to visualize: plot(dt, c[0])
    """
    c=[]
    cr=[]
    cs=[]
    nd,nb=bp1.shape
    if len(lrt.shape)==1:
        lrt=lrt.reshape((nd,nb))
    else:
        assert len(lrt.shape)==2 and lrt.shape[0]==nd and lrt.shape[1]==nb
    for bp in [bp1, bpa, bp1z, bpaz]:
        c0=[]
        for ix0 in np.arange(nb).astype(int):
            c0.append(np.corrcoef(bp[:,ix0],lrt[:,ix0])[0,1])
        c.append(c0)
        m0=np.mean(bp*lrt[:,:],axis=0)
        cr.append(m0)
        cs.append(m0/np.std(bp*lrt[:,:],axis=0))
    return np.array(c), np.array(cr), np.array(cs)

def bp_picking(bp, lrtsnz):

    # pick non-agreeing sign
    ixnz=np.nonzero(bp.flatten()[1:-1]*bp.flatten()[:-2]<-0.0)[0]

    # pick second one big
    ixnz=ixnz[np.nonzero(np.abs(bp).flatten()[ixnz+1]>0.35)[0]]

    lrnz=lrs[dayix,:].flatten()[ixnz+2]
    print(np.corrcoef(bp.flatten()[ixnz+1], lrnz)[0,1])
    print(len(ixnz)/len(bp.flatten()))
    print((np.sign(bp.flatten()[ixnz+1])*lrnz).mean()/lrnz.std())
    print((np.sign(bp.flatten()[ixnz+1])*lrnz).mean())


def spd_zscore(spd, lpx, spd_hist_days=50):
    spdz=zscore(spd/lpx,z_hist_days=spd_hist_days)
    spdz[np.isnan(spdz)]=0
    spdz = OT.soft1(spdz, np.ones(spdz.shape), 1, 1)
    return spdz

def bp_spd(bpr, spdz):
    """
    if spd increases, trend on,
    if spd decreases, trend stops

    """
    pass

######################
# vbs and qbs
######################


#########################################
### not used, check the summary above to plot
##########################################
### vol trendy and mean-reversion
def vol_bucket_px(px_close_ra, bad_days = [], agg_bar=1):
    ndays, nbars = px_close_ra.shape
    day_ix = np.delete(np.arange(ndays).astype(int), np.array(bad_days).astype(int))
    px_close_ra = px_close_ra.copy()[day_ix,:]
    ndays = len(day_ix)

    #   using px_close_ra for lr
        # aggregate
    aggix = np.arange(0,nbars,agg_bar).astype(int)
    px_open0  = np.r_[px_close_ra[0,0], px_close_ra.flatten()][:-1].reshape((ndays,nbars))[:,aggix]
    px_close0_d = np.r_[px_open0.flatten()[1:], px_close_ra[-1,-1]].reshape((ndays,nbars//agg_bar))
    px_close0 = np.hstack((px_close0_d[:-1,:],px_close0_d[1:,:]))
    px_open0 = px_open0[:-1,:] # remove the last day
    vol = []
    nbars = nbars//agg_bar
    for ix in np.arange(nbars).astype(int):
        px_close_ix = px_close0[:,ix:ix+nbars]
        px_open_ix = px_open0[:,ix]
        vol.append(np.std(np.log(px_close_ix.T/px_open_ix).T,axis=0))
        print(ix, nbars)

    return np.array(vol).reshape((nbars,nbars))

def vol_bucket(lra, bad_days = [], agg_bar=1):
    """
    calculate for each daily bar time an aggregated volatility

    lra: shape [ndays, nbars] log return, roll adjusted, possible smoothed
    agg_bar: number of aggregated bars to start from. i.e. 
             if barsec = 10S, agg_bar=3 will calculate
             based on barsec = 30
    """
    ndays, nbars = lra.shape
    day_ix = np.delete(np.arange(ndays).astype(int), np.array(bad_days).astype(int))
    lrd = lra[day_ix,:].copy()  # bad days removed
    lrd = np.hstack((lrd[:-1,:],lrd[:-1,:])) # stack up across a day
    lrcs = np.hstack((np.zeros((lrd.shape[0],1)),np.cumsum(lrd, axis=1))) #add first colume as 0
    ixagg = np.arange(0,nbars,agg_bar).astype(int)+agg_bar
    vol = []
    nbars = nbars//agg_bar
    for ix in np.arange(nbars).astype(int)*agg_bar:
        lrs = lrcs[:,ixagg+ix].T-lrcs[:,ix]
        vol.append(np.std(lrs,axis=1))
        if ix%300 == 0:
            print(ix, nbars*agg_bar)
    return np.array(vol).reshape((nbars,nbars))

def plot_vol_1d(vol, ax, ix, utc=None):
    """
    visualize the vol to compare the iid sum since a given ix,
    with the relized vol since ix
    vol: shape [nbars, nbars]
    utc, if given, length nbars utc for xlabel
    """
    nbars,_=vol.shape
    dt = np.arange(nbars*2).astype(int)
    t0 = 'ix:'+str(ix)
    if utc is not None:
        dt=[]
        for t in utc:
            dt.append(datetime.datetime.fromtimestamp(t))
        for t in utc+3600*24:
            dt.append(datetime.datetime.fromtimestamp(t))
        dt=np.array(dt)
        t0 += '(%s)'%(dt[0].strftime('%H:%M:%S'))

    vol0 = vol[:,0]**2
    vol0 = np.r_[vol0,vol0]
    ax.plot(dt[ix:ix+nbars], np.sqrt(np.cumsum(vol0[ix:ix+nbars])), label='std since '+ t0 +' iid sum')
    ax.plot(dt[ix:ix+nbars], vol[ix,:], label='std since '+t0 + ' realized')
    ax.legend(loc='best')

def plot_vol_2d(vol, fig, utc=None):
    """visualize the vol in 2d on the difference between the vol at bar i, 
    with the difference of vol(k,i)-vol(k,i-1), plotted as (k,i). when
    k>i, i is the next day's index.
    negative shows bar i contributes negatively to vol(k,i), given vol(k,i-1),
    which indicates a negative correlation of return of bar i with return of (k,i-1)
    """
    nbars, _=vol.shape
    x = np.arange(nbars).astype(int)
    y = np.arange(2*nbars).astype(int)[::-1]
    """
    if utc is not None:
        x=[]
        for t in utc:
            x.append(datetime.datetime.fromtimestamp(t))
        x = np.array(x)
        y = x.copy()
    """
    vold = np.hstack((vol[:,:1]**2, vol[:,1:]**2-vol[:,:-1]**2))
    voldv=np.zeros((2*nbars,nbars))


################################
# use a penalized lr to find
# short term forecasts, to be
# used as an indicator for
# possibly weekly pyres
#
# To mine indicators using 1s
# data, three things need to be fixed
# 1. general time of day
#    - vbs/qbs/lrs/bp/vol/spd/ov1/ov2
#    - agreement and surprise as states
#    - 
#    - run at each whole minute
#    - target at next 1s, 4s, 29s, 59s, 299s
#
# 2. time of day(week)
#    - general 1S perdictability would largely
#      due to 
# 
# the signal -
# 
#     we need to 
#     
#################################

class StateVQR:
    """state space based on vbs/qbs/lrs
    There are 4 states fixed from vqr. For each state,
    generate indicators from a given lookback, also
    a target from a given look forward. 
    Setup to allow clustering or surface fit into
    input space expansion.

    Indicators: 
    1, lrss, lrs, vbs, qbs, bp, ov1, ov2, spd, vol
    """
    def __init__(self, file_dict=None):
        if file_dict is None:
            fn0='/tmp/md_pyres/collect_raw/wti1s_vqbs_20100101_20230224.dill'
            fn1='/tmp/md_pyres/collect_raw/wti1s_bpspdvol_20100101_20230224.dill'
            fn2='/tmp/md_pyres/collect_raw/wti1s_volov_20100101_20230224.dill'
            fnix='/tmp/md_pyres/collect_raw/ix_lb1_lb5.dill'
            file_dict={'lrs':fn0,'vbs':fn0,'qbs':fn0,'bp':fn1,'spd':fn1,'vol':fn1,'opt_v1':fn2,'opt_v2':fn2,'fnix':fnix}
        self.file_dict=file_dict  # {'name':fn}
        self.func_dict={'lrs':self._get_lb_agg,\
                        'vbs':self._get_lb_agg,\
                        'qbs':self._get_lb_agg,\
                        'bp' :self._get_lb_avg,\
                        'ov1':self._get_lb_agg,\
                        'ov2':self._get_lb_agg,\
                        'spd':self._get_lb_avg,\
                        'vol':self._get_lb_agg,\
                        'bsz':self._get_lb_avg,\
                        'asz':self._get_lb_avg}

    def _get_data(self, name):
        fn=self.file_dict[name]
        d=dill.load(open(fn,'rb'))
        return d[name]

    def _get_state_from_fnix(self, key='ix_lb1', states=[0,1,2,3,4]):
        ix=[]
        lb_flat=dill.load(open(self.file_dict['fnix'],'rb'))[key].flatten()
        for s in states:
            ix.append(np.nonzero(lb_flat==s)[0])
        return ix

    def _get_lb_agg(self, x, lookback):
        """ x_lb=_get_lb_agg(x, lookback)
        x: shape [nday,nbar] data
        return: x_lb for aggregated upto each bar on each day
        """
        nd,nb=x.shape
        xc=np.cumsum(x.flatten())
        xc[lookback:]-=xc[:-lookback]
        return xc.reshape((nd,nb))

    def _get_lb_avg(self, x, lookback):
        nd,nb=x.shape
        xc=np.cumsum(x.flatten())
        xc[lookback:]-=xc[:-lookback]
        if lookback>1:
            xc=xc.astype(float)
            xc[lookback:]/=lookback
            xc[:lookback]/=(np.arange(lookback)+1)
        return xc.reshape((nd,nb))

    def _get_lf(self, x, state_ix, lf, func=None):
        """
           x: flattened nd*nb data,
           state_ix: index at which a prediction is made, lf starts at +1
           lf: scalar of number of bars to include from state_ix+1
           func: if not None, takes a vector (x) returns a value, 
                 i.e. np.sum, or np.mean
        return: 
           v: shape(len(state_ix), lf) if func is none, else len-lf
        """
        v=[]
        xf=x.flatten()
        for lf0 in np.arange(lf)+1:
            v0 = xf[np.clip(state_ix+lf0,0,len(xf)-1)]
            v.append(v0)
        v=np.array(v).T
        if func is not None:
            v=func(v,axis=1)
        return v

    def _state_vqr(self, lrs, vbs, qbs, lookback):
        """
        first two sign agree, with the 3rd disagree
        ixvrnq=np.nonzero((np.clip(np.sign(vbs*-qbs),0,1)*np.sign(vbs*lrs)).flatten()>0)[0]
        Note: 
            Can vary the sequence of the lrs/vbs/qbs for different state.
            Can also negate qbs for all agree state
        """
        lrsb=self._get_lb_agg(vbs,lookback).flatten()
        ind0=lrsb*(self._get_lb_agg(lrs,lookback).flatten())
        ix0=np.nonzero(ind0>0)[0]
        ind0=ind0[ix0]
        lrsb=lrsb[ix0]
        ix1=np.nonzero((self._get_lb_agg(qbs,lookback).flatten()[ix0])*lrsb < 0)[0]
        return ix0[ix1]


    #########################
    ## state transition 
    #########################
    def state_code(self, lookback, data_dict=None):
        """
        ix_label, data_dict = state_code(self, lookback, data_dict=None)
        input: 
            data_dict: if not None, expect the huge 'lrs','vbs','qbs',
                   all shape [nday,nbar] 1S bar
        return:
           ix_label: shape(nday,nbar) state label, one of [0,1,2,3,4]
           data_dict: the input data_dict, populated if not given, otherwise, same
        """

        if data_dict is None:
            data_dict={}
            for name in ['lrs','vbs','qbs']:
                data_dict[name]=self._get_data(name)
        (lrs, vbs, qbs) = (data_dict['lrs'], data_dict['vbs'], data_dict['qbs'])

        ix_s1=self._state_vqr(lrs, vbs, qbs, lookback)  #qbs disagree
        ix_s2=self._state_vqr(lrs, qbs, vbs, lookback)  #vbs disagree
        ix_s3=self._state_vqr(vbs, qbs, lrs, lookback)  #lrs disagree
        ix_s4=self._state_vqr(lrs, vbs, -qbs,lookback)  #all agree
        nd,nb=lrs.shape
        ix=np.zeros((nd,nb))
        for ixs, s in zip([ix_s1,ix_s2,ix_s3,ix_s4], [1,2,3,4]):
            ix[(ixs//nb).astype(int),(ixs%nb).astype(int)]=s
        #assert len(np.nonzero(ix==0)[0])==0
        return ix, data_dict

    def state_summary(self, ix, lrs, vbs, qbs):
        # prints all 5 state's summary statistics
        # return:
        #   xxlist: list of ix_state, into the lrs.flatten()
        #           i.e. xxlist[0]: ix of state 0, etc
        nd,nb=ix.shape
        ixf=ix.flatten()
        xxlist=[]
        for s in [0,1,2,3,4]:
            xxs=np.nonzero(ixf==s)[0]
            print('state %d(%02.2f%%):\tlr_std(0-1):%.5f-%.5f\tlr_corr(0-1):%.3f'%(s, \
                    len(xxs)/(nd*nb)*100.0, np.std(lrs.flatten()[xxs]), \
                    np.std(lrs.flatten()[xxs[:-1]+1]), \
                    np.corrcoef(lrs.flatten()[xxs[:-1]], lrs.flatten()[xxs[:-1]+1])[0,1]))
            xxlist.append(xxs)
        return xxlist

    def state_transit(self, ix, xxlist):
        # get transition state
        ixs0,ixs1,ixs2,ixs3,ixs4=xxlist
        ixf0=ix.flatten().copy()
        ixf0[ixs0]=np.nan
        pd=pandas.DataFrame(ixf0)
        pd.fillna(method='bfill',inplace=True)

        xs1_prob=[]
        print('state\t\tstate 1\tstate 2\tstate 3\tstate 4')
        for s, xs in zip([0,1,2,3,4], xxlist):
            xs1=ixf0[xs[:-1]+1]
            scnt=[]

            print('%d(%.3f)'%(s,len(xs)/len(ixf0)),end="")
            for s1 in [1,2,3,4]:
                scnt.append(len(np.nonzero(xs1==s1)[0])/len(xs1))
                print('\t%.2f'%(scnt[-1]),end="")
            print('')
            xs1_prob.append(scnt)
        return np.array(xs1_prob)

    def _ind(self, name, x, state_ix, lookback_list):
        """state_ix is for flatten() index, return from _state_vqr.
           at bar state_ix[i]'s close, prediction starts
           lookback_lsit: list of look back, 1 mean the just the start_ix[i],
           return:
               length nhist list of x_at_state, each with len(state_ix)
        """

        if x is None:
            #x=_get_data(name)
            raise 'for preventing memory leak, save x by _get_data(%s)'%(name)
        x_list=[]
        for lookback in lb_list:
            x_list.append( self.func_dict[name](x,lookback).flatten()[ix] )
        return x_list

    def _tgt(self, lrs, ix, lookforward):
        nday,nbar=lrs.shape
        lrsf=lrs.flatten()
        tgt=[]
        ixlf=np.tile(ix,(lookforward,1)).T
        ixlf=np.clip(ixlf+(np.arange(lookforward)+1),0,nday*nbar-1).flatten()
        return lrs.flatten()[ixlf].reshape((len(ix),lookforward))

"""
xx=ix221[ix2_ix2_ix2_ix1] ; 
lf=300 ; 
tgt=vqr._get_lf(lrs,xx-lf,2*lf) ; 
tgt0=vqr._get_lf(lrs, xx, lf//2) ; 
rt=np.cumsum(tgt0.T*np.sign(np.sum(tgt[:,lf-1:lf],axis=1)),axis=0) ; 
plot(np.mean(rt, axis=1)); 
rts=rt[-1,:] ; 
print(rts.mean()/rts.std(), \
      np.sum(rts)/len(rts), \
      np.corrcoef(np.sign(np.sum(tgt[:,lf-1:lf],axis=1)), np.sum(tgt0,axis=1))[0,1], \
      np.corrcoef(np.sum(tgt[:,lf-1:lf],axis=1),np.sum(tgt0,axis=1))[0,1])

tgt=vqr._get_lf(lrs,ix21-300, 600) ; 
tgts2=   np.sign(np.sum(tgt[:,:240],axis=1))+ \
         np.sign(np.sum(tgt[:,240:290],axis=1))+\
         np.sign(np.sum(tgt[:,290:299],axis=1)) + \
         np.sign(tgt[:,299]) ; 
ix11_=np.nonzero(np.abs(tgts2)==4)[0] ; ix11_.shape
   ix11_ shape=23709

LF =3600//7, SHP: 0.08, mean per-trade gain: 1.96 ticks(???)

10am trade about 3300
- removing 10am gets better recent performance: LF=3600//7, SHP: 0.08, mean per-trade gain: 1.87 ticks
- backward search for lr sign consistency with the reversal state

using ix11_ on top of ix221 instead of ix21 doesn't

But ix11_ doesn't go well in the first 5 months of 2023, I am less convinced on this
signal, unless put in a larger regression to show on state of ix1 or ix21

"""

class StateFeatures :
    def __init__(self, vqr, lrs, vbs, qbs):
        self.vqr=vqr;
        self.lrs=lrs
        self.vbs=vbs
        self.qbs=qbs

    def gen(self, state=1, state_lookback=(1,5), tgt_lookforward=(60,300,600)):
        pass


