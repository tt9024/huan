import mts_repo
import numpy as np
import datetime
import copy
import dill
import os
import strat_utils
import symbol_map
from scipy.linalg import hankel

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

ALL_Symbols = Comm_Symbols + Eqt_Symbols + Rate_Symbols + FX_Symbols

def collect(symbol_arr, start_day, end_day, contract='N1', md_dict_in=None, persist_fn = None, barsec=300, cols = ['utc','lpx','bsz','asz','spd','bqd','aqd', 'vol', 'vbs'],repo = None, hours=(-6,0,17,0), get_holiday=False):
    if repo is None:
        repo = mts_repo.MTS_REPO_TickData()
    md_dict={}
    for symbol in symbol_arr:
        md_dict[symbol] = repo.get_bars(symbol+'_'+contract, start_day, end_day, barsec=barsec, cols = cols,\
                is_mts_symbol=True, get_roll_adj=False, hours=hours, get_holiday=get_holiday, allow_bfill=True)

    if md_dict_in is not None:
        for symbol in md_dict.keys():
            if symbol in md_dict_in.keys():
                md0 = md_dict_in[symbol]
                if md0[-1,-1,0] > md_dict[symbol][0,0,0] :
                    # replace existing with new
                    dt0 = md0[:,0,0]
                    ix = np.nonzero(dt0<md_dict[symbol][0,0,0])[0]
                    print('remove ', len(dt0)-len(ix), ' days from md_dict_in')
                    md0 = md0[ix,:,:]
                md_dict_in[symbol] = np.vstack((md0, md_dict[symbol]))
            else :
                md_dict_in[symbol] = md_dict[symbol]
        md_dict = md_dict_in

    if persist_fn is not None:
        print('saving to ', persist_fn)
        with open(persist_fn, 'wb') as fp:
            dill.dump(md_dict, fp)

    return md_dict

def md_dict_crop_trading_hours(md_dict, barsec):
    md = {}
    mdata = mts_repo.MTS_DATA(-6,0,17,0)
    smap = symbol_map.SymbolMap(max_N=1)
    for k in md_dict.keys():
        day = datetime.datetime.fromtimestamp(md_dict[k][-1,-1,0]).strftime('%Y%m%d')
        stuc,eutc=mdata._get_utc(day, mts_sym=k+'_N1', smap=smap)
        k0 = int((stuc+barsec-md_dict[k][-1,0,0])//barsec)
        k1 = int((eutc - md_dict[k][-1,0,0])//barsec+1)
        md[k] = md_dict[k][:,k0:k1,:]
    return md

def plot_hourly(symbol_arr, utc, data_dict, axlist, highlight_points = 5):
    """
    data_dict: {'symbol': {'label': label, 'data': data} }
    """
    dt = []
    for t in utc:
        dt.append(datetime.datetime.fromtimestamp(t))
    dt=np.array(dt)

    for sym, ax in zip(symbol_arr, axlist):
        dd = data_dict[sym]
        ax.plot(dt, dd['data'], label=dd['label'])
        if highlight_points is not None:
            ax.axvspan(dt[-highlight_points], dt[-1], color='y', alpha=0.4, lw=0)
        if 'avg' in dd.keys() and 'std' in dd.keys():
            y0 = dd['avg']-dd['std']
            y1 = dd['avg']+dd['std']
            ax.fill_between(dt, y0, y1, alpha=0.5, color='r', lw=0, label=dd['label_std'])

def get_intraday(bars, merge_cnt=12) :
    """
    get the hourly bbo and spd, utc and price
    """

    ndays, n, _ = bars.shape
    utc = bars[:,:,0].reshape((ndays, n//merge_cnt, merge_cnt))[:,:,0] # get the first utc of each hour
    lpx_m = bars[:,:,1].reshape((ndays, n//merge_cnt, merge_cnt)).mean(axis=2) # get the mean px of each hour
    lpx = bars[:,:,1].reshape((ndays,n//merge_cnt,merge_cnt))[:,:,-1]

    # simple return from lpx, the first return is set to zero, to be addressed by roll adj, or adding open/close
    lpx0 = np.hstack((lpx[:,:1],lpx))
    ret=lpx0[:,1:]-lpx0[:,:-1]

    bbo = ((bars[:,:,2]+bars[:,:,3])/2).reshape((ndays, n//merge_cnt, merge_cnt)).mean(axis=2) # get the mean bbo of each hour
    spd = bars[:,:,4].reshape((ndays, n//merge_cnt, merge_cnt)).mean(axis=2)

    spd = spd/lpx_m*1e+4  # to the basis point
    #bbo_notional = bbo*lpx # to the notional
    vol = bars[:,:,7].reshape((ndays, n//merge_cnt, merge_cnt)).sum(axis=2)

    return np.array([utc, lpx, bbo, spd, vol, ret])

def plot_intraday(symbol_arr, md_dict, axlist_bbo=None, axlist_spd=None, axlist_vol=None) :
    """
    plot avg bbo and spd for 5 day vs 21 day
    """
    # generate data
    bbo_dict5 = {}
    bbo_dict21 = {}
    bbo_dict50 = {}
    spd_dict5 = {}
    spd_dict21 = {}
    spd_dict50 = {}
    vol_dict5 = {}
    vol_dict21 = {}
    vol_dict50 = {}
    utc = None
    for sym in symbol_arr :
        d = get_intraday(md_dict[sym], merge_cnt = 1)
        utc = d[0,-1,:]
        for avg_days, dd_bbo, dd_spd, dd_vol in zip([5, 21, 50], [bbo_dict5, bbo_dict21, bbo_dict50], [spd_dict5, spd_dict21, spd_dict50], [vol_dict5, vol_dict21, vol_dict50]):
            dd_bbo[sym] = {'label': sym+' bbo:'+str(avg_days)+'d', 'data':np.mean(d[2,-avg_days:,:], axis=0)}
            dd_spd[sym] = {'label': sym+' spd:'+str(avg_days)+'d', 'data':np.mean(d[3,-avg_days:,:], axis=0)}
            dd_vol[sym] = {'label': sym+' volume:'+str(avg_days)+'d', 'data':np.mean(d[4,-avg_days:,:], axis=0)}

    if axlist_bbo is not None:
        # plot hourly bbo
        for dd in [bbo_dict5, bbo_dict21, bbo_dict50]:
            plot_hourly(symbol_arr, utc, dd, axlist_bbo, highlight_points=None)

    if axlist_spd is not None:
        # plot hourly bbo
        for dd in [spd_dict5, spd_dict21, spd_dict50]:
            plot_hourly(symbol_arr, utc, dd, axlist_spd, highlight_points=None)

    if axlist_vol is not None:
        for dd in [vol_dict5, vol_dict21, vol_dict50]:
            plot_hourly(symbol_arr, utc, dd, axlist_vol, highlight_points=None)

def gen_intraday_figures(md_dict, symbol_arr, title_str='', pdf_file=None):
    from matplotlib import pylab as pl
    import matplotlib.dates as mdates
    #for symbol_arr in [Energy_Symbols, Metal_Symbols, Agri_Symbols, LiveStock_Symbols, Softs_Symbols] :
    fig = pl.figure()
    n = len(symbol_arr)
    axlist_bbo = []
    axlist_spd = []
    axlist_vol = []
    for i in np.arange(n):
        ax_spd=fig.add_subplot(3,n,i+1)
        ax_bbo=fig.add_subplot(3,n,n+i+1, sharex=ax_spd)
        ax_vol=fig.add_subplot(3,n,2*n+i+1, sharex=ax_spd)
        for ax, axlist in zip([ax_bbo, ax_spd, ax_vol], [axlist_bbo, axlist_spd, axlist_vol]):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            axlist.append(ax)

    plot_intraday(symbol_arr, md_dict, axlist_bbo=axlist_bbo, axlist_spd=axlist_spd, axlist_vol=axlist_vol)
    fig.autofmt_xdate()
    for ax in axlist_bbo+axlist_spd+axlist_vol:
        ax.grid() ; ax.legend(loc='best')

    #fig.set_title(title_str + ' Hourly BBO and Spread')

    figure = pl.gcf()
    figure.set_size_inches(24,4)
    import matplotlib
    matplotlib.rc('font', size=10)

    if pdf_file is not None:
        pl.savefig(pdf_file+'_intraday.pdf',dpi=400)

### daily
def get_daily(bars) :
    """
    get the hourly bbo and spd, utc and price
    """
    ndays, n, _ = bars.shape
    utc = bars[:,-1,0]
    lpx = bars[:,:-1,1].mean(axis=1)
    bbo = np.median((bars[:,:-1,2]+bars[:,:-1,3])/2,axis=1) # get the mean bbo of each hour
    spd = np.median(bars[:,:-1,4], axis=1)
    spd = spd/lpx*1e+4

    # adding bqd-aqd
    bdif = bars[:,:-1,5].mean(axis=1)
    adif = bars[:,:-1,6].mean(axis=1)
    bbo_dif = bdif-adif

    # adding vol
    vol = bars[:,:-1,7].sum(axis=1)
    vbs = bars[:,:-1,8].sum(axis=1)

    return np.array([utc, lpx, bbo, spd, vol])

def plot_daily(symbol_arr, md_dict, axlist_bbo=None, axlist_spd=None, axlist_vol=None, lookback=75):
    """
    plot daily avg bbo and spd 
    """
    # generate data
    bbo_dict = {}
    spd_dict = {}
    vol_dict = {}
    utc = None
    for sym in symbol_arr :
        d = get_daily(md_dict[sym])
        utc = d[0,-lookback:]

        dd = d[2,:].copy()
        dda = strat_utils.rolling_window(dd, 5, np.mean)
        dds = strat_utils.rolling_window(dd, 40, np.std)
        bbo_dict[sym] = {'label': sym+' daily bbo', \
                'data':dd[-lookback:], 'avg':dda[-lookback:], 'std':dds[-lookback:], 'label_std':'1-std'}

        dd = d[3,:].copy()
        dda = strat_utils.rolling_window(dd, 5, np.mean)
        dds = strat_utils.rolling_window(dd, 40, np.std)
        spd_dict[sym] = {'label': sym+' daily spd', \
                'data':dd[-lookback:], 'avg':dda[-lookback:], 'std':dds[-lookback:], 'label_std':'1-std'}

        dd = d[4,:].copy()
        dda = strat_utils.rolling_window(dd, 5, np.mean)
        dds = strat_utils.rolling_window(dd, 40, np.std)
        vol_dict[sym] = {'label': sym+' daily vol', \
                'data':dd[-lookback:], 'avg':dda[-lookback:], 'std':dds[-lookback:], 'label_std':'1-std'}


    if axlist_bbo is not None:
        plot_hourly(symbol_arr, utc, bbo_dict, axlist_bbo)

    if axlist_spd is not None:
        plot_hourly(symbol_arr, utc, spd_dict, axlist_spd)

    if axlist_vol is not None:
        plot_hourly(symbol_arr, utc, vol_dict, axlist_vol)

def gen_daily_figures(md_dict, symbol_arr, title_str = '', pdf_file=None):
    from matplotlib import pylab as pl
    #for symbol_arr in [Energy_Symbols, Metal_Symbols, Agri_Symbols, LiveStock_Symbols, Softs_Symbols] :
    fig = pl.figure()
    n = len(symbol_arr)
    axlist_bbo = []
    axlist_spd = []
    axlist_vol = []
    for i in np.arange(n):
        axlist_spd.append(fig.add_subplot(3,n,i+1))
        axlist_bbo.append(fig.add_subplot(3,n,n+i+1,sharex=axlist_spd[-1]))
        axlist_vol.append(fig.add_subplot(3,n,2*n+i+1, sharex=axlist_spd[-1]))

    plot_daily(symbol_arr, md_dict, axlist_bbo=axlist_bbo, axlist_spd=axlist_spd, axlist_vol=axlist_vol)

    fig.autofmt_xdate()
    for ax in axlist_bbo+axlist_spd+axlist_vol:
        ax.grid() ; ax.legend(loc='best')

    #axlist_bbo[0].set_title(title_str + ' Daily BBO and Spread')
    figure = pl.gcf()
    figure.set_size_inches(24,4)
    import matplotlib
    matplotlib.rc('font', size=10)

    if pdf_file is not None:
        pl.savefig(pdf_file+'_daily.pdf',dpi=400)

def gen_summary_csv(md_dict, symbol_arr, csv_file=None) :
    # generate csv from the daily data
    csv_dict = {}
    for sym in symbol_arr :
        d = get_daily(md_dict[sym])
        bbo = d[2,:]
        spd = d[3,:]
        vol = d[4,:]
        dv = []

        mean_week=vol[-5:].mean()
        dv.append(mean_week)
        for x in [vol, bbo, spd]:
            dif_wow=(x[-5:].mean() - x[-10:-5].mean())/x[-10:-5].mean()
            dif_mom=(x[-20:].mean() - x[-40:-20].mean())/x[-40:-20].mean()
            dv.append(dif_wow) ; dv.append(dif_mom)
        csv_dict[sym] = np.array(dv)

    if csv_file is not None:
        with open(csv_file,'wt') as fp:
            fp.write('            ,volume, volume_chg(wow%), volume_chg(mom%), spread_chg(wow%), spread_chg(mom%), bbo_chg(wow%), bbo_chg(mom%)\n')
            for s in csv_dict.keys():
                dv = csv_dict[s]
                fp.write('%s'%(s))
                fp.write(',%d'%(int(dv[0]))) # the weekly mean volume
                for x in dv[1:]:
                    fp.write(',%d'%(int(np.round(x*100))))
                fp.write('\n')

def gen_daily_bbo_spd_csv(md_dict, symbol_arr, path=None):
    # save a csv for daily series, bbo and spd
    # save figures for all symbol types

    daily_bbo = []
    daily_spd = []
    daily_vol = []
    utc = []
    for sym in symbol_arr:
        d = get_daily(md_dict[sym])
        if len(utc) == 0:
            utc = d[0,-50:]
        daily_bbo.append(d[2,-50:])
        daily_spd.append(d[3,-50:])
        daily_vol.append(d[4,-50:])

    daily_bbo = np.array(daily_bbo).T
    daily_spd = np.array(daily_spd).T
    daily_vol = np.array(daily_vol).T

    if path is None:
        return utc, daily_bbo, daily_spd, daily_vol

    # write to the daily csv file
    for name, x in zip (['daily_vol', 'daily_bbo','daily_spd'], [daily_vol, daily_bbo, daily_spd]):
        csv_fn = os.path.join(path, name + '.csv')
        with open(csv_fn, 'wt') as fp:
            fp.write('    ')
            for s in symbol_arr:
                fp.write(',%s'%(s))
            fp.write('\n')
            for utc0, x0 in zip(utc, x):
                dt = datetime.datetime.fromtimestamp(utc0).strftime('%Y-%m-%d')
                fp.write('%s'%dt)
                for x00 in x0:
                    fp.write(',%.2f'%(x00))
                fp.write('\n')

def gen_all(md_dict, upd_date, path='/mnt/mts/weekly_liquidity'):
    # generate the summary csv
    path0 = os.path.join(path, upd_date)
    path = os.path.join(path0, 'summary')
    os.system('mkdir -p ' + path + ' > /dev/null 2>&1')
    for sa, fn in zip([Comm_Symbols, Eqt_Symbols, Rate_Symbols, FX_Symbols], ['Commodity', 'Equity', 'Rates', 'FX']) :
        gen_summary_csv(md_dict, sa, csv_file = os.path.join(path, fn+'.csv'))

    path = os.path.join(path0, 'detail')
    os.system('mkdir -p ' + path + ' > /dev/null 2>&1')
    # generate the daily bbo and spd csv
    gen_daily_bbo_spd_csv(md_dict, ALL_Symbols, path)

    # generate all the daily figures
    for name, sym in zip(['Energy', 'Metal', 'Agriculture', 'LiveStock', 'Softs', 'EquityUS', 'EquityEU', 'RatesUS', 'RatesEU', 'FXFuture'], \
            [Energy_Symbols, Metal_Symbols, Agri_Symbols, LiveStock_Symbols, Softs_Symbols, EquityUS_Symbols, EquityEU_Symbols, RatesUS_Symbols, RatesEU_Symbols, FXFuture_Symbols]):
        gen_daily_figures(md_dict, sym, pdf_file=os.path.join(path, name))
        gen_intraday_figures(md_dict, sym, pdf_file=os.path.join(path, name))


def run_all(end_day, persist_path = '/mnt/mts/weekly_liquidity'):
    # this loads the md_dict from persist_path/bars/md_all.dill
    # and update the market data upto (include) the end_day
    # and generates all the report
    fn = os.path.join(persist_path, 'bars', 'md_all.dill')
    md_dict = dill.load(open(fn, 'rb'))
    day = datetime.datetime.fromtimestamp(md_dict['WTI'][-1,-1,0]).strftime('%Y%m%d')
    import mts_util
    tdi = mts_util.TradingDayIterator(day)
    tdi.begin()
    sday = tdi.next()
    md_dict = collect(ALL_Symbols, sday, end_day, md_dict_in=md_dict, persist_fn=fn)
    gen_all(md_dict, end_day, path=persist_path)

### for the risk monitor output ###
def output_5m_volume_spread_csv(md_dict_dill_list, out_csv, volume_col=7, spread_col=4, utc_col=0, start_day_ix = 0, end_day_ix = -1, lookbacks=[1,3,12]):
    """
    generates 5m volume spread file for risk monitor

    md_dict_dill_list: a list of md_dict_dill files, dumped from the md_dict returned by
        md_dict = collect(symbol_arr, start_day, end_day, contract='N1', md_dict_in=None, persist_fn = None)
    md_dict: {'WTI': shape [ndays, nbars, ncols] }, volume_col is 7 and spread_col is 4

    Typically run N0,N1,N2
    Aggregates all volumes of same symbol and average of spreads
    append to out_csv with each market with per bar: [avg_volume_lookback1, std_volume_lookback1, avg_volume_lookback2, std_volume_lookback2,...] avg_spd, std_spread
    """

    d = {} 
    import Outliers as OT
    ym = 10
    scl = 1
    with open (out_csv, 'at+') as fp:
        for md_dict_dill in md_dict_dill_list:
            md = dill.load(open(md_dict_dill,'rb'))
            for k in md.keys():
                nd, nb, nc = md[k].shape
                assert nb == 276
                v = md[k][start_day_ix:end_day_ix,:,volume_col]
                s = md[k][start_day_ix:end_day_ix,:,spread_col]
                utc = md[k][start_day_ix:end_day_ix,-1,utc_col]
                v = OT.soft1(v, np.std(v,axis=0), ym, 1)
                s = OT.soft1(s, np.std(s,axis=0), ym, 1)
                """
                if k == 'WTI':
                    import pdb
                    pdb.set_trace()
                """
                if len(v[np.isnan(v)]) > 0 or \
                   len(v[np.isinf(v)]) > 0 or \
                   len(v[np.isnan(s)]) > 0 or \
                   len(v[np.isinf(s)]) > 0:

                    # sometimes it could be the market not open 
                    # on all 276 bars from 18:00 to 17:00. 
                    # Not a concern but check it out
                    print(md_dict_dill, k, ' replaced nan/inf with 0')
                    v[np.isnan(v)]=0
                    v[np.isinf(v)]=0
                    s[np.isnan(s)]=0
                    s[np.isnan(s)]=0

                assert len(np.nonzero(v<0)[0]) == 0
                assert len(np.nonzero(s<0)[0]) == 0

                if k in d.keys():
                    # aggregate volumes of same mkt (of N0,N1,etc) at each bar
                    # average spread of same mkt (of N0, N1,etc) at each bar
                    utc0 = d[k]['utc']
                    if len(utc0) > len(utc):
                        ix = np.clip(np.searchsorted(utc0, utc),0,len(utc0)-1)
                        d[k]['v'][ix,:] += v
                        d[k]['s'][ix,:] = (d[k]['s'][ix,:]+s)/2.0
                    else:
                        ix = np.clip(np.searchsorted(utc, utc0),0,len(utc)-1)
                        v[ix,:] += d[k]['v']
                        s[ix,:] = (s[ix,:]+d[k]['s'])/2
                        d[k]['v'] = v
                        d[k]['s'] = s
                        d[k]['utc'] = utc
                else :
                    d[k]={}
                    d[k]['v'] = v
                    d[k]['s'] = s
                    d[k]['utc'] = utc

        vd = {}
        sd = {}
        for k in d.keys():
            v = d[k]['v']
            s = d[k]['s']
            v = OT.soft1(v, np.std(v,axis=0), ym, 1)
            s = OT.soft1(s, np.std(s,axis=0), ym, 1)

            # take a lookback
            vms = []
            for lookback in lookbacks:
                if lookback > 1:
                    # aggregate v with trailing lookback
                    nd, nb = v.shape
                    v0 = v.copy().flatten()
                    v0 = (np.sum(hankel(v0[:-lookback+1], v0[-lookback:]),axis=1)[-(nd-1)*nb:]).reshape((nd-1,nb))
                else :
                    v0 = v.copy()
                vms.append(np.mean(v0,axis=0))
                vms.append(np.std(v0,axis=0))

            vms.append(np.mean(s,axis=0))
            vms.append(np.std(s,axis=0))
            bar_data = np.ravel(np.array(vms).T)
            # remove nan/inf from bar_data
            bar_data[np.isnan(bar_data)]=0
            bar_data[np.isnan(bar_data)]=0
            line = k + ', ' + str(bar_data.tolist())[1:-1] + '\n'
            fp.write(line)
            fp.flush()
        return vms

def run_5m_volume_spread_csv(start_day, end_day, out_csv):
    tmp_file = '/tmp/ml_dump_start_date_end_date_'
    file_list = []


    #cols = ['utc','lpx','bsz','asz','spd','bqd','aqd', 'vol', 'vbs']
    cols = ['utc','spd', 'vol']
    #syms = ALL_Symbols + VIX_Symbols
    syms = ['Corn']

    for n in ['N0','N1','N2']:
        try :
            md_dict = collect(syms, start_day, end_day, contract=n, persist_fn = tmp_file+n, cols = cols)
        except KeyboardInterrupt:
            return 
        file_list.append(tmp_file+n)

    return output_5m_volume_spread_csv(file_list, out_csv, volume_col=2, spread_col=1)

