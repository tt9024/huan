import numpy as np
import mts_repo
import mts_util

k0 = ['WTI', 'Brent', 'NG', 'HO', 'RBOB']
k1 = ['Gold','Silver','HGCopper','Palladium']
k2 = ['SPX','NDX','DAX','EuroStoxx']
k3 = ['FV', 'TY', 'US','BUXL','Bund','Schatz']
k4 = ['JPY','EUR','CAD','GBP','AUD','NZD','MXN','ZAR','RUB'] 
k5 = ['Corn','Wheat','Soybeans','SoybeanMeal','SoybeanOil','LeanHogs','LiveCattle']

repo_path = '/home/mts/run/repo/tickdata_prod'
BARSEC = 300

def _get_lr0(tradable, yyyy, repo) :
    yyyy = int(yyyy)
    tdi = mts_util.TradingDayIterator(str(yyyy)+'0101',str(yyyy)+'0201')
    d2 = tdi.begin()
    cnt = 0
    while True :
        bars = repo.get_bars(tradable, d2, d2, barsec=BARSEC,ignore_prev=True)
        if len(bars) == 1 :
            break
        cnt += 1
        if cnt > 2 :
            raise RuntimeError('not found! ' + tradable + ' ' + d2)
        d2 = tdi.next()

    lr = np.log(bars[0,:,4]) - np.log(bars[0,:,1])
    return lr

def get_lr(sym, yyyy_start, yyyy_end, repo) :
    yyyy = int(yyyy_start)
    tradable = sym+'_N1'
    lr = []
    for yyyy in np.arange(int(yyyy_start), int(yyyy_end)+1).astype(int) :
        try :
            lr.append(_get_lr0(tradable, yyyy, repo))
        except KeyboardInterrupt as e :
            raise 'stopped'
        except :
            print ('problem getting ' + str(yyyy))

    lr = np.array(lr)
    lr_m = np.mean(lr, axis=0)
    lr_s = np.std(lr, axis=0)
    return lr, lr_m, lr_s

def get_lr_all(sym_list, yyyy_start, yyyy_end) :
    """
    sym_list = k0 + k1 + k2 + k3 + k4 + k5
    yyyy_start, yyyy_end = 1990, 2021
    """
    lr_dict = {}
    repo = mts_repo.MTS_REPO(repo_path)
    for sym in sym_list :
        lr, lrm, lrs = get_lr(sym, yyyy_start, yyyy_end, repo)
        lr_dict[sym] = [lr, lrm, lrs]
    return lr_dict

def _get_dt(barsec) :
    import datetime
    utc0 = int(datetime.datetime.strptime('20210101-18:00:00','%Y%m%d-%H:%M:%S').strftime('%s'))
    utc1 = int(datetime.datetime.strptime('20210102-17:00:00','%Y%m%d-%H:%M:%S').strftime('%s'))
    utc = np.arange(utc0+barsec,utc1+1,barsec)
    dt = []
    for t in utc:
        dt.append(datetime.datetime.fromtimestamp(t))
    dt=np.array(dt)
    return dt

def plot_lr_all(lr_dict, fig, ax_shape, k_list=None, k_name = None, dt=None, pin_bar=186, title_str = '') :
    """
    k_list = [k0, k1, k2, k3, k4, k5]
    k_name = ['Energy', 'Metal', 'Equity', 'Fix Income', 'FX', 'Agriculture']
    fig = figure()
    ax_shape = [2,3], so that
        ax_list= ax1=fig.add_subplot(2,3,1); ax2=fig.add_subplot(2,3,2,sharex=ax1); etc
    """
    if k_list is None :
        k_list = [k0, k1, k2, k3, k4, k5]
        k_name = ['Energy', 'Metal', 'Equity', 'Fix Income', 'FX', 'Agriculture'] 
    if dt is None :
        dt = _get_dt(BARSEC)
    a1,a2=ax_shape
    N = len(k_list)
    assert a1*a2 == N
    assert N == len(k_name)
    ax_list = [fig.add_subplot(a1,a2,1)]
    for i in np.arange(N-1)+2:
        ax_list.append(fig.add_subplot(a1,a2,i,sharex=ax_list[0]))

    from matplotlib.dates import DateFormatter
    from matplotlib import dates
    import Outliers as OT

    date_form = DateFormatter("%H:%M")
    for k, n, ax in zip(k_list, k_name, ax_list) :
        nn = len(k)
        ax.set_ylabel(n + ' return')
        for i in np.arange(nn).astype(int) :
            lr0 = lr_dict[k[i]][0]
            if len(lr0) == 0 :
                continue
            nix = np.nonzero(np.isnan(lr0))
            if len(nix[0]) > 0 :
                lr0[nix] = 0
            lr_sd = np.std(lr0, axis=0)
            lr0 = OT.soft1(lr0, lr_sd, 3, 1)
            nix = np.nonzero(np.isnan(lr0))
            if len(nix[0]) > 0 :
                lr0[nix] = 0
            lrm = np.mean(lr0,axis=0)
            lrm = OT.soft1(lrm, np.std(lrm),5,1)
            lrc = np.cumsum(lrm)
            lrc -= lrc[pin_bar]
            ax.plot(dt, lrc, label=k[i])
        ax.legend(loc='best') 
        ax.grid()
        ax.axvspan(dt[pin_bar], dt[-1], color='y', alpha=0.5, lw=0)
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(dates.HourLocator(interval=1))

    ax_list[0].set_title(title_str)
    fig.autofmt_xdate()
    return ax_list
