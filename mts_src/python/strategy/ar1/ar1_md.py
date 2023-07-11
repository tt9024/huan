import numpy as np
import datetime
import time
import copy
import os
import traceback
import pandas
import subprocess

import mts_util
import symbol_map
import tpmon
import mts_repo

mts_repo_mapping_all = { \
        'CL':'WTI_N1', 'LCO':'Brent_N1', 'NG':'NG_N1', 'HO':'HO_N1', 'RB':'RBOB_N1', \
        'HG':'HGCopper_N1', 'GC':'Gold_N1', 'PA':'Palladium_N1', 'SI':'Silver_N1', \
        'ES':'SPX_N1',\
        'ZF':'FV_N1', 'ZN':'TY_N1','ZB':'US_N1', \
        '6J':'JPY_N1','6E':'EUR_N1', '6C':'CAD_N1','6B':'GBP_N1','6A':'AUD_N1','6N':'NZD_N1',\
        '6M':'MXN_N1', '6R':'RUB_N1','6Z':'ZAR_N1', \
        'ZC':'Corn_N1', 'ZW':'Wheat_N1','ZS':'Soybeans_N1','ZM':'SoybeanMeal_N1', 'ZL':'SoybeanOil_N1',\
        'HE':'LeanHogs_N1', 'LE':'LiveCattle_N1', \
        'FGBX':'BUXL_N1', 'FGBL':'Bund_N1', 'FGBS':'Schatz_N1', \
        'FDX':'DAX_N1', 'STXE':'EuroStoxx_N1' \
        }

mts_repo_mapping_all_n2 = { \
        'CL':'WTI_N2', 'LCO':'Brent_N2', 'NG':'NG_N2', 'HO':'HO_N2', 'RB':'RBOB_N2', \
        'HG':'HGCopper_N2', 'GC':'Gold_N2', 'PA':'Palladium_N2', 'SI':'Silver_N2', \
        'ES':'SPX_N2',\
        'ZF':'FV_N2', 'ZN':'TY_N2','ZB':'US_N2', \
        '6J':'JPY_N2','6E':'EUR_N2', '6C':'CAD_N2','6B':'GBP_N2','6A':'AUD_N2','6N':'NZD_N2',\
        '6M':'MXN_N2', '6R':'RUB_N2','6Z':'ZAR_N2', \
        'ZC':'Corn_N2', 'ZW':'Wheat_N2','ZS':'Soybeans_N2','ZM':'SoybeanMeal_N2', 'ZL':'SoybeanOil_N2',\
        'HE':'LeanHogs_N2', 'LE':'LiveCattle_N2', \
        'FGBX':'BUXL_N2', 'FGBL':'Bund_N2', 'FGBS':'Schatz_N2', \
        'FDX':'DAX_N2', 'STXE':'EuroStoxx_N2' \
        }

mts_repo_mapping2 = { \
        'CL':'WTI_N1', 'NG':'NG_N1', 'HG':'HGCopper_N1', 'GC':'Gold_N1', 'SI':'Silver_N1', 'ES':'SPX_N1', \
        'ZF':'FV_N1', 'ZN':'TY_N1','ZB':'US_N1',  '6J':'JPY_N1','6E':'EUR_N1', '6C':'CAD_N1','6B':'GBP_N1','6A':'AUD_N1','6N':'NZD_N1'
        }

mts_repo_mapping = mts_repo_mapping2
mts_repo_mapping_n2 = mts_repo_mapping_all_n2

md_dict_col_mapping = {'lr':0, 'vol':1, 'vbs':2, 'lpx':3, 'utc':4}

def col_from_md_dict(md_dict, symbol, col_name):
    return md_dict[symbol][md_dict_col_mapping[col_name]]

def md_dict_to_array(md_dict) :
    d = []
    dl=0
    for k in md_dict.keys():
        dl = len(md_dict[k].flatten())
        break
    for sym in mts_repo_mapping.keys():
        try :
            data = md_dict[sym].flatten()
        except :
            data = np.zeros(dl)
        d= np.r_[d, data]
    return np.array(d)

def md_dict_from_array(d):
    #assert len(d.shape)==1
    kl = len(mts_repo_mapping.keys())
    dl = len(d)//kl
    md_dict = {}
    for i, sym in enumerate(mts_repo_mapping.keys()) :
        md_dict[sym] = d[i*dl:(i+1)*dl]
    return md_dict

def md_dict_from_array_multiday(d):
    n,m = d.shape
    kl = len(mts_repo_mapping.keys())
    dl = m//kl
    md_dict = {}
    for i, sym in enumerate(mts_repo_mapping.keys()) :
        md_dict[sym] = d[:,i*dl:(i+1)*dl]
    return md_dict

def get_bar_dict(csv_line):
    bar_dict = {}
    try :
        line = csv_line.split(',')
        # bar line in the format of
        # bar_utc, open, high, low, close, vol, last_px, last_micro, vbs
        bar_dict['close_px'] = float(line[4])
        lpx = float(line[6])
        if np.isnan(lpx) or np.abs(lpx)<1e-10:
            lpx=bar_dict['close_px']
        bar_dict['last_trade_px'] = lpx
        bar_dict['last_upd'] = float(line[7])/1000000.0
        bar_dict['vol'] = int(line[5])
        bar_dict['vbs'] = int(line[8])
        bar_dict['utc'] = int(line[0])
    except :
        print ('error getting bar line')
    return bar_dict

def get_next_bar(fd) :
    # return the next bar using the file description fd
    # return None if no updates
    bar_dict = None
    line = fd.readline()
    if len(line) > 0 :
        bar_dict = get_bar_dict(line[:-1].decode())
    return bar_dict

def get_bar_lastn(fname, last_n):
    bar_dict_arr = []
    if last_n > 0 :
        lines = subprocess.check_output(['tail', '-n', str(last_n), fname])[:-1].decode().split('\n')
        for line in lines :
            bar_dict = get_bar_dict(line)
            bar_dict_arr.append(bar_dict)
    return bar_dict_arr

def get_bar_lastn_fd(fname, last_n):
    # this returns the latest n bars, together with 
    # the corresponding fd, that can be used to 
    # continuously monitor the bar file onwards
    bar = get_bar_lastn(fname, last_n)
    fd = open(fname, 'rb')
    fd.seek(0,2)
    return bar, fd

def get_bar_file_name(mts_symbol_arr, trade_day, bar_sec) :
    sm = symbol_map.SymbolMap()
    mts_sym_map = sm.get_tradable_map(trade_day, mts_key=True)
    import mts_util
    tdi = mts_util.TradingDayIterator(trade_day)
    tdi.begin()
    trade_day_prev = tdi.prev()
    mts_sym_map_prev = sm.get_tradable_map(trade_day_prev, mts_key=True)
    for k in mts_sym_map_prev.keys():
        if k not in mts_sym_map.keys():
            mts_sym_map[k] = mts_sym_map_prev[k]

    bf_dict = {}
    for mts_symbol in mts_symbol_arr :
        tm = mts_sym_map[mts_symbol]
        bf_dict[mts_symbol] = tpmon.get_bar_file(tm['tradable'], tm['venue'], bar_sec=bar_sec)
    return bf_dict

def md_setup(trade_day, barsec):
    mts_symbol_arr = []
    for symbol in mts_repo_mapping.keys():
        mts_symbol_arr.append(mts_repo_mapping[symbol])
    bf_dict0 = get_bar_file_name(mts_symbol_arr, trade_day, 1)
    bf_dict1 = get_bar_file_name(mts_symbol_arr, trade_day, barsec)
    bf_dict_1S = {}
    bf_dict_300S = {}
    for symbol in mts_repo_mapping.keys():
        bf1s = bf_dict0[mts_repo_mapping[symbol]]
        bf_dict_1S[symbol] = bf1s
        bf300s = bf_dict1[mts_repo_mapping[symbol]]
        bf_dict_300S[symbol] = bf300s
    return bf_dict_1S, bf_dict_300S

def md_from_bar_arr(bar_dict_arr):
    """
    This gets a cumulative updates from the bar_dict_arr
    bar_dict_arr should be strictly in order, i.e. missings are filled with bars
    with previous utc. 
    the first bar's lr/vol/vbs are not included
    """
    bar_dict0 = bar_dict_arr[0]
    md_dict = np.array([0,0,0, bar_dict0['last_trade_px'], bar_dict0['utc']])
    for bar_dict1 in bar_dict_arr[1:] :
        vol = bar_dict1['vol']
        vbs = bar_dict1['vbs']
        lr = 0
        lpx = bar_dict1['last_trade_px']
        utc = bar_dict1['utc']

        px0 = bar_dict0['close_px']
        if utc > bar_dict0['utc'] :
            if px0 > 1e-10 :
                lr = np.log(bar_dict1['close_px']/px0)
            md_dict[:3] += np.array([lr,vol,vbs])
            md_dict[3:] = np.array([lpx, utc])
            bar_dict0 = bar_dict1
        elif utc == bar_dict0['utc']:
            continue
        else :
            print('bar_dict utc loopback?')
            raise RuntimeError('bar_dict utc loopback')
    return md_dict

def fill_bar_array(bar_array, t0, t1, barsec) :
    """
    forward and backfill the missing bars, return bar_array strictly 
    increasing barsec from t0, to t1, inclusive

    note: utc from bar_array should be a multiple of barsec
          it is not checked here
    """
    utc = []
    for bar in bar_array:
        utc.append(bar['utc'])
    utc = np.array(utc).astype(int)
    if len(utc) == 0:
        # nothing to process
        utc0 = np.arange(t0, t1+barsec, barsec).astype(int)
        return utc0, [], np.array([])

    # sort the utc and crop t0/t1
    ix = np.argsort(utc, kind='stable')
    utc = utc[ix]
    sutc = min(utc[0], t0)
    eutc = max(utc[-1], t1)
    utc0 = np.arange(sutc, eutc+barsec, barsec).astype(int)

    uix = np.array([np.nan]*len(utc0))
    uix[(utc-sutc)//barsec] = np.arange(len(ix)).astype(int)

    df = pandas.DataFrame(uix)
    df.fillna(method='ffill',inplace=True)
    df.fillna(method='backfill',inplace=True)
    uix = uix.astype(int)

    # populate the bar_array
    bar_array_out = []
    ix0 = np.searchsorted(utc0, t0)
    ix1 = np.searchsorted(utc0, t1)
    ix2 = ix[uix[ix0:ix1+1]]
    for i0 in ix2:
        bar_array_out.append(copy.deepcopy(bar_array[i0]))
    return utc0[ix0:ix1+1], bar_array_out, ix2

def normalize_bar_array(bar_array, barsec) :
    """
    forward fill the missing bars, return bar_array strictly 
    increasing barsec from t0, to t1, where 
    t0: earliest utc
    t1: latest utc
    """
    utc = []
    for bar in bar_array:
        utc.append(bar['utc'])
    utc = np.array(utc).astype(int)
    if len(utc) < 2 :
        return bar_array, utc
    t0 = min(utc)
    t1 = max(utc)
    utc0, bar_array_out, uix = fill_bar_array(bar_array, t0, t1, barsec)
    return bar_array_out, utc0

def snap_update(bfd_dict, snap_dict):
    """
    bfd_dict: format of  {symbol: {'fname', 'fd'} }
    snap_dict: state of snap data, in format of {'last_bar', 'last_md'}
               last_bar: the latest snap bar dict, format of return value of get_bar_dict
               last_md:  aggregated update since start of time of latest bar time upto 'last_bar'
                         it is reset upon start of a new bar (see on_bar() and snap_new_bar())
                         format of bar_data, an array of [ lr, vol, vbs, lpx, ... ]
                         updated at last_bar
    return:  updated md_dict
    """
    snap_sec = 1
    md_dict = {}
    for sym in bfd_dict:
        bar_array = [snap_dict[sym]['last_bar']]
        bfd = bfd_dict[sym]['fd']
        while True:
            md_ = get_next_bar(bfd)
            if md_ is None :
                break
            if md_['utc'] > bar_array[-1]['utc']:
                bar_array.append(md_)
            else :
                if len(bar_array) > 1 :
                    print ('ignored non-inc. in ', bfd_dict[sym]['fname'], str(md_))

        if len(bar_array) > 1 :
            # get a bar array from snap_dict's utc to the latest
            t0 = bar_array[0]['utc']
            t1 = bar_array[-1]['utc']
            utc0, bar_array_out, uix = fill_bar_array(bar_array, t0, t1, snap_sec)

            # update md_dict with the new bars
            snap_md0 = md_from_bar_arr(bar_array_out)
            snap_md0[:3] += snap_dict[sym]['last_md'][:3]
            snap_dict[sym]['last_md'] = snap_md0

            # save the last bar
            snap_dict[sym]['last_bar'] = bar_array_out[-1]
        md_dict[sym] = snap_dict[sym]['last_md']
    return md_dict

def snap_new_bar(new_bar, snap_dict) :
    snap_dict['last_bar'] = new_bar
    snap_dict['last_md'] = np.array([0,0,0, new_bar['last_trade_px'], new_bar['utc']])

class AR1Data :
    """
    This module is supposed to be loaded and run in real-time from sod to eod. 
    The creation time is supposed to be just before sod, say 5:55pm, and
    the eod time is supposed to be run around 5pm just after the last bar.

    In case it is created during a trading time, the driver of this object,
    i.e. the model gets the missing bars since sod, fed into the forecast and pop.

    State:
    md_dict_snap  - the on-going snap md_dict (to be fed to model)  within the bar
    bar_dict_snap - the latest snap bar dict so far
    bar_dict      - the latest bar dict (at last bar close time)
    """
    def __init__(self, barsec, shour=-6, smin=0, ehour=17, emin=0): 
        self.tdu = mts_util.TradingDayUtil(shour=shour, smin=smin, ehour=ehour, emin=emin)
        self.barsec = int(barsec)
    
    def sod(self) :
        """
        typically called before 6pm, but could be called during trading day
        """
        self.trade_day = self.tdu.get_trading_day(snap_forward=True)
        self.bf_snap, self.bf = md_setup(self.trade_day, self.barsec)

        # calculate the 'k' based on time of day
        dtnow = datetime.datetime.now()
        utcnow = int(dtnow.strftime('%s'))
        trade_day = self.tdu.get_trading_day_dt(dtnow, snap_forward=True)
        t0 = self.tdu.get_start_utc(trade_day)
        t1 = self.tdu.get_end_utc(trade_day)
        cur_bar_utc = max(utcnow//self.barsec*self.barsec,t0)
        k = self.tdu.get_current_bar(dtnow, self.barsec)

        # k is -1 if not in trading
        if k <= 0 :
            bcnt = 1
        else :
            bcnt = k*2+1 # in case there were duplication

        snap_cnt = utcnow%self.barsec
        if k < 0 : 
            snap_cnt = 0

        snap_dict = {}
        bar_dict = {}
        for sym in self.bf.keys():
            # the bar state
            fname = self.bf[sym]
            fname_snap = self.bf_snap[sym]
            bar_arr, fd = get_bar_lastn_fd(fname, bcnt)
            snap_bar_arr, fd_snap = get_bar_lastn_fd(fname_snap, snap_cnt)

            if len(bar_arr) ==0  :
                fd.close() ; snap_fd.close()
                raise RuntimeError('no bar for %s'%(sym))

            self.bf[sym] = {'fname':fname, 'fd':fd}
            self.bf_snap[sym] = {'fname':fname_snap, 'fd':fd_snap}

            # t0's bar will be forward filled by previous bars
            # first bar is over-night, with 'utc' being the prevous day's t1
            tarr, barr, ix = fill_bar_array(bar_arr, t0, cur_bar_utc, self.barsec)

            # initialize bar state
            bar_dict[sym] = {'bt': copy.deepcopy(tarr.astype(int)), 'bar': copy.deepcopy(barr)}

            # initialize snap state
            snap_bar_arr = [barr[-1]] + snap_bar_arr
            ts0 =  barr[-1]['utc']
            ts1 = max(ts0, snap_bar_arr[-1]['utc'])
            tarr, barr, ix = fill_bar_array(snap_bar_arr, ts0, ts1, 1)
            snap_md = md_from_bar_arr(barr)
            snap_dict[sym] = {'last_md': copy.deepcopy(snap_md), 'last_bar': copy.deepcopy(barr[-1])}

        self.snap_dict = snap_dict
        self.bar_dict = bar_dict
        self.trade_day = trade_day
        self.t0 = t0
        self.t1 = t1

    def get_snap(self):
        """
        note this doesn't check if the snap would cross the bar boundary.
        the model thread should make sure the bar is updated to the latest before
        calling get snap
        """
        return snap_update(self.bf_snap, self.snap_dict)

    def on_bar(self, bar_utc):
        """
        return a md_dict from (bar_utc - barsec) to bar_utc
        the bar update at bar_utc may have already been received, as in the case of
        starting in the middle of a day.
        If the bar update for bar_utc is not available for any of the 
        symbols, it returns None. The model thread can try again after a while

        state detail:
        'bt': a numpy array of strictly increaing utc from t0 to t1, len=n+1
        'bar': list of same length, for corresponding bar dict

        Note this assumes the bar update in sequence, i.e. the bar time would not
        decrease. It checks and ignores such bars.
        """
        md_dict = {}
        bar_utc = int(bar_utc)
        for sym in self.bar_dict.keys() :
            bd=self.bar_dict[sym]
            last_utc = bd['bt'][-1]
            if bar_utc > last_utc:
                # try to get a new bar
                fd = self.bf[sym]['fd']
                while True :
                    bar = get_next_bar(fd)
                    if bar is None :
                        break

                    utc0 = int(bar['utc'])
                    if utc0 == last_utc + self.barsec:
                        # in sequence update
                        bd['bt'] = (np.r_[bd['bt'], utc0]).astype(int)
                        bd['bar'].append(bar)
                    elif utc0 > last_utc + self.barsec:
                        # skip
                        print ('skipped bar for', sym, ' at ', str(bar_utc), ' : ', str(bar), '\n', str(bd))
                        bd['bt'], bd['bar'], _ = fill_bar_array(bd['bar'].appand(bar), t0, utc0, self.barsec)
                    elif utc0 == last_utc:
                        print ('Warning! updated duplicated bar for', sym, ' at ', str(bar_utc), ' : ', str(bar), '\n', str(bd))
                        bd['bar'][-1] = bar
                    else :
                        print ('Error! ignored decreasing bar for', sym, ' at ', str(bar_utc), ' : ', str(bar), '\n', str(bd))
                        continue
                    last_utc = bd['bt'][-1]

                # did we got it?
                if bar_utc <= bd['bt'][-1] :
                    # update the snap on the new bar
                    snap_new_bar(bd['bar'][-1], self.snap_dict[sym])
                else :
                    #print('no update for ', self.bf[sym]['fname'], datetime.datetime.now(), datetime.datetime.fromtimestamp(bar_utc))
                    return None

            # generate md_update for this bar_utc
            bix = np.searchsorted(bd['bt'], bar_utc)
            md_dict[sym] = md_from_bar_arr([bd['bar'][bix-1], bd['bar'][bix]])
        return md_dict

def md_dict_from_mts_col(symbol, start_day, end_day, barsec, cols = ['open', 'close', 'vol','vbs','lpx','utc', 'absz','aasz','aspd','bdif','adif'],  use_live_repo=True, get_roll_adj=True):
    if use_live_repo:
        repo = mts_repo.MTS_REPO_Live()
    else:
        repo = mts_repo.MTS_REPO_TickData()
    bars, roll_adj_dict = repo.get_bars(symbol, start_day, end_day, barsec=barsec, cols = cols, is_mts_symbol=True, ignore_prev=False, remove_zero=False, get_roll_adj=get_roll_adj)
    return bars, roll_adj_dict

def md_dict_from_mts(symbol, start_day, end_day, barsec, use_live_repo=True, roll_adj_lpx=False):
    """
    if not give, use the open of the first bar on start_day.
    return:
        bar: shape(ndays,n,5)) for [lr, vol, vbs, lpx, utc]
        Note both the lr and lpx are roll adjusted
    """
    cols=['open', 'close', 'vol','vbs','lpx','utc']
    bars, roll_adj_dict = md_dict_from_mts_col(symbol, start_day, end_day, barsec, cols=cols, use_live_repo=use_live_repo, get_roll_adj=roll_adj_lpx)

    # construct md_days
    ndays, n, cnt = bars.shape
    bars=bars.reshape((ndays*n,cnt))

    # use the close/open as lr
    # for over-night lr, repo adjusts first bar's open as previous day's close
    # format of [lr, vol, vbs, lpx, utc]
    lr = np.log(bars[:,1]/bars[:,0])
    bars = np.vstack((lr, bars[:,2], bars[:,3], bars[:,4], bars[:,5])).T.reshape((ndays,n,5))
    if roll_adj_lpx:
        utc_col = 4
        lpx_col = 3
        bars=mts_repo.MTS_REPO.roll_adj(bars, utc_col, [lpx_col], roll_adj_dict)
    return bars

def get_md_days(symbol_arr, start_day, end_day, barsec, use_live_repo=True, roll_adj_lpx=False):
    md_dict = {}
    for symbol in symbol_arr :
        try :
            md_dict[symbol] = md_dict_from_mts(mts_repo_mapping_all[symbol], start_day, end_day, barsec=barsec, use_live_repo=use_live_repo, roll_adj_lpx=roll_adj_lpx)
        except :
            traceback.print_exc()
            print('failed to get symbol %s, skipping'%(symbol))
            
    return md_dict

def pick_md_days(md_dict, sday, eday):
    # sday and eday are index into the first dim of md_dict[key], i.e. days
    # sday and eday are inclusive, do not use -1 for eday, as eday+1 would be 0
    md_dict0 = {}
    for k in md_dict.keys() :
        md_dict0[k] = md_dict[k][sday:eday+1,:,:]
    return copy.deepcopy(md_dict0)

def merge_md_days(md_dict1, md_dict2):
    md_dict = {}
    for k in md_dict1:
        md_dict[k] = np.vstack((md_dict1[k], md_dict2[k]))
    return md_dict

