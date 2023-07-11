import numpy as np
import datetime
import time
import copy
import os
import subprocess
import yaml
import traceback
import sys
import threading

import mts_util
import symbol_map
import tpmon

"""
A Reference Parameter from Camid2, other strategies
could take similar forms

param = { \
        'spread_list':\
        [ \
             { \
             'symbol':'Brent_N1-Brent_N6', \
             'bar_file': '',\
             'state': { \
                 'prev_px':[], \
                 'bar_array':np.zeros(5), \
                 'last_utc': 0,\
                 'cur_sig': 0\
                 } \
             } \
        ], \
        'trade_symbol': 'Brent_N1',\
        'strat_code': 'TSC-7000-1', \
        'strat_weight': 1.0, \
        'barsec':300, \
        'smooth_n':5, \
        'smooth_method':'median',\
        'bar_trade':900 ,\
        'alpha':0.00361663652802893297, \
        'scale':4000, \
        'thres':0, \
        'buffer':0.1, \
        'ceil':5, \
        'pos_mul':27.35293614,\
        'time_from': '0800', \
        'time_to':   '1430', \
        'start_utc': 0, \
        'end_utc': 0, \
        'prev_end_utc':0, \
        'trade_day': '20220215', \
        'last_ewma_time':'2022-02-14 1800',\
        }
"""

#####################
# Market Data Utils #
#####################
def get_bar(barfile, barcnt):
    try:
        lines = subprocess.check_output(['tail', '-n', str(barcnt), barfile]).decode().split('\n')[:-1]
        # bar line in the format of
        # bar_utc, open, high, low, close, vol, last_px, last_micro, vbs
        bars = []
        utc_col = 0
        close_px_col = 4
        for line in lines:
            v=line.split(',')
            bars.append([int(v[utc_col]), float(v[close_px_col])])
        return np.array(bars)
    except :
        traceback.print_exc()
        return None

def get_bar_utc(barfile, utc_array, max_search_cnt = 3600*24*2):
    utc_array = np.array(utc_array).astype(int)
    bars = get_bar(barfile, max_search_cnt)
    if bars is None:
        raise RuntimeError('Failed to get bar from %s'%(barfile))
    bcnt = len(bars)
    ix = np.clip(np.searchsorted(bars[:,0].astype(int),utc_array),0,bcnt-1)
    ixnz = np.nonzero(bars[ix,0].astype(int) != utc_array)[0]
    if len(ixnz) > 0:
        ix[ixnz] = np.clip(ix[ixnz]-1,0,bcnt-1)
    return bars[ix,1]

def get_latest_bar(barfile, last_utc):
    lbar = get_bar(barfile, 1)
    if lbar is None or lbar[0,0]<=last_utc:
        return None
    return lbar[0,:]


##################################
# get bar with column seletion  ##
##################################
def get_bar_cols(barfile, barcnt, cols=[0,1,2,3,4]):
    """
    bar columns from live bar file:
    bar_utc, open, high, low, close, vol, last_px, last_micro, vbs

    same as get_bar(), with column selection
    input:
        cols: the index into the above list
    """
    try:
        lines = subprocess.check_output(['tail', '-n', str(barcnt), barfile]).decode().split('\n')[:-1]
        # bar line in the format of
        # bar_utc, open, high, low, close, vol, last_px, last_micro, vbs
        bars = []
        for line in lines:
            v=line.split(',')
            b = []
            for c in cols:
                if c == 0:
                    b.append(int(v[c]))
                else :
                    b.append(float(v[c]))
            bars.append(b)
        return np.array(bars)
    except :
        traceback.print_exc()
        return None

def get_bar_file(symbol, trade_day, barsec, symbol_spread_dict={}):
    # could throw if not a trading day
    smap = symbol_map.SymbolMap()
    tinfo = smap.get_tinfo(symbol, trade_day, True, symbol_spread_dict = symbol_spread_dict)
    fn = tpmon.get_bar_file(tinfo['tradable'], tinfo['venue'], int(barsec))
    return fn

def get_symbol_day_detail(symbol, trade_day, barsec, symbol_spread_dict={}, smap=None):
    # could throw if not a trading day
    if smap is None:
        smap = symbol_map.SymbolMap()
    tinfo = smap.get_tinfo(symbol, trade_day, True, symbol_spread_dict = symbol_spread_dict)
    live_bar_file = tpmon.get_bar_file(tinfo['tradable'], tinfo['venue'], int(barsec))
    contract = tinfo['contract_month']
    contract_size = int(float(tinfo['point_value']))
    tick_size = tinfo['tick_size']
    start_time = tinfo['start_time']
    end_time = tinfo['end_time']
    return live_bar_file, contract, float(contract_size), float(tick_size), start_time, end_time

##############################
# Smooth => EWMA => Position #
##############################
def get_smooth(param, bar_array) :
    assert param['smooth_method'] == 'median'
    smooth_n = param['smooth_n']
    return np.median(bar_array[-smooth_n:])

def rolling_window(x, roll_period, roll_func, initial_val=None) :
    """
    getting a rolling value of input 'x' using roll_func.  
    input:
        x: 1d array
        roll_period: int, i.e. 14
        roll_func: expect to be able to run with roll_func(xv, axis=1),
                   i.e. np.min/max/mean/std/sum/cumsum, etc
        initial_val: the value to use before roll_period can be obtained, 
                     i.e. the initial roll_period-1 
                     if None, then uses the x[0]
    return:
        y: same length of x
    """
    from scipy.linalg import hankel
    nd = len(x)
    if roll_period == 1:
        return roll_func(x.reshape((nd,1)),axis=1)

    assert roll_period <= nd, "roll_period must not be more than data length"
    x0 = initial_val
    if x0 is None:
        x0 = x[0]
    x0=np.array([x0]*(roll_period-1))
    xv=hankel(np.r_[x0, x[:-roll_period+1]],x[-roll_period:])
    return roll_func(xv,axis=1)

##########################
# State init from config #
##########################
def get_strategy_weight(strategy_key, strat_weight_file):
    try :
        with open(strat_weight_file, 'r') as f:
            sw = yaml.safe_load(f)
            return float(sw[strategy_key])
    except :
        raise ValueError("failed to get weight for %s from %s!"%(strategy_key,strat_weight_file))

# submodels instead of spread_list
def init_symbol_config(param, logger, trade_day=None, symbol_spread_dict={}):
    sym_info = {}
    sym_list = []
    if trade_day is None:
        tdu = mts_util.TradingDayUtil()
        trade_day = tdu.get_trading_day(
                yyyymmdd_hh_mm_ss=trade_day,snap_forward=True)
    assert trade_day == param['trade_day'], \
            "current trading day (%s) mismatch with config (%s)"%(trade_day, param['trade_day'])

    # getting the bar files for spreads
    smap = symbol_map.SymbolMap()
    barsec = param['barsec']
    for slist in param['submodels']:
        sym = slist['spread_symbol']
        tinfo = smap.get_tinfo(sym, trade_day, True, symbol_spread_dict = symbol_spread_dict)
        fn = tpmon.get_bar_file(tinfo['tradable'], tinfo['venue'], 1)
        slist['bar_file'] = fn

    # getting the start/stop times
    utc0 = int(datetime.datetime.strptime(trade_day, '%Y%m%d').strftime('%s'))
    ssutc = []
    for tstr in [tinfo['start_time'], tinfo['end_time']]:
        ts = tstr.split(':')
        ssutc.append(int(ts[0])*3600+int(ts[1])*60+int(ts[2]))
    if ssutc[0] >= ssutc[1]:
        ssutc[0]-=3600*24
    param['start_utc'] = utc0+ssutc[0]+barsec
    param['end_utc'] = utc0+ssutc[1]
    param['bar_utc'] = np.arange(param['start_utc'],
         param['end_utc']+barsec,barsec).astype(int)

    # getting the trade from/to
    ssutc = []
    for tstr in [param['time_from'], param['time_to']]:
        t=int(tstr[:2])*3600+int(tstr[2:4])*60+utc0
        if t>param['end_utc']:
            t-=(3600*24)
        ssutc.append(t)
    assert ssutc[0] <= ssutc[1], 'time_from not before time_to'
    param['trade_start_utc'] = ssutc[0]
    param['trade_end_utc'] = ssutc[1]

    # getting previous end time
    tdi = mts_util.TradingDayIterator(trade_day)
    tdi.begin()
    try :
        prev_trade_day = tdi.prev()
        tinfo = smap.get_tinfo(sym, prev_trade_day, True)
        logger.logInfo('got previous trading day of '+ prev_trade_day)
    except :
        logger.logInfo('problem getting previous trading day, try earlier day')
        prev_trade_day = tdi.prev()
    ts=tinfo['end_time'].split(':')
    utc0 = int(datetime.datetime.strptime(prev_trade_day, '%Y%m%d').strftime('%s'))
    param['prev_end_utc'] = utc0 + int(ts[0])*3600+int(ts[1])*60+int(ts[2])

# Previous end of day signal is in the LiveHO
def init_state(param):
    """
        [ \
             { \
             'spread_symbol':'Brent_N1-Brent_N2', \
             'bar_file': '',\
             'state': { \
                 'prev_px':[], \
                 'bar_array':np.zeros(5), \
                 'last_utc': 0,\
                 'cur_ma':0 \
                 } \
             } \
        ], \
    """
    barcnt = param['smooth_n']
    barsec = param['barsec']
    utc_array = param['prev_end_utc']-np.arange(barcnt).astype(int)[::-1]*barsec

    for sdict in param['submodels']:
        fn = sdict['bar_file']
        prev_sig = sdict['sigprev']
        prev_bar = get_bar_utc(fn, utc_array)
        prev_px = get_smooth(param, prev_bar)
        state = { \
                'prev_px':[prev_px], \
                'bar_array':prev_bar, \
                'last_utc': utc_array[-1],\
                'cur_sig':prev_sig,
                }
        sdict['state']=copy.deepcopy(state)

def update_signal(param, sdict, lbar=None, update_state_signal=True):
    if lbar is None:
        # get the latest
        barfile = sdict['bar_file']
        barsec = param['barsec']
        last_utc = sdict['state']['last_utc']
        # normalize last_utc w.r.t barsec
        last_utc = int(np.round(float(last_utc)/barsec)*barsec)
        lbar = get_latest_bar(barfile, last_utc)
        if lbar is None:
            print('lbar is None ' + barfile + str(last_utc))
            return None
        if lbar[0] < last_utc+barsec:
            print('lbar[0] < last_utc+barsec: %d %d %d %s'%(lbar[0], last_utc, barsec, \
                    str(datetime.datetime.now())))
            return None

    state_sym = sdict['state']
    last_utc, lpx = lbar
    state_sym['last_utc']=last_utc
    state_sym['bar_array']=np.r_[state_sym['bar_array'][1:],lpx]
    smooth_px = get_smooth(param, state_sym['bar_array'])
    sig = None
    if update_state_signal:
        # Signal is Z-score of daily ret - standardized
        # Get current return
        cur_ret = smooth_px - sdict['pxprev']
        # Calc 'z-score'
        zeta = (cur_ret - sdict['sig_m']) / sdict['sig_s']
        # Apply scale factor
        zsc = zeta * sdict['sig_sf']
        # Apply buffer to get raw signal
        buf = sdict['buffer']
        raw = int(abs(zsc)/buf)*buf * np.sign(zsc)
        # Get previous binary signal
        prev_sig = state_sym['cur_sig']
        # Get new binary signal
        # No position -> Check entry threshold
        if prev_sig == 0 and abs(raw) >= sdict['entry']:
            sig = np.sign(raw)
        # Position on -> Check exit threshold
        elif prev_sig > 0 and raw <= sdict['exit']:
            sig = 0
        elif prev_sig < 0 and raw >= -sdict['exit']:
            sig = 0
        # Keep previous position
        else:  
            sig = prev_sig

        # Update state
        sig = np.sign(sig)
        state_sym['cur_sig'] = sig

    state_sym['prev_px'].append(smooth_px)  # not used in current version
    return sig

def get_pos(ret_ma, param):
    pos_mul = param['pos_mul']
    sig = ret_ma
    sigs = np.sign(sig)
    sig*=sigs
    pos = int(sig*pos_mul+0.5)*sigs
    return pos

def dump_state(sdict):
    # dump format 
    # spread_symbol, ma, last_px, smoothed_px, last_utc
    return '%s, %.7f, %.7f, %.7f, %d'%\
            (sdict['spread_symbol'], \
             sdict['state']['cur_sig'], \
             sdict['state']['bar_array'][-1],\
             sdict['state']['prev_px'][-1], \
             sdict['state']['last_utc'])

def pnl_from_p0(p00, p0, lpx, tcost, fee = 2.0, contract_size=1000, lr0=0) :
    """
    p00: scalar, the starting position
    p0: len(n) integer vector, target position at bar k, k=0,1,2,...,n-1.
        p0[k] is the target position at close of bar k.  bar 0 is the first
        bar from 6:00 to  6:05pm
    lpx: len(n) vector, price at close of each bar
    tcost: scalar, usually half of spread plus a slippage, in price
    fee: scalar, fee per transaction, regardless of size
    lr0: the first log return at 6:05, used to apply p00 for initial pnl

    return:
       pnl, pnl_cost - 
             length n+1 vector, pnl[k] is the value of cash+asset at close of bar k.
             the last pnl is the ending value.
    """
    p0d = p0-np.r_[p00,p0[:-1]]
    trd_cst = np.abs(p0d)*tcost*contract_size+np.abs(np.sign(p0d))*fee
    lpx0 = np.r_[lpx[0]/np.exp(lr0),lpx]
    pnl = (lpx0[1:]-lpx0[:-1])*np.r_[p00,p0[:-1]]* contract_size
    #pnl = np.r_[0,(lpx[1:]-lpx[:-1])*p0[:-1]]* contract_size
    pnl_cst = pnl-trd_cst
    return pnl, pnl_cst

##################
# MTS Floor Util #
##################
def set_target_position(strat, symbol, qty, logger, twap_minutes = 15, trader_type='T'):
    # set position for symbol with target position to be qty
    run_str = ['/home/mts/run/bin/flr',  'X',  strat + ', ' + symbol + ', ' + str(qty) + ', %s%dm'%(trader_type,twap_minutes)]
    try :
        ret = subprocess.check_output(run_str)
    except subprocess.CalledProcessError as e:
        if logger is not None:
            logger.logError("Failed to run command " + str(run_str) + ", return code: " + str(e.returncode) + ", return output: " + str(e.output))

def get_mts_position(strat, symbol, logger=None, verbose=False):
    run_str = ['/home/mts/run/bin/flr', 'P', ',']
    symbol_str = symbol.split('_')[0]
    try:
        ret = subprocess.check_output(run_str)
        lines = ret.decode().split('\n')[:-1]
        pos = []
        for line in lines:
            line = ' '.join(line.split())
            if strat in line and symbol_str in line:
                pos.append(int(line.split(' ')[2]))
        if len(pos) == 0:
            if verbose:
                if logger is not None:
                    logger.logInfo('position not found for strategy %s, symbol %s'%(strat, symbol_str))
        if len(pos) > 1:
            if logger is not None:
                logger.logInfo('%d positions found for strategy %s, symbol %s: %s'%(len(pos), strat, symbol_str, str(pos)))
        return np.sum(pos)
    except subprocess.CalledProcessError as e:
        if logger is not None:
            logger.logError("Failed to run command " + str(run_str) + ", return code: " + str(e.returncode) + ", return output: " + str(e.output))
    except Exception as e:
        if logger is not None:
            logger.logError("Failed to run command " + str(run_str) + ", return code: " + str(e))
    return None
