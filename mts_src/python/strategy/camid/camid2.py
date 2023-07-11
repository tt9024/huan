import numpy as np
import datetime
import time
import copy
import sys
import os
import threading
import traceback
import mts_util

import strat_utils

###################
# CAMID2ZSC Logic #
###################
class CAMID2ZSC :
    def __init__(self, param, logger):
        self.param = param
        self.logger = logger
        self.spread_dict = get_symbol_spread_dict([param])
 
    def sod(self, trade_day=None, cur_utc=None, dump_file=None):
        strat_utils.init_symbol_config( \
                self.param, self.logger, \
                trade_day=trade_day, \
                symbol_spread_dict=self.spread_dict)
        strat_utils.init_state(self.param)
        
        self.pos = np.nan
        self.next_utc_ix = 0
        self.barsec = self.param['barsec'] 
        self.bar_utc = self.param['bar_utc']
        self.bar_trade = self.param['bar_trade']
        if cur_utc is None:
            cur_utc = int(datetime.datetime.now().strftime('%s'))
        self._catchup(cur_utc)

        # log initial parameters/states
        self.param.pop('bar_utc')
        self.param.pop('bar_trade')
        self.logger.logInfo('CAMID2ZSC started with param:\n%s\
           \nnext_utc_ix: %d (%s)'%(str(self.param), self.next_utc_ix,\
           str(datetime.datetime.fromtimestamp(self.bar_utc[self.next_utc_ix]))))

        # setup dump_file
        if dump_file is None:
            dump_file = '/home/mts/upload/STRATS/CAMID2ZSC/sim/live_dump_%s.csv'%(self.param['trade_day'])
        self.dump_file = open(dump_file, 'at')
        assert self.dump_file is not None
        self.logger.logInfo('dump file: %s'%(dump_file))

    def _catchup(self, cur_utc):
        # catch up if needed
        bar_utc = self.bar_utc[self.next_utc_ix:]
        if len(bar_utc)>=2 and cur_utc>=bar_utc[1]:
            ix = np.searchsorted(bar_utc,cur_utc)
            self.logger.logInfo('catching up with %d bars'%(ix))
            for sdict in self.param['submodels']:
                fn = sdict['bar_file']
                utc_arr=bar_utc[:ix]
                lpx_arr=strat_utils.get_bar_utc(fn,utc_arr)
                for utc, lpx in zip(utc_arr, lpx_arr):
                    self._update_signal(self.param, sdict, utc, lpx)
            self.next_utc_ix += ix

    def _update_signal(self, param, sdict, utc, lpx=None):
        # only update the 'signal' if in trading, 
        # otherwise, only update the bar data
        in_trading = False
        if utc >= param['trade_start_utc'] and \
           utc <= param['trade_end_utc'] +1 :
               in_trading = True

        # lpx is not None in _catchup() and offline
        # in live, set lbar to None to get the latest lpx
        lbar = [utc, lpx] if lpx is not None else None
        sig = strat_utils.update_signal(param, sdict, lbar=lbar, update_state_signal=in_trading)
        return in_trading, sig

    def on_sec(self, cur_utc, lbar_dict=None):
        if self.next_utc_ix >= len(self.bar_utc):
            return np.nan
        self._catchup(cur_utc)
        utc_bar = self.bar_utc[self.next_utc_ix]
        if cur_utc < utc_bar:
            return np.nan
        # update the bar
        comb_sig = 0
        for sdict in self.param['submodels']:
            cnt = 0
            sig = None
            lbar = None
            if lbar_dict is not None:
                # take the given market data
                lbar = lbar_dict[sdict['spread_symbol']][self.next_utc_ix]
            (utc0, lpx0) = lbar if lbar is not None else (cur_utc, None)
            while cnt < 100:
                in_trading, sig = self._update_signal(self.param, sdict, utc0, lpx0)
                if sig is not None or not in_trading:
                    break
                time.sleep(0.005)
                cnt+=1
            if in_trading :
                if sig is None:
                    self.logger.logError('problem in update signal for sdict %s'%(str(sdict)))
                    comb_sig = np.nan
                    break
                    #raise RuntimeError("problem updating siganl - sign is None in trading")
                comb_sig += sig

        self.next_utc_ix+=1
        if in_trading and not np.isnan(comb_sig):
            # in trading live update with new signal
            self.logger.logInfo('bar updated with signal %f'%(comb_sig))
            if (utc_bar%self.bar_trade)==0 or np.isnan(self.pos):
                pos = strat_utils.get_pos(comb_sig, self.param)
                self.pos = pos
        else :
            self.pos = np.nan
        self._dump(utc_bar)
        return self.pos

    def to_string(self):
        # dump format
        # pos [,sdict_dump]
        ret_str = '%.0f'%(self.pos)
        for sdict in self.param['submodels']:
            ret_str += (', '+strat_utils.dump_state(sdict))
        return ret_str

    def _dump(self, cur_utc):
        if self.dump_file is not None:
            self.dump_file.write('%s, %s\n'%(\
                datetime.datetime.fromtimestamp(cur_utc).strftime('%Y%m%d-%H:%M:%S'),\
                self.to_string()))
            self.dump_file.flush()
            
###################################
# Reading Config File into a Dict #
###################################
def read_config(config_file, strat_weight_file = 'STRATEGY_WEIGHTS.yaml', POD_id = 'TSC', config_path='/home/mts/upload'):
    cfg = mts_util.MTS_Config(os.path.join(config_path,config_file))
    param = {}
    param_live_output = []
    param['trade_day'] = cfg.get('trade_date')
    param['strat_code'] = mts_util.StratUtil.strat_code(cfg.get('strategy_code'), pod_id=POD_id)

    # read the strategy weight
    strat_key = cfg.get('strategy_key')
    param['strat_weight'] = strat_utils.get_strategy_weight(strat_key, os.path.join(config_path,strat_weight_file))
    symbols = cfg.listSubKeys('live_output')
    for sym in symbols:
        param_sym = copy.deepcopy(param)
        param_sym['trade_symbol'] = sym
        key = 'live_output.'+sym+'.'

        kvpair= [('barsec',        'bar_signal',     int), \
                 ('bar_trade',     'bar_trade',      int),\
                 ('time_from',     'time_from',      str), \
                 ('time_to',       'time_to',        str), \
                 ('smooth_n',      'smooth_n',       int), \
                 ('smooth_method', 'smooth_method',  str),\

                 ('trade_date',     'trade_date',     int), \
                 ('trade_contract', 'contract',       int), \
                 ('pos_mul',        'pos_mult',       float), \
                 ('mdl_n',          'mdl_n',          int)\
                ]

        for (k,v,t) in kvpair:
            param_sym[k] = cfg.get(key+v,t)

        # read the submodel list
        mdl_spreads = cfg.getArr(key+'mdl_spreads')
        mdl_pxprev = cfg.getArr(key+'mdl_pxprev',float)
        mdl_sigprev = cfg.getArr(key+'mdl_sig_prev',float)
        mdl_buffer = cfg.getArr(key+'mdl_buffer',float)
        mdl_entry = cfg.getArr(key+'mdl_entry',float)
        mdl_exit = cfg.getArr(key+'mdl_exit',float)
        mdl_sig_sf = cfg.getArr(key+'mdl_sig_sf',float)
        mdl_sig_m = cfg.getArr(key+'mdl_sig_m',float)
        mdl_sig_s = cfg.getArr(key+'mdl_sig_s',float)
        param_sym['submodels'] = []
        for i in range(param_sym['mdl_n']):
            sdict = {'spread_symbol':mdl_spreads[i],\
                     'pxprev':mdl_pxprev[i],\
                     'sigprev':mdl_sigprev[i],\
                     'buffer':mdl_buffer[i],\
                     'entry':mdl_entry[i],\
                     'exit':mdl_exit[i],\
                     'sig_sf':mdl_sig_sf[i],\
                     'sig_m':mdl_sig_m[i],\
                     'sig_s':mdl_sig_s[i],\
                     }
            param_sym['submodels'].append(sdict)
            
        param_live_output.append(param_sym)

    return param_live_output

def get_symbol_spread_dict(param):
    sd = {}
    for pm in param:
        for sm in pm['submodels']:
            sym = sm['spread_symbol']
            mts_sym = sym.split('-')[0].split('_')[0]
            assert mts_sym == sym.split('-')[1].split('_')[0], 'has to be a spread of same symbol'
            if mts_sym not in sd.keys():
                sd[mts_sym] = set()
            nlist = []
            for s0 in sym.split('-'):
                nlist.append(int(s0.split('_')[1][1:]))
            sd[mts_sym].add(tuple(nlist))
    for k in sd.keys():
        sd[k] = list(sd[k])
    return sd

def run_live(config_file = 'LiveHO-INTRADAY_MTS_CAMID2_DIFFZSC.cfg', launch_hour='0745'):
    logger=mts_util.MTS_Logger('CAMID2')
    # waiting for the configuration file to be updated
    # usually around 8am
    tdu = mts_util.TradingDayUtil()
    trade_day = tdu.get_trading_day(snap_forward=True)
    hour = int(launch_hour[:2])
    minute = int(launch_hour[2:])
    if hour >= 18:
        hour -=24
    launch_utc = int(datetime.datetime.strptime(trade_day,'%Y%m%d').strftime('%s')) + hour*3600 + minute*60
    logger.logInfo("waiting for launch time at %s"%(str(datetime.datetime.fromtimestamp(launch_utc))))
    while True:
        cur_utc = int(datetime.datetime.now().strftime('%s'))
        if cur_utc >=launch_utc:
            break
        time.sleep(1)

    param_live_output = read_config(config_file)
    logger=mts_util.MTS_Logger('CAMID2ZSC')
    strats = []
    for param in param_live_output:
        strats.append(CAMID2ZSC(copy.deepcopy(param), logger))

    logger.logInfo('created %d strategies'%(len(param_live_output)))
    for strat in strats:
        strat.sod()

    logger.logInfo('running %d strategies'%(len(param_live_output)))
    done = False
    while not done:
        try :
            cur_utc = int(datetime.datetime.now().strftime('%s'))
            for strat in strats:
                target_pos=strat.on_sec(cur_utc)
                if not np.isnan(target_pos):
                    # set target position during trading period
                    symbol = strat.param['trade_symbol']
                    strat_code = strat.param['strat_code']
                    scale = strat.param['strat_weight']
                    strat_utils.set_target_position(strat_code, symbol, target_pos*scale, logger)
            time.sleep(1)
        except KeyboardInterrupt as e :
            logger.logInfo("keyboard interrupt, exiting")
            done = True
        except Exception as e :
            traceback.print_exc()
            logger.logError("Exception: %s"%(str(e)))
            done = True

def run_offline(trade_day=None, \
                 config_file='LiveHO-INTRADAY_MTS_CAMID2_DIFFZSC.cfg', \
                 config_path = '/home/mts/upload', \
                 dump_file_path = '/home/mts/upload/STRATS/CAMID2ZSC/sim'):
    param_offline = read_config(config_file, config_path = config_path)
    if trade_day is None:
        trade_day = param_offline[0]['trade_day']
    logger=mts_util.MTS_Logger('CAMID2ZSC_Offline', logger_file='/tmp/camid2zsc_sim_log')

    strats = []
    for param in param_offline:
        strats.append(CAMID2ZSC(copy.deepcopy(param), logger))

    # setup dump file from trade_day
    dump_file = os.path.join(dump_file_path, 'offline_dump_%s.csv'%(trade_day))
    logger.logInfo('created %d strategies'%(len(param_offline)))
    # starting from the trade day's (previous day's) 6pm
    for strat in strats:
        param = strat.param
        logger.logInfo('simulating %s %s'%(param['strat_code'], param['trade_symbol']))
        cur_utc = int(datetime.datetime.strptime(trade_day,'%Y%m%d').strftime('%s'))-6*3600
        strat.sod(dump_file=dump_file, trade_day=trade_day, cur_utc=cur_utc)
        # setup market data
        lbar_dict = {}
        utc_array = strat.bar_utc

        for sdict in strat.param['submodels']:
            bar_file = sdict['bar_file']
            symbol = sdict['spread_symbol']
            lbar_dict[symbol] = np.vstack((utc_array, strat_utils.get_bar_utc(bar_file, utc_array))).T

        # simulating on each bar time
        for cur_utc in utc_array:
            strat.on_sec(cur_utc, lbar_dict=lbar_dict)
