import numpy as np
import datetime
import time
import copy
import os
import subprocess
import traceback
import sys
import threading
import copy

import mts_util
import symbol_map
import ar1_md
import ar1_model
import ar1_sim_ens

Default_ENS_Config = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US/ENS.cfg'
Default_REF_Config = '/home/mts/upload/ZFU_STRATS/INTRADAY_MTS_AR1_WTI_US/LiveHO-INTRADAY_MTS_AR1_WTI_ENS.cfg'
Default_Logger_File = '/tmp/ar1_ens'
MKT = 'WTI_N1'

class AR1_WTI_Ens:
    def __init__(self, mkt, logger_file, ens_cfg_fname, ref_cfg_fname=None):
        self.logger =  mts_util.MTS_Logger('AR1_%s'%(mkt), logger_file = logger_file)
        self.mkt = mkt
        self.ens_cfg_fname = ens_cfg_fname
        self.ref_cfg_fname = ref_cfg_fname
        self.should_run = False

    def start(self):
        # read the ens config and ref config
        # create md sod
        # load ens models and sod() passing in param
        # load ref models and sod() passing in param
        # generate trigger
        # setup trading time
        self.mp_dict, self.param_dict = ar1_sim_ens.read_ens_cfg(self.ens_cfg_fname, self.mkt)
        self.ens_weight = self.mp_dict['ens_weight']
        self.ref_weight = self.mp_dict['ref_weight']
        self.logger.logInfo('starting AR1_ENS, loaded mp_dict:%s'%(str(self.mp_dict)))
        self.param_ref = None
        if self.ref_cfg_fname is not None:
            self.param_ref = ar1_model.read_ar1_config(self.ref_cfg_fname, self.mp_dict['symbol'])
            self.logger.logInfo('loaded a referene model:%s'%(str(self.param_ref)))
            assert self.param_ref['n']==self.mp_dict['n']
            assert self.param_ref['barsec']==self.mp_dict['barsec']
        else:
            self.ref_weight = 0

        self.n = int(self.mp_dict['n'])
        self.barsec = int(self.mp_dict['barsec'])
        self.md = ar1_md.AR1Data(self.barsec)

        # setup scale and time
        self.sw_tsc = self.mp_dict['sw_tsc']
        self.sw_tsd = self.mp_dict['sw_tsd']
        self.sn = self.mp_dict['strategy_key']
        self.logger.logInfo('getting the scale: sw_tsc:%f, sw_tsd:%f'%(self.sw_tsc, self.sw_tsd))

        dtnow = datetime.datetime.now()
        utcnow = int(dtnow.strftime('%s'))
        self.tdu = mts_util.TradingDayUtil()
        trade_day = self.tdu.get_trading_day_dt(dtnow, snap_forward=True)
        t0 = self.tdu.get_start_utc(trade_day)  # t0/t1 is model run time, not same as t0_trd/t1_trd
        t1 = self.tdu.get_end_utc(trade_day)
        self.overnight_utc=t0

        # to specify an overnight hour, set 'trading_hour', for example, to [ 7, 21]
        # to allow trading between 18:00 to 21:00
        trd_hours = self.mp_dict['trading_hour']  # i.e. [-6, 17]
        # t0_trd to t1_trd allowed to trade
        self.t0_trd = int(int(datetime.datetime.strptime(trade_day, '%Y%m%d').strftime('%s')) + float(trd_hours[0])*3600+0.5)
        self.t1_trd = self.t0_trd + int((float(trd_hours[1])-float(trd_hours[0]))*3600+0.5)
        if self.t1_trd > t1+3600:  # allow over-night trade
            self.overnight_utc=self.t1_trd-24*3600
        self.logger.logInfo('setup time: model runtime: (%s-%s), trading time: (%s-%s, overnight_utc: %s)'% \
                (datetime.datetime.fromtimestamp(t0).strftime('%Y%m%d-%H:%M:%S'), \
                 datetime.datetime.fromtimestamp(t1).strftime('%Y%m%d-%H:%M:%S'), \
                 datetime.datetime.fromtimestamp(self.t0_trd).strftime('%Y%m%d-%H:%M:%S'), \
                 datetime.datetime.fromtimestamp(self.t1_trd).strftime('%Y%m%d-%H:%M:%S'), \
                 datetime.datetime.fromtimestamp(self.overnight_utc).strftime('%Y%m%d-%H:%M:%S')))

        # setup reference model time
        self.overnight_utc_ref=t0
        self.t0_trd_ref = self.t0_trd
        self.t1_trd_ref = self.t1_trd
        if self.param_ref is not None:
            trd_hours_ref = self.param_ref['trading_hour']  # i.e. [5.75, 23.9]
            self.t0_trd_ref = int(int(datetime.datetime.strptime(trade_day, '%Y%m%d').strftime('%s')) + float(trd_hours_ref[0])*3600+0.5)
            self.t1_trd_ref = self.t0_trd_ref + int((float(trd_hours_ref[1])-float(trd_hours_ref[0]))*3600+0.5)
            if self.t1_trd_ref > t1+3600:  # allow over-night trade
                self.overnight_utc_ref=self.t1_trd_ref-24*3600
            self.logger.logInfo('setup time: ref model runtime: (%s-%s), trading time ref: (%s-%s, overnight_utc_ref: %s)'% \
                    (datetime.datetime.fromtimestamp(t0).strftime('%Y%m%d-%H:%M:%S'), \
                     datetime.datetime.fromtimestamp(t1).strftime('%Y%m%d-%H:%M:%S'), \
                     datetime.datetime.fromtimestamp(self.t0_trd_ref).strftime('%Y%m%d-%H:%M:%S'), \
                     datetime.datetime.fromtimestamp(self.t1_trd_ref).strftime('%Y%m%d-%H:%M:%S'), \
                     datetime.datetime.fromtimestamp(self.overnight_utc_ref).strftime('%Y%m%d-%H:%M:%S')))

        next_bar_utc = t0+self.barsec
        k = 0

        # load the models
        pyres_symbol = self.mp_dict['symbol']
        mdl_dict = {}
        for mdl_id in self.mp_dict['ens_model'].keys():
            if len(self.mp_dict['ens_model'][mdl_id]) == 0:
                continue
            mdl_dict[mdl_id] = ar1_model.AR1_Model(self.logger, self.ens_cfg_fname, pyres_symbol)
            mdl_dict[mdl_id].sod(trade_day, param_in = self.param_dict[mdl_id], check_trade_day=False)
            self.logger.logInfo('loaded ens model %d'%(int(mdl_id)))
        self.logger.logInfo('%d ens models loaded sod'%(len(mdl_dict.keys())))
        self.mdl_dict = mdl_dict

        mdl_ref = None
        if self.param_ref is not None:
            mdl_ref = ar1_model.AR1_Model(self.logger, self.ref_cfg_fname, pyres_symbol)
            mdl_ref.sod(trade_day=trade_day, check_trade_day=False, param_in=self.param_ref)
            self.logger.logInfo('loaded reference model')
        self.mdl_ref = mdl_ref

        self.logger.logInfo('setting up trigger time')
        if self.mdl_ref is not None:
            stdv = self.mdl_ref.fm.state_obj.std.v[0,:]
        else:
            assert len(self.mdl_dict) > 0, 'no ens models loaded and no ref model'
            stdv = self.mdl_dict[ list(self.mdl_dict.keys())[0] ].fm.state_obj.std.v[0,:]

        self.trigger_time = ar1_model.gen_trigger_time(stdv, self.mp_dict['barsec'], self.mp_dict['trigger_cnt'])
        trigger_utc = (np.r_[1,self.trigger_time]+t0).astype(int)
        trigger_utc = np.r_[trigger_utc, t1+self.barsec].astype(int)

        cur_utc = int(datetime.datetime.now().strftime('%s'))
        tr_ix = np.clip(np.searchsorted(trigger_utc, cur_utc)-1, 0, len(trigger_utc)+1)
        self.logger.logInfo('trigger time: %i:%s\n%s'%(tr_ix, str(datetime.datetime.fromtimestamp(trigger_utc[tr_ix])), str(trigger_utc)))

        # intialize the md
        self.md.sod()
        self.should_run=True
        while cur_utc < t1 + self.barsec and k < self.n and self.should_run:
            try :
                # make sure we are update-to-date with the bar
                p0 = None

                """
                # debug
                self.logger.logInfo('cur_utc %i:%s-%i:%s'%(k, \
                        datetime.datetime.fromtimestamp(next_bar_utc).strftime('%Y%m%d-%H:%M:%S'), \
                        tr_ix, datetime.datetime.fromtimestamp(trigger_utc[tr_ix]).strftime('%Y%m%d-%H:%M:%S')))
                """
                while next_bar_utc <= cur_utc :
                    md_dict = self.md.on_bar(next_bar_utc)
                    if md_dict is None :
                        # wait a bit
                        time.sleep(0.001)
                        continue
                    p0, p0_ref = self._ens_on_bar(k, md_dict)
                    self.logger.logInfo('run on_bar of %i:%s\nmd=%s'%(k, datetime.datetime.fromtimestamp(next_bar_utc).strftime('%Y%m%d-%H:%M:%S'),str(md_dict[pyres_symbol])))
                    next_bar_utc += self.barsec
                    k+=1
                    cur_utc = int(datetime.datetime.now().strftime('%s'))

                # we are out into the current bar, work with trigger
                if trigger_utc[tr_ix] <= cur_utc :
                    a = 1.0-float(next_bar_utc-cur_utc)/float(self.barsec)
                    md_dict = self.md.get_snap()
                    p0, p0_ref = self._ens_on_snap(k, a, md_dict)
                    self._set_pos(p0, p0_ref)
                    self.logger.logInfo('run on_snap k=%i, pos=(%f), a=%f\nmd=%s'%(k,p0,a,str(md_dict[pyres_symbol])))
                    tr_ix += 1
                else :
                    # guard against running trigger right after running bar
                    if p0 is not None:
                        self.logger.logInfo('onbar - setting position of %i'%(p0))
                        self._set_pos(p0, p0_ref)

                time.sleep(0.01)
                cur_utc = int(datetime.datetime.now().strftime('%s'))
            except KeyboardInterrupt as e :
                self.logger.logInfo("keyboard interrupt, exiting")
                return
            except :
                self.logger.logError("Exception while running: " + traceback.format_exc())
                time.sleep(1)

        # eod
        if self.should_run:
            self.logger.logInfo("Done for the day, running eod...")
            for mdl_id in self.mdl_dict.keys():
                self.mdl_dict[mdl_id].eod()
            if self.mdl_ref is not None:
                self.mdl_ref.eod()
        else:
            self.logger.logInfo("Stopped intraday...")
        self.logger.logInfo("Exit!")

    def _ens_on_bar(self, k, md_dict):
        pos_dict = {}
        karr = np.array([k+1])  # k=0 and a = 1 at 18:05, ref agg_pos_dict()
        for mdl_id in self.mdl_dict.keys():
            model = self.mdl_dict[mdl_id]
            p0 = model.on_bar(k, md_dict)
            pos_dict[mdl_id] = np.array([p0])
        p0 = ar1_sim_ens.agg_pos_day(self.mp_dict, pos_dict, karr, log_func=self.logger.logInfo)[0]
        self.logger.logInfo('ens_on_bar (k=%d): pos_dict(%s), p0(%f)'%(k, str(pos_dict), p0))
        p0_ref = 0
        if self.mdl_ref is not None:
            p0_ref = self.mdl_ref.on_bar(k, md_dict)
            self.logger.logInfo('ref_on_bar(k=%d): p0_ref(%f)'%(k,p0_ref))
        return p0, p0_ref

    def _ens_on_snap(self, k, a, md_dict):
        pos_dict = {}
        karr = np.array([k+a])  
        for mdl_id in self.mdl_dict.keys():
            model = self.mdl_dict[mdl_id]
            p0 = model.on_snap(k, a, md_dict)
            pos_dict[mdl_id] = np.array([p0])
        p0 = ar1_sim_ens.agg_pos_day(self.mp_dict, pos_dict, karr, log_func=self.logger.logInfo)[0]
        self.logger.logInfo('ens_on_snap (k=%d,a=%.2f): pos_dict(%s), p0(%f)'%(k,a,str(pos_dict), p0))
        p0_ref = 0
        if self.mdl_ref is not None:
            p0_ref = self.mdl_ref.on_snap(k, a, md_dict)
            self.logger.logInfo('ref_on_snap(k=%d,a=%.2f): p0_ref(%f)'%(k,a,p0_ref))
        return p0, p0_ref

    def _set_pos(self, qty, qty_ref, symbol = 'WTI_N1'):
        # set position for symbol with ens target position and reference target pos

        # check trading time
        cur_utc = int(datetime.datetime.now().strftime('%s'))
        if (cur_utc > self.overnight_utc) and (cur_utc < self.t0_trd or cur_utc > self.t1_trd) :
            self.logger.logInfo('qty %i not set, not in trading: %d to %d, (overnight %d)'%(qty, self.t0_trd, self.t1_trd, self.overnight_utc))
            qty = 0
        if (cur_utc > self.overnight_utc_ref) and (cur_utc < self.t0_trd_ref or cur_utc > self.t1_trd_ref) :
            self.logger.logInfo('ref qty %i not set, not in ref trading: %d to %d, (overnight %d)'%(qty_ref, self.t0_trd_ref, self.t1_trd_ref, self.overnight_utc_ref))
            qty_ref = 0
            if self.mdl_ref is not None:
                self.mdl_ref.reset_prev_pos()

        ens_maxpos = self.mp_dict['ens_maxpos']
        tgt_qty = np.round(np.clip(qty*self.ens_weight + qty_ref*self.ref_weight, -ens_maxpos, ens_maxpos))
        run_str_tsc = ['/home/mts/run/bin/flr',  'X',  'TSC-7000-'+str(self.mp_dict['strat_code']) + ', ' + symbol + ', ' + str(np.round(tgt_qty*self.sw_tsc)) + ', Y10m']
        run_str_tsd = ['/home/mts/run/bin/flr',  'X',  'TSD-7000-'+str(self.mp_dict['strat_code']) + ', ' + symbol + ', ' + str(np.round(tgt_qty*self.sw_tsd)) + ', Y10m']
        try :
            for run_str in [run_str_tsc, run_str_tsd]:
                # debug
                self.logger.logInfo(str(run_str))
                ret = subprocess.check_output(run_str)
            self.logger.logInfo("run command for qty(%f) sw(%f,%f)"%(tgt_qty, self.sw_tsc,self.sw_tsd))
        except subprocess.CalledProcessError as e:
            self.logger.logError("Failed to run command " + str(run_str) + ", return code: " + str(e.returncode) + ", return output: " + str(e.output))

