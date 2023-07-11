#!/usr/bin/python3

import numpy as np
import datetime
import copy
import os
import subprocess
import yaml
import traceback

import mts_util
import strat_sim
import symbol_map

class IDBO_Sim :
    def __init__(self, liveheader_path, strat_name = 'INTRADAY_MTS_IDBO_TF_TEST', strat_weight = 'STRATEGY_WEIGHTS.yaml', POD_id = None) :
        self.lhpath = liveheader_path
        self.lhname = 'LiveHO_'+ strat_name
        self.keys = ['signal_tf', 'time_from', 'time_to', 'thres_h', 'thres_l', 'sar_ds', 'sar_ic', 'sar_cl', 'pos_n', 'inactive_bars']
        self.swname = os.path.join(liveheader_path,strat_weight)
        self.sw = 0.0
        self.pod = POD_id
        try :
            with open(self.swname, 'r') as f:
                sw = yaml.safe_load(f)
                if POD_id is not None and len(POD_id)>0:
                    strat_name += ('_'+POD_id)
                self.sw = float(sw[strat_name])
        except :
            traceback.print_exc()
            print ('failed to get strategy weight for ' + strat_name, ', setting weight to 0')

        print ("got weight " + str(self.sw) + " from " + strat_name + " of " + self.swname)
        self.strat_name = strat_name

    def update_multiday(self, start_yyyymmdd, end_yyyymmdd, out_cfg_file) :
        tdi = mts_util.TradingDayIterator(start_yyyymmdd, end_yyyymmdd)
        d = tdi.begin()
        while d <= end_yyyymmdd :
            try :
                self.update(d, out_cfg_file)
            except :
                traceback.print_exc()
            d = tdi.next()

    def update(self, day_yyyymmdd, out_cfg_file=None, mts_symbol_arr=None, isLive=False) :
        """
        read the live and multiply file and update the idbo strategy cfg and risk 
        return a dict of all the defined symbol to be written to idbo_tf.cfg
        """
        idbo_cfg = {}
        strat_code = mts_util.StratUtil.strat_code_prefix(pod_id=self.pod)
        fname = self._get_file_name(day_yyyymmdd, isLive=isLive)
        with open(fname, 'r') as ly :
            lh = yaml.safe_load(ly)
            try :
                strat_code += str(lh["strategy_code"])
            except :
                if not isLive :
                    strat_code += str(lh["sub_strategy_code"])
                else :
                    raise ValueError("strategy_code not found in header file " + fname)

            trade_day_yaml = str(lh["trade_date"])
            if trade_day_yaml != day_yyyymmdd:
                raise ValueError("trade day mismatch! yaml has " + trade_day_yaml)

            lout = lh["live_output"]
            symbols = list(lout.keys())
            for s in symbols :
                if mts_symbol_arr is not None and s not in mts_symbol_arr :
                    continue
                idbo_cfg[s] = {}
                for k in self.keys: 
                    idbo_cfg[s][k] = lout[s]["live_output"][k]

                idbo_cfg[s]['pos_n'] *= self.sw

        if out_cfg_file is not None :
            self._write_LiveHeader(idbo_cfg, out_cfg_file, day_yyyymmdd, strat_code)

        return idbo_cfg, strat_code

    def _get_file_name(self, day_yyyymmdd, isLive=False) :
        """
        Historical Header name is supposed to be like LiveHO_INTRADAY_MTS_IDBO_TF_TEST-20210202.yaml
        Live Header name is supposed to be like LiveHO_INTRADAY_MTS_IDBO_TF_TEST.yaml
        """
        if not isLive :
            fn = self.lhname + '-' + day_yyyymmdd + '.yaml'
        else :
            fn = self.lhname + '.yaml'

        return os.path.join(self.lhpath, fn)

    def _write_LiveHeader(self, idbo_cfg, out_cfg_file, day_yyyymmdd, strat_code) :
        with open(out_cfg_file, "w") as f :
            f.write("update_date = " + day_yyyymmdd + "\n")
            f.write("strategy_code = " + str(strat_code) + "\n")
            f.write("strategy_weight = " + str(self.sw) + "\n")
            f.write("symbols = {\n")
            for k in idbo_cfg.keys() :
                f.write("    " + k + " = {\n")
                sv = idbo_cfg[k]
                for k0 in self.keys :
                    f.write("        " + k0 + " = " + str(sv[k0]) + "\n")
                f.write("    }\n\n")

            f.write("}\n")

def run_sim(start_day, end_day, mts_symbol_arr, clear_bar_dir=True) :
    liveheader_history_path = '/home/mts/upload/test/INTRADAY_MTS_IDBO_TF_TEST/liveho_history'
    out_cfg_file = '/home/mts/sim/config/strat/idbo_tf.cfg'
    sim = IDBO_Sim(liveheader_history_path)
    run_list_func = {"1": ["IDBO_TF", "config/strat/idbo_tf.cfg", sim.update]}

    repo_path = '/home/mts/data/mts'
    xml_path = 'config/symbol'
    max_N = 2
    assets_xml = "assets.xml"
    repo_object = strat_sim.create_mts_repo( repo_path=repo_path, xml_path=xml_path, max_N=max_N, assets_xml=assets_xml)

    bar_sec = [ 300 ]
    sim_name = "IDBO_TF_Sim"

    strat_sim.run_multi_day(sim_name, start_day, end_day, mts_symbol_arr, repo_object,  run_list_func,  bar_sec, clear_bar_dir=clear_bar_dir)


class IDBO_TF_Live (IDBO_Sim) :
    def __init__(self, liveheader_upload_path = '/home/mts/run/upload', strat_name = 'INTRADAY_MTS_IDBO_TF', POD_id = 'TSC'):
        super().__init__( liveheader_upload_path, strat_name, POD_id =POD_id)

    def updateLive(self, trading_day_yyyymmdd = None, main_cfg = 'config/main.cfg', update_risk=False, do_reload=True):
        # Live update from the uploaded header files
        # get today's trading day
        # get idbo_cfg, start_code

        # update the live_cfg using cfgupd
        #     1. get runlist, make sure strat_code (as strat_name) is in
        #     2. get the strat_cfg filename
        #     3. update the strat_cfg
        # update the risk.cfg using cfgupd
        #     1. get the risk on all the strat/symbol pair
        #     2. make sure the number is less than the pos_n

        print ("IDBO_TF Update Live")
        if trading_day_yyyymmdd is None :
            tdu = mts_util.TradingDayUtil()
            trading_day_yyyymmdd = tdu.get_trading_day(snap_forward=True)

        # get today's symbol map and check risk positions with model's trading symbols
        mts_cfg = mts_util.MTS_Config('config/main.cfg')
        xml_path = mts_cfg.get('XmlPath')
        max_N = mts_cfg.get('MaxN', int)
        assets_xml = mts_cfg.get('AssetsXml')
        sym_map = symbol_map.SymbolMap(xml_path = xml_path, max_N = max_N, assets_xml = assets_xml)
        smap = sym_map.get_tradable_map(trading_day_yyyymmdd, mts_key=True)

        # figure out the subscription list
        sub_symbol = sym_map.getSubscriptionList(mts_cfg, trading_day_yyyymmdd)

        #sub_venue = mts_cfg.getArr('MTSVenue')
        #sub_symbol= set(sym_map.get_mts_symbol_from_field(trading_day_yyyymmdd,  mts_cfg.getArr('MTSSymbol'), 'symbol'))
        #sub_symbol.update(sym_map.get_mts_symbol_from_mts_venue(sub_venue, trading_day_yyyymmdd))

        # update the model configuration according to the the liveHeader
        idbo_cfg, strat_code = self.update(trading_day_yyyymmdd, mts_symbol_arr = list(sub_symbol), isLive=True)
        strat_cfg = mts_cfg.get('Strat')
        mts_strat_cfg = mts_util.MTS_Config(strat_cfg)
        # make sure the strategy code is in the runlist, otherwise
        # raise exception!
        runlist = mts_strat_cfg.getArr('RunList')
        assert (strat_code in runlist), "strategy " + strat_code + " not in runlist " + str(runlist)

        model, model_cfg = mts_strat_cfg.getArr(strat_code)
        self._write_LiveHeader(idbo_cfg, model_cfg, trading_day_yyyymmdd, strat_code)

        # optional safety check on the idbo's max position agaist risk
        # check the risk based on the position 
        risk_cfg = mts_cfg.get('Risk')
        mts_risk_cfg = mts_util.MTS_Config(risk_cfg)
        
        # make sure the strategy is in the risk
        risk_strat_list = mts_risk_cfg.listSubKeys('strat')
        assert (strat_code in risk_strat_list), "strategy " + strat_code + " not in risk strategy list " + str(risk_strat_list)
        risk_key = 'strat.'+strat_code+'.Symbols'
        risk_symbols = mts_risk_cfg.listSubKeys(risk_key)

        mts_model_cfg = mts_util.MTS_Config(model_cfg)
        strat_sym = mts_model_cfg.listSubKeys('symbols')
        for ssym in strat_sym :
            key = 'symbols.'+ssym

            # ignore symbols with signal_tf != 1
            signal_tf = mts_model_cfg.get(key +'.signal_tf', int)
            if signal_tf != 1 :
                continue
            max_pos = mts_model_cfg.get(key+'.pos_n', float)
            symbol = smap[ssym]['symbol']
            assert (symbol in risk_symbols), "Symbol " + symbol + " defined in " + strat_code  + " not in risk!"
            risk_mul = 1.0
            risk_pos = mts_risk_cfg.get(risk_key + "." + symbol + ".position", float)
            assert (risk_pos*risk_mul > max_pos), "risk position for " + symbol + " is less than the strategy max position: " + str(risk_pos*risk_mul) + "<" + str(max_pos)

        # all good, reload the strategy
        if do_reload :
            self._reload_strat(strat_code, model_cfg)
            self._start_strat(strat_code)

    def _reload_strat(self, strat_name, cfg_file) :
        cmd_str = ["bin/flr", "@" + strat_name, "R", cfg_file]
        print ("Running " + str(cmd_str))
        subprocess.check_call(cmd_str)

    def _start_strat(self, strat_name):
        cmd_str = ["bin/flr", "@" + strat_name, "S"]
        print ("Running " + str(cmd_str))
        subprocess.check_call(cmd_str)

def DailyUpdate(strat_dict = {'TSC':['INTRADAY_MTS_IDBO_TF','INTRADAY_MTS_IDBO_TF_V2']}, do_reload=True, liveheader_upload_path = '/home/mts/upload') :
    strat_code_mapping = { 'INTRADAY_MTS_IDBO_TF_TSC': 'TSC-7000-335', \
                           'INTRADAY_MTS_IDBO_TF_V2_TSC': 'TSC-7000-337'}

    for pod_id in strat_dict.keys() :
        for sn in strat_dict[pod_id] :
            # check if the sn in the runlist of the config/strat.cfg
            sn0 = sn + '_' + pod_id
            if sn0 not in strat_code_mapping.keys() :
                raise RuntimeError(sn0 + " not in strat_code_mapping for idbo daily update")
            sc = strat_code_mapping[sn0]
            scfg = mts_util.MTS_Config('config/strat.cfg')
            rlist = scfg.getArr('RunList')
            if sc not in rlist :
                print (sn0 + ' with code ' + sc + ' not in strategy run list ' + str(rlist))
                continue
            idbo = IDBO_TF_Live(strat_name = str(sn), POD_id = str(pod_id), liveheader_upload_path = liveheader_upload_path)
            idbo.updateLive(do_reload=do_reload)

