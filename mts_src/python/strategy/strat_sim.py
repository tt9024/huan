import numpy as np
import datetime
import copy
import json
import os
import traceback

import symbol_map as symmap
import mts_repo
import mts_util

ConfigPath =      'config'
MainCFG =         ConfigPath+"/main.cfg"
MainCFGTemplate = ConfigPath+"/main.cfg.template"
StratCFG =        ConfigPath+"/strat.cfg"
SymbolMapCFG =    "symbol_map.cfg"
EodPosCSV =       "recovery/eod_pos.csv"
PosTraceCSV =     "recovery/pos_trace.csv"

StratSim = "bin/strat_sim"

def sim_1day(sim_name, trade_day, run_list, mts_bar, bar_sec, symbol_map_obj) :
    """
    Simulates one day of strateiges defined in run_list
    using market data in mts_bar.  It returns each strategy's
    position and pnl time serieses.

    Input:
    sim_name:  name of the simulation, i.e. "idbo_tf_sim1"
    trade_day: YYYYMMDD, i.e. "20210325"
    run_list:  a dict of format {strat_name: [strat_type, strat_cfg], ... }
               i.e. {"IDBO_TF001": ["IDBO_TF", "config/strat/idbo_tf.cfg"]}
    mts_bar:   an array of format [ [ tradable, bar_file ], ... ]
               i.e. [ [ "ESH1", "bar/ESH1_20210325.csv" ] ... ]
               where bar_file is the MTS bar file of 1 second, used to
               generate market data updates during simulation.
    bar_sec:   an array of integer i.e. [ 300, 5 ], the bar periods
               at which bar writer updates "realtime" bar used by strategies.

    Return: 
    boolean, false if the simulation fails.  It could raise.

    Note: it first creates the config/strat.cfg, then updates the config/main.cfg,
          and finally runs one day of simulation, and the simulation has the 
          following output:
          1. The Intraday position time series (in csv lines) written to "SimTraceFile" 
             of main.cfg, with each line being a position change for a 
             strategy on a tradable.
          2. A summary of end of day position/pnl written to recovery/eod_pos.csv 
    """

    # create start.cfg
    with open(StratCFG, 'w') as scfg :
        sn = list(run_list.keys())
        if len(sn) > 0 :
            scfg.write("RunList = [ " + sn[0]);
            for s in sn[1:] :
                scfg.write(", " + s)
            scfg.write(" ]\n")
            for s in sn :
                scfg.write(s + " = [ ")
                stype, sfile = run_list[s]
                scfg.write(stype + ", " + sfile + " ]\n")

    # create main file
    upd_fields = { "SymbolMap": os.path.join(ConfigPath, SymbolMapCFG), \
                  "Strat": StratCFG,      \
                  "SimName": sim_name,    \
                  "SimDay" : trade_day,   \
                  "MtsBar" : json.dumps(mts_bar).replace('\"',''), \
                  "BarSec" : json.dumps(bar_sec) }

    with open (MainCFGTemplate, 'r') as mtpl :
        with open (MainCFG, 'w') as mcfg :
            while True :
                l = mtpl.readline()
                if len(l) == 0 :
                    break
                l = l.strip()
                if len(l) == 0 or l[0] == '#' :
                    mcfg.write(l+'\n')
                    continue
                try :
                    k, v = l.split('=')
                    k = k.strip()
                    v = v.strip()
                    if k in upd_fields.keys() :
                        mcfg.write(k + " = " + upd_fields[k] + '\n')
                    else :
                        mcfg.write(l+'\n')
                except :
                    traceback.print_exc()

    # update the symmap
    yyyymmdd = upd_fields["SimDay"]
    trd_day = yyyymmdd[:4] + "-" + yyyymmdd[4:6] + "-" + yyyymmdd[6:8]
    symbol_map_obj.update_config(trd_day, cfg_path=ConfigPath, map_file = SymbolMapCFG)

    # gunzip all the mts bar
    for (tradable, bar_file) in mts_bar :
        try :
            os.system('gunzip -f ' + bar_file)
        except :
            pass

    # kick off the simulation
    print('running ' + StratSim + ' on ' + yyyymmdd)
    ret = os.system(StratSim + ' ' + yyyymmdd)

    # gzip all the mts bar
    for (tradable, bar_file) in mts_bar :
        try :
            os.system('gzip -f ' + bar_file)
        except :
            pass

    return ret


def run_multi_day(sim_name, start_yyyymmdd, end_yyyymmdd, mts_symbol_arr, mts_repo_object, run_list_func, bar_sec, clear_bar_dir=True) :
    """
    mts_symbol_arr: array of symbols for market data
    mts_repo_object: a MTS_REPO object from mts_repo
    run_list_func: a dict of strategy_name to a list of strategy type and 
              a function to get parameter files.
              i.e. {"IDBO_TF001": ["IDBO_TF", cfg_file, param_upd_func]}
              where the param_upd_func is a function with input being 
              the date, a cfg_file as the filename, and the mts_symbol array.
              The function updates configuration file used in simulation
                  param_upd_func(day, cfg, mts_symbol_arr)
    bar_sec: a list of bar seconds on which bars will be created
    """
    print ("cleanning up simulation positions: " + EodPosCSV + " and " + PosTraceCSV)
    os.system("rm -f " + EodPosCSV + " > /dev/null 2>&1")
    os.system("rm -f " + PosTraceCSV + " > /dev/null 2>&1")

    if clear_bar_dir :
        print ("clearning up historical bar files")
        os.system("rm -f bar/*.csv > /dev/null 2>&1")

    tdi = mts_util.TradingDayIterator(start_yyyymmdd, end_yyyymmdd)
    day = tdi.begin()
    while day <= end_yyyymmdd :
        # prepare for the strategy config file
        try :
            rl = {}
            for st in run_list_func.keys() :
                sn, cfg, func = run_list_func[st]
                func(day, cfg, mts_symbol_arr)
                rl[st] = [sn, cfg]
            
            mts_bar = []

            tradable = mts_repo_object.symbol_map.get_tradable_from_mts_symbol(mts_symbol_arr, day)
            for t,m in zip(tradable, mts_symbol_arr) :
                if t is not None :
                    mts_bar.append([t, mts_repo_object.get_file_tradable(t, day)])
                else :
                    print ('no market data for ' + m + ' on ' + day)

            if len(mts_bar) == 0 :
                print('no market data is available on day ' + day + ', skipping!')
                continue

            sim_1day(sim_name, day, rl, mts_bar, bar_sec, mts_repo_object.symbol_map)
        except :
            traceback.print_exc()
            print ("problem running on " + day + ", skipping")
        day = tdi.next()

def create_mts_repo(repo_path = '~/data/mts', xml_path = 'config/symbol', max_N = 2, assets_xml = "assets.xml") :
    sym_map = symmap.SymbolMap(xml_path=xml_path, max_N=max_N, assets_xml=assets_xml)
    return mts_repo.MTS_REPO(repo_path, sym_map)


