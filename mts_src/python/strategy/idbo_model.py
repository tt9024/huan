import time
import os
import traceback
import sys
import signal
import idbo_ens
import dill
import glob
import mts_util
import numpy as np
import datetime
import time

def load_obj(f0, model_name, backtest_path, persist_path, persist_fn, strat_codes):
    obj_dict = dill.load(open(f0, 'rb'))
    return idbo_ens.IDBO_Live.retrieve(obj_dict,
                         backtest_path = backtest_path, 
                         persist_path = persist_path, 
                         persist_fn = persist_fn,
                         strat_codes = strat_codes, 
                         model_name = model_name)

def load_sod(cur_trd_day, model_name, backtest_path, persist_path, persist_fn, strat_codes):
    # find current trading day's idbo_ens_obj_yyyymmdd_upd.dill
    # if not found, find previous trading days obj and
    # run sim on mts bar from.  If found, then simply
    # gets the running object (the file without date), in which
    # stores the latest stop loss state of that trading day.

    # 1. get prev_trd_day from the persist files as the latest updated object file
    # 2. if not equal to cur_trd_day 
    #    - run offline simulation using mts bar upto cur_trd_day for the sl states,
    # Note:  param will be updated at sod_live(), which is run at object's 
    #        run_live() method.  The sod_live() is run if the object's live_dict
    #        utc0 is earlier than current day's utc0.
    # in order to run state clean from previous day, we need that object from
    # the previous day.
    #
    # Dated object files contain the object with updated
    # parameter and state from previous day, at initial starting time
    # of the trade_day.
    # Note - we could use the latest object persisted on the last day,
    # if no missing days, but we choose to run using offline mts bar, 
    # instead of relying on live update, for the case of missing in live.

    fn = os.path.join(persist_path, model_name, persist_fn +'????????')
    flist = glob.glob(fn)
    ix = np.argsort(flist)
    f0 = flist[ix[-1]]
    prev_trd_day = f0[-8:]
    assert prev_trd_day <= cur_trd_day
    if prev_trd_day == cur_trd_day:
        print('current day updated, loading running object')
        f0 = f0[:-8] # running file without date
        return load_obj(f0, model_name, backtest_path, persist_path, persist_fn, strat_codes)

    # gets the sday as latest updated object file, eday as
    # the day before cur_trd_day.  This rerun
    # a simulation on the day(s) to update the state
    # using the mts bar, as the live bar may have missing
    # due to restarts. 
    idbo_obj = load_obj(f0, model_name, backtest_path, persist_path, persist_fn, strat_codes)
    tdi = mts_util.TradingDayIterator(prev_trd_day)
    sday = tdi.begin()
    eday = sday
    days=[eday]
    while True:
        day = tdi.next()
        if day >= cur_trd_day:
            break
        eday = day
        days.append(day)

    print('current day not updated, loading object from %s, running sim from %s to %s'\
            %(f0, sday, eday))

    # try to see if the idbo object is fully update, in which case we don't need to perform simulation
    # for stop loss state
    if sday==eday:
        fn = os.path.join(persist_path, model_name, persist_fn)
        try :
            idbo_obj = load_obj(fn, model_name, backtest_path, persist_path, persist_fn, strat_codes)
            utc1 = idbo_obj.live_dict['utc1']
            utc = idbo_obj.live_dict['utc']
            if utc >= utc1-300:
                return idbo_obj
        except Exception as e:
            print('failed to get previous day runing object %s: %s\ntrying to run the simulation on %s'%(fn, str(e), sday))
            pass

    # run live trading for stop loss states for the latest obj(persisted after sod_live()on the start 
    # of that last day), this won't write backoffice nor runs sod()
    # the subsequent will be run in idbo_obj.run_live(), where sod_live 
    # would write backoffice, update the parameters and live_dict
    live_dict = idbo_obj.live_dict
    if live_dict['utc'] == live_dict['utc0']:
        print('idbo stop-loss state run from ', sday, ' to ', days[0])
        idbo_ens.sim_live(idbo_obj, sday, days[0], write_backoffice=False, run_live_sod=False, lastday_sod=False, persist_sod=False)
    elif live_dict['utc'] == live_dict['utc1']:
        print ('idbo stop-loss state skipping ', sday)
        assert sday == day[0]
    if len(days)>1:
        print('catching up from %s to %s'%(days[1], days[-1]))
        # in case we missed a day, then run_live() with live_sod, which writes backoffice, update parameters
        # run market data with stop loss state updated
        if len(days)>2:
            idbo_ens.sim_live(idbo_obj, days[1], days[-2], write_backoffice=True, run_live_sod=True, lastday_sod=False, persist_sod=False)
        idbo_ens.sim_live(idbo_obj, days[-1], days[-1], write_backoffice=True, run_live_sod=True, lastday_sod=False, persist_sod=True)

    # At this point, idbo_obj has stop loss states updated at the end of previous trading day. 
    # idbo_obj.run_live() assumes that idbo_obj has end of day state but the parameters are not updated.
    # It will run sod_live() if current trade_day's utc0 more than idbo_obj.live_dict['utc0']
    #idbo_ens.live_dict['utc']=idbo_ens.live_dict['utc1']
    return idbo_obj

