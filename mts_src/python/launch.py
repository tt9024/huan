#!/usr/bin/python3

import subprocess
import os
import datetime
import signal
import sys
import time
import symbol_map
import traceback
import daily_update
import glob
import tpmon
import numpy as np
import sys_util as su
import gc

_should_run = False
_exit_code = 0

def signal_handler(signal, frame) :
    global _should_run
    global _exit_code
    _exit_code = 2
    try :
        su.printfl( 'got signal '+ str(signal))
        if not _should_run :
            su.printfl( 'signal is being processed during shutdown...')
            return
        _should_run = False
    except:
        pass

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
procs=[\
#procs=['bin/tpmain','bin/floor', 'bin/strat_run', 'bin/ftrader ftrader1',\
#       'python/strategy/hics_calh', \
#       'python/strategy/camid/CAMID2', 'python/strategy/ar1/ar1_ens_MODEL', \
#       'python/strategy/idbo_MODEL',    'python/strategy/idbo_MODEL_tactical', \
#       'python/strategy/idbo_MODEL_HV', 'python/strategy/idbo_MODEL_HV_tactical', \
#       'python/strategy/idbo_MODEL_LV', 'python/strategy/idbo_MODEL_LV_tactical' \
       ]
#procs=['bin/tpmain','bin/floor', 'bin/strat_run']
#tp_procs=['bin/bpmain', 'bin/tpmain']
tp_procs=['bin/bpmain']
flr = 'bin/flr'
proc_map={}
RESET_WAIT_SECOND = 5

### data machine settings
BAR_PATH  = 'bar'
PYTHON_PATH = 'python'

### the object for performing daily roll/update tasts
### needs to be called approximately every couple seconds
daily_update_object = daily_update.DailyUpdate()

def is_weekend() :
    dt=datetime.datetime.now()
    wd=dt.weekday()
    if wd < 4 :
        return False
    if wd == 5 :
        return True
    if wd==4 :
        return dt.hour>22 or (dt.hour == 22 and dt.minute > 5 ) # note the weekend clean up would follow
    if wd==6 :
        return dt.hour<17 or (dt.hour == 17 and dt.minute <= 55)

def is_in_daily_trading() :
    if is_weekend() :
        return False
    dt=datetime.datetime.now()
    wd=dt.weekday()
    if wd <= 4:
        # mon-friday
        return dt.hour != 17 or (dt.minute+dt.second) == 0 or dt.minute > 55
    return True

def get_utcstart() :
    ymd = datetime.datetime.now().strftime('%Y%m%d')
    hms = "17:55:59"
    dt = datetime.datetime.strptime(ymd+ "-" + hms, '%Y%m%d-%H:%M:%S')
    return int(dt.strftime('%s'))

def get_cur_utc() :
    return int(datetime.datetime.now().strftime('%s'))

def update_config(trd_day = "") :
    """
    trd_day is in "YYYY-MM-DD", as the trading day.  
    If not given, infer from the current time, i.e. if 
    current time is later than 17:00, use tomorrow, 
    otherwise use today.
    """
    try :
        if trd_day is None or len(trd_day) == 0 :
            dtnow = datetime.datetime.now()
            utc = int(dtnow.strftime('%s'))
            if dtnow.hour >= 17 :
                # for the next trading day
                utc += 24*3600
            trd_day = datetime.datetime.fromtimestamp(utc).strftime('%Y-%m-%d')
        cfg_path = 'config'
        symbol_xml_file = os.path.join(cfg_path, 'symbol')
        main_cfg_file =   os.path.join(cfg_path, 'main.cfg')
        su.printfl ("updating symbol map for trading day " + trd_day)
        symbol_spread_dict = symbol_map.get_symbol_spread_set(main_cfg_file)
        max_N = symbol_map.get_max_N(main_cfg_file)
        t, v = symbol_map.update_symbol_map(trd_day, symbol_xml_file, max_N=max_N, symbol_spread_dict=symbol_spread_dict, add_prev_day_symbols=True)
        symbol_map.writeToConfigure(t,v,cfg_path)
    except :
        traceback.format_exc()
        raise "failed to update symbol map!"

def should_run() :
    global _should_run
    return (not is_weekend()) and _should_run

def weekly_cleanups() :
    # weekly clean up jobs to be performed here
    # this is run while system is still up
    daily_update_object.onWeekendEoD()
    pass

def eod_reconcile() :
    su.printfl ('run reconcile.  Sending requestion: ' + flr + ' E')
    ret = subprocess.check_output([flr, "E"])
    su.printfl (ret)
    su.printfl ('waiting for EoD reconcile')
    time.sleep(15)

def kill_all() :
    try :
        proc_keys = proc_map.keys()
        for p in proc_keys:
            try :
                su.kill_python_proc(proc_map[p])
            except:
                pass
        proc_map.clear()
    except:
        pass

    # second round, make sure all instances are killed
    for cmd_line in procs:
        try:
            su.kill_by_name(cmd_line)
        except:
            pass

def kill_tp() :
    try :
        proc_keys = proc_map.keys()
        for cmd_line in tp_procs:
            if cmd_line in proc_keys:
                try:
                    su.kill_python_proc(proc_map[cmd_line])
                    proc_map.pop(cmd_line)
                except:
                    pass
    except:
        pass

    # second round, make sure all instances are killed
    for cmd_line in tp_procs:
        try:
            su.kill_by_name(cmd_line)
        except:
            pass

def launch(p) :
    su.kill_by_name(p)
    su.printfl ('launching ' + p + ' ' + str(datetime.datetime.now()))
    cmd = p.strip().split(' ')
    proc_map[p]=subprocess.Popen(cmd)

def launch_sustain(argv) :
    su.printfl("launching at " + str(datetime.datetime.now()))
    alive = False
    global _should_run
    global _exit_code
    _should_run = True
    _exit_code = 0
    dtnow = datetime.datetime.now()
    kill_all()
    if not should_run() and dtnow.weekday() != 6:
        # Friday night or Saturday
        su.printfl ('Currently friday night or Saturday, wait for Sunday open...')
        while not should_run() and dtnow.weekday() != 6:
            time.sleep( 5 )
            dtnow = datetime.datetime.now()
            # in case stopped
            if _exit_code != 0:
                return

    while dtnow.weekday() == 6 and not should_run():
        # Sunday
        # make sure we start on time
        utcnow=get_cur_utc()
        utcstart=get_utcstart()
        if utcnow < utcstart -  RESET_WAIT_SECOND - 10 :
            su.printfl ('Currently Sunday, wait for Sunday open...' +  str(utcnow) + " " + str(utcstart) + " " + str(utcstart-utcnow))
        while utcnow < utcstart -  RESET_WAIT_SECOND - 10 :
            time.sleep( RESET_WAIT_SECOND )
            utcnow=get_cur_utc()
            # in case stopped
            if _exit_code != 0:
                return

        su.printfl ('getting on-line, updating roll ' + str(datetime.datetime.now()))
        utcnow=get_cur_utc()
        if utcstart > utcnow and not is_in_daily_trading():
            time.sleep(utcstart-utcnow)

        if _exit_code != 0:
            return
        utcnow=get_cur_utc()
        su.printfl ('spining for start ' +  str(utcnow))
        while not is_in_daily_trading() :
            utcnow=get_cur_utc()
            time.sleep(1)
            #time.sleep( float((1000000-utcnow.microsecond)/1000)/1000.0 )
        su.printfl ('starting on ' +  str(utcnow))

    while should_run() :  # 17:55 Sun to 22:05 Friday
        if is_in_daily_trading() : # Same range, but False at 17:00:01, back on 17:55
            if not tpmon.check_order() :  # this won't throw
                su.printfl("exit upon failed checking on order routing 35=3")
                break

            if not alive :
                su.printfl ('launch getting on-line, kill all...')
                kill_all()
                su.printfl ('launch getting on-line, update config...')
                update_config()  # this could throw
                su.printfl('bouncing tp on symbol map update!')
                kill_tp()
                tpm = tpmon.TPMonSnap()  # this could throw
                alive = True
            # poll and sustain
            for p in procs :
                if p not in proc_map.keys():
                    try: 
                        launch(p)
                        time.sleep(1)
                        continue
                    except Exception as e:
                        su.printfl('Launch ERR: failed to launch ' + p + ' ' + str(e))
                        kill_all()
                        _should_run = False
                        break
                
                # check alive
                if not su.is_python_proc_alive(proc_map[p]) :
                    su.printfl('Launch ERR: '+p+' not responsive, killing and restarting')
                    su.kill_python_proc(proc_map[p])  # this won't throw
                    proc_map.pop(p)
                    continue

                # check multiple instances
                pid_array = su.get_pid_array(p, match_exact=True)
                if len(pid_array) != 1:
                    su.printfl('Launch ERR: multiple instances detected for '+p+' pid_array: ' + str(pid_array) + ' killing all and restarting')
                    su.kill_by_name(p)
                    proc_map.pop(p)
                    continue

                # all good, do nothing

            time.sleep(1)
            # check market data, bounce if stale detected
            try :
                if not tpm.check():
                    # the market data hasn't been updated for a while, 
                    # exit the process and retry in outer (while [ 1 ]) loop
                    su.printfl ('stale detected, bouncing tp!')
                    kill_tp()
                    time.sleep(5)
                    tpm = tpmon.TPMonSnap()
            except Exception as e:
                su.printfl('problem checking tp: ' + str(e))
        
        # Not in daily trading anymore
        else:
            if alive :
                # getting offline after 5pm
                su.printfl ('launch getting off-line...')
                try:
                    eod_reconcile()
                except Exception as e:
                    su.printfl('Launch ERR: failed in eod_reconcile ' + str(e))
                for i in np.arange(10):
                    time.sleep(5)
                    su.printfl('waiting for process to finish... %d/10'%(int(i)))

                su.printfl('EoD kill all mts processes!')
                kill_all()
                alive = False
            time.sleep(1)

        # perform daily update/roll 
        daily_update_object.onOneSecond()
        sys.stdout.flush()
        sys.stderr.flush()
    
    su.printfl ('stopped ' + str(datetime.datetime.now()))
    if is_weekend() :
        # only do it on friday close
        dt=datetime.datetime.now()
        wd=dt.weekday()
        if wd == 4 :
            weekly_cleanups() # this should block until the end

    kill_all()
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    sys.path.append(PYTHON_PATH)
    try :
        launch_sustain(sys.argv)
    except KeyboardInterrupt as e:
        sys.exit(2)
    except Exception as e:
        su.printfl('Launch ERR: unhandled exception out of launch: ' + str(e))
    finally:
        kill_all()

    sys.exit(_exit_code)
