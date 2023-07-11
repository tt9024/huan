#!/usr/bin/python3

import os
import datetime
import traceback
import glob
from multiprocessing import Process

import idbo_tf
import ar1_sim
import camid2
import mts_repo

###
# utilities for daily operations
###
def daily_file_clean(bar_days = 15) :
    try :
        print ('removing recovery files')
        os.system('rm -f recovery/*replay.csv > /dev/null 2>&1')

        print ('removing unused log files')
        os.system('rm -f log/TEST_202* > /dev/null 2>&1')
        os.system('rm -f log/Engine_202* > /dev/null 2>&1')
        os.system('rm -f log/Market*Data_202* > /dev/null 2>&1')

        print ('move fix logs')
        FIXLOGPATH ='log/fix/'
        print ('deleting TT_DATA logs')
        for logfile in [ os.path.join(FIXLOGPATH, 'FIX.4.4-massar_fix_data-TT_DATA.messages.current.log') ]:
            # empty the logfile
            open(logfile, 'w').close()

        print ('gzip TT_ORDER logs')
        for logfile in [ os.path.join(FIXLOGPATH, 'FIX.4.4-massar_fix_trading-TT_ORDER.messages.current.log') ]:
            # append log to daily log file
            if os.stat(logfile).st_size == 0:
                print ('Empty log ' + logfile)
                continue
            newfile = logfile+'.'+datetime.datetime.now().strftime('%Y%m%d')
            os.system('gunzip -f ' + newfile + '.gz' + ' > /dev/null 2>&1')
            os.system('cat ' + logfile + ' >> ' + newfile)
            os.system('gzip -f ' + newfile)

            # empty the logfile
            open(logfile, 'w').close()

        print ('cropping bar files to hold less than ' + str(bar_days) + ' days worth of 1S bars')
        fn = glob.glob('/home/mts/run/bar/*_1S.csv')
        lines = 23 * 3600 * bar_days
        for f in fn :
            f0 = f+'.org'
            print ('cropping ' + f)
            try :
                os.system('mv \"' + f + '\" \"' + f0 + '\" > /dev/null 2>&1')
                os.system('tail -n ' + str(lines) + ' \"' + f0 + '\" > \"' + f + '\"')
                os.system('rm -f \"' + f0 + '\" > /dev/null 2>&1')
            except :
                print ('problem moving bar file ' + f + ', skipped')

        print ('Done with daily file cleaning!')
    except :
        traceback.print_exc()
        print('problem with daily file clean!')   

class DailyUpdate :
    def __init__(self) :
        """
        setup an update_dict: {'update_name': {'hms': '173000', 'func': 'bar_update'}, ...}
        where func will be called with two parameters, current(future) trading_day_ymd and hms
        """
        # the idbo update
        #self.idbo = idbo_tf.IDBO_TF_Live(strat_name = 'INTRADAY_MTS_IDBO_TF_TEST')
        #self.idbo = idbo_tf.IDBO_TF_Live(strat_name = 'INTRADAY_MTS_IDBO_TF')
        idbo_upd = {'name': 'idbo_tf'}
        idbo_upd['hms'] = '074000'
        idbo_upd['func'] = idbo_tf.DailyUpdate

        idbo_upd2 = {'name': 'idbo_tf_pass_2'}
        idbo_upd2['hms'] = '075000'
        idbo_upd2['func'] = idbo_tf.DailyUpdate

        # the mts bar update
        """
        self.mts_data = mts_repo.MTS_DATA(-6,0,17,0)
        mts_bar_upd = {'name': 'mts_bar'}
        mts_bar_upd['hms'] = '170500'
        mts_bar_upd['func'] = self.mts_data.updateLive
        """
        # ar1 strategy
        ar1_upd = {'name': 'ar1_upd'}
        ar1_upd['hms'] = '173000'
        ar1_upd['func'] = ar1_sim.backtest_eod

        # camid2 strategy
        camid2_sim = {'name': 'camid2_sim'}
        camid2_sim['hms'] = '170100'
        camid2_sim['func'] = camid2.run_offline

        # the file cleaning
        file_clean = {'name': 'file_clean'}
        file_clean['hms'] = '173000'
        file_clean['func'] = daily_file_clean

        # construct the update dictionary
        self.upd={}
        #for upd in [idbo_upd, idbo_upd2, mts_bar_upd, file_clean] :
        for upd in [idbo_upd, idbo_upd2, camid2_sim, ar1_upd, file_clean] :
            self.upd[upd['name']] = upd
            self.upd[upd['name']]['lastday'] = None

        # setup weekend eod task list
        # note after Friday's close, around 5:01pm,
        # the following tasks will be called immediately
        # in the order of the list.
        # Afterwards, normal onOneSecond() won't be
        # called until Sunday open.
        #self.weekend_eod_tasks = ['mts_bar', 'file_clean']
        self.weekend_eod_tasks = ['file_clean']

    def onOneSecond(self) :
        dt = datetime.datetime.now()
        hms = dt.strftime('%H%M%S')
        ymd = dt.strftime('%Y%m%d')
        for name in self.upd.keys() :
            upd = self.upd[name]
            if upd['hms'][:4] == hms[:4] and upd['lastday'] != ymd :
                try :
                    self._run(name)
                except :
                    traceback.print_exc()
                upd['lastday'] = ymd

    def _run(self, name, fork_proc=True) :
        if name not in self.upd.keys() :
            raise RuntimeError("update " + name + " not found!")
        upd = self.upd[name]
        if fork_proc :
            print ("kicking off DailyUpdate task: " + name)
            p = Process(target=upd['func'], args=())
            p.start()
        else :
            print("running task: " + name)
            target = upd['func']
            target()

    def onWeekendEoD(self) :
        dt = datetime.datetime.now()
        ymd = dt.strftime('%Y%m%d')
        for t in self.weekend_eod_tasks:
            if self.upd[t]['lastday'] != ymd:
                print('weekend eod task ' + t)
                # have to run them one-by-one
                # since we don't observe the 
                # onOneSecond anymore
                self._run(t, fork_proc=False)
                self.upd[t]['lastday'] = ymd
            else:
                print('not repeating weekend eod task ' + t)
        print('weekend eod tasks done!')

