#!/usr/bin/python3

import os
import datetime
import traceback
import glob
from multiprocessing import Process

import idbo_tf
import ar1_sim_ens
import mts_repo
import mts_mtm_pnl
import copy
import numpy as np

###
# utilities for daily operations
###
def daily_file_clean():
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
    except :
        traceback.print_exc()
        print('problem with daily file clean!')

def bar_file_crop(bar_days = 15) :
    try :
        print ('cropping bar files to hold less than ' + str(bar_days) + ' days worth of 1S bars')
        fn = glob.glob('/home/mts/run/bar/*_*S.csv')
        md = mts_repo.MTS_DATA(-6,0,17,0)
        for f in fn :
            try:
                venue, tradable, barsec = md._parse_bar_file(f)
            except:
                print('failed to parse barsec from file name %s'%(f));
                continue
            lines = int(23 * 3600 * bar_days // barsec)
            f0 = f+'.org'
            print ('cropping ' + f)
            try :
                os.system('rm -f \"' + f0 + '\" > /dev/null 2>&1')
                os.system('mv \"' + f + '\" \"' + f0 + '\" > /dev/null 2>&1')
                os.system('tail -n ' + str(lines) + ' \"' + f0 + '\" > \"' + f + '\"')
            except :
                print ('problem moving bar file ' + f + ', skipped')

        print ('Done with daily file cleaning!')
    except :
        traceback.print_exc()
        print('problem with daily file clean!')   

venue_list_17 = [\
    'CBF',\
    'CBT',\
    'CEC',\
    'CME',\
    'EOP',\
    'EUR',\
    'IFCA',\
    'IFLL',\
    'IFLX',\
    'MGE',\
    'NYM']

venue_list_18 = [
    'IFEU',\
    'IFUS']

class DailyUpdate :
    def __init__(self) :
        """
        setup an update_dict: {'update_name': {'hms': '173000', 'func': 'bar_update'}, ...}
        where func will be called with two parameters, current(future) trading_day_ymd and hms
        """
        # the idbo update
        #self.idbo = idbo_tf.IDBO_TF_Live(strat_name = 'INTRADAY_MTS_IDBO_TF_TEST')
        #self.idbo = idbo_tf.IDBO_TF_Live(strat_name = 'INTRADAY_MTS_IDBO_TF')

        #idbo_upd = {'name': 'idbo_tf'}
        #idbo_upd['hms'] = '074000'
        #idbo_upd['func'] = idbo_tf.DailyUpdate

        #idbo_upd2 = {'name': 'idbo_tf_pass_2'}
        #idbo_upd2['hms'] = '075000'
        #idbo_upd2['func'] = idbo_tf.DailyUpdate

        # the mts bar update
        # setup mts bar update for venues end at 17
        mts_bar_upd_17 = []
        for venue in venue_list_17:
            mts_data = mts_repo.MTS_DATA(-6,0,17,0)
            mts_data.venue_str = venue
            mts_bar_upd = {'name': 'mts_bar_'+venue}
            mts_bar_upd['hms'] = '170500'
            mts_bar_upd['obj'] = mts_data
            mts_bar_upd['weekdays'] = [0,1,2,3,4] # every week day, not sunday open
            mts_bar_upd['func'] = mts_data.updateLiveVenueDay
            mts_bar_upd_17.append(copy.deepcopy(mts_bar_upd))

        mts_bar_upd_18 = []
        for venue in venue_list_18:
            mts_data = mts_repo.MTS_DATA(-6,0,17,0)
            mts_data.venue_str = venue
            mts_bar_upd = {'name': 'mts_bar_'+venue}
            mts_bar_upd['hms'] = '180505'
            mts_bar_upd['obj'] = mts_data
            mts_bar_upd['weekdays'] = [0,1,2,3,4] # every week day, not sunday open
            mts_bar_upd['func'] = mts_data.updateLiveVenueDay
            mts_bar_upd_18.append(copy.deepcopy(mts_bar_upd))

        # daily mtm pnl
        mtm_pnl = {'name':'mtm_pnl'}
        mtm_pnl['hms'] = '190000'
        mtm_pnl['weekdays'] = [0,1,2,3,4] # every week day, not sunday open
        mtm_pnl['func'] = mts_mtm_pnl.gen_mtm

        # ar1 strategy
        ar1_upd = {'name': 'ar1_upd'}
        ar1_upd['hms'] = '190000'
        ar1_upd['weekdays'] = [0,1,2,3,4] # every week day, not sunday open
        ar1_upd['func'] = ar1_sim_ens.backtest_eod_ens


        # the file cleaning
        file_clean = {'name': 'file_clean'}
        file_clean['hms'] = '223000'
        file_clean['func'] = daily_file_clean

        # construct the update dictionary
        self.upd={}
        #for upd in [idbo_upd, idbo_upd2, mts_bar_upd, ar1_upd, camid2_sim, file_clean] :
        #for upd in mts_bar_upd_17 + mts_bar_upd_18 + [mtm_pnl, ar1_upd, file_clean] :
        for upd in mts_bar_upd_17 + mts_bar_upd_18 + [file_clean] :
            self.upd[upd['name']] = upd
            self.upd[upd['name']]['lastday'] = None

    def onOneSecond(self) :
        dt = datetime.datetime.now()
        hms = dt.strftime('%H%M%S')
        ymd = dt.strftime('%Y%m%d')
        wday = dt.weekday()

        for name in self.upd.keys() :
            upd = self.upd[name]
            if 'weekdays' in upd.keys() and int(wday) not in np.array(upd['weekdays']).astype(int):
                continue
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
        print('running weekend task')
        try :
            bar_file_crop()
        except:
            traceback.print_exc();
        print('weekend eod tasks done!')
