import symbol_map
import mts_util
import os
import datetime
import sys
import copy
import subprocess

def get_snap_file(tradable, venue, level='L1', provider='') :
    v = venue;
    if len(provider)>0 :
        v = provider+'_'+venue
    return '/dev/shm/'+ v + '_'+tradable+'_'+level

def get_bar_file(tradable, venue, bar_sec= 1, provider='') :
    v = venue;
    if len(provider)>0 :
        v = provider+'_'+venue
    return '/home/mts/run/bar/'+v + '_' + tradable+'_' + str(bar_sec) + 'S.csv'

CheckList = ['SPX_N1', \
             'WTI_N1', \
             'Brent_N1', \
              #Soybeans_N1', \  # Grain has a pause 8:45 to 9:30
             'HGCopper_N1', \
             'NDX_N1', \
             'DJIA_N1',\
             'EUR_N1',\
             'JPY_N1',\
             #'TY_N1', \
             #'Gilt_N1', \
             'Gold_N1']

####
# This checks some of the active symbols' last trade time
# from their 1S bar file, against a stale threshold (stale_sec)
# The check is only perfomred during the contract's trading hour AND
# regular trading hours, i.e. 18:00 to 17:00.
####
class TPMonSnap :
    def __init__(self, mts_symbol_arr=None, sym_map = None, trade_day = None, check_list = CheckList) :
        self.check_list = check_list
        self.tdu = mts_util.TradingDayUtil(shour=-6, smin=0, ehour=17, emin=0)
        self.trade_day = None
        if trade_day is None:
            trade_day = self.tdu.get_trading_day(snap_forward=True)
        self._upd_trading_day(trade_day, mts_symbol_arr = mts_symbol_arr, sym_map = sym_map)

    def _upd_trading_day(self, trade_day, mts_symbol_arr=None, sym_map = None) :
        if self.trade_day == trade_day :
            return

        print ("reseting trading day from %s to %s"%(str(self.trade_day), trade_day))
        self.trade_day = trade_day
        self.logger = mts_util.MTS_Logger('TPMonSnap')
        if sym_map is None :
            sym_map = symbol_map.SymbolMap()
        self.sym_map = sym_map

        if trade_day is None:
            trade_day = datetime.datetime.now().strftime('%Y%m%d')

        if mts_symbol_arr is None:
            mts_cfg = mts_util.MTS_Config('/home/mts/run/config/main.cfg')
            mts_symbol_arr = sym_map.getSubscriptionList(mts_cfg, trade_day)

        self.trade_day = trade_day
        self.mts_sym_map = sym_map.get_tradable_map(self.trade_day, mts_key=True, mts_symbols = mts_symbol_arr)

        # prepare for the setup
        mon_map = {}
        for mts_sym in self.mts_sym_map.keys():
            # only check mts_symbol in check_list
            if mts_sym not in self.check_list:
                continue
            tm = self.mts_sym_map[mts_sym]
            mon_map0 = {'bfname': get_bar_file( tm['tradable'], tm['venue']), \
                        'qfname': get_snap_file( tm['tradable'], tm['venue'])}
            st = tm['start_time'] # in HH:MM:SS
            et = tm['end_time']   # in HH:MM:SS
            sutc = int(datetime.datetime.strptime(self.trade_day+'-'+st, '%Y%m%d-%H:%M:%S').strftime('%s'))
            eutc = int(datetime.datetime.strptime(self.trade_day+'-'+et, '%Y%m%d-%H:%M:%S').strftime('%s'))
            if st > et:
                sutc -= 24*3600
            mon_map0['sutc'] = sutc
            mon_map0['eutc'] = eutc
            # initialize the last update to now, to allow some warmup time
            mon_map0['last_upd'] = int(datetime.datetime.now().strftime('%s'))
            mon_map[mts_sym] = mon_map0

        self.mon_map = mon_map
        self.logger.logInfo('TPMonSnap initialized with ' + str(len(self.mon_map.keys())) + ' symbols')

    def upd(self) :
        for mts_sym in self.mon_map.keys() :
            # before update, save the current last update timestamp
            # this is needed to avoid repeated failure on certain symbol
            mon_map0 = copy.deepcopy(self.mon_map[mts_sym])
            last_upd = mon_map0['last_upd']

            # get the latest bar file and and 
            self._upd_from_last_trade(mon_map0)
            if mon_map0['last_upd'] >= last_upd :
                self.mon_map[mts_sym] = mon_map0

    def _upd_from_last_trade(self, mon_map0) :
        # get the latest bar file and and 
        fname = mon_map0['bfname']
        mon_map0['vol'] = 0
        mon_map0['vbs'] = 0
        try :
            line = subprocess.check_output(['tail', '-n', '1', fname])[:-1].decode().split(',')
            # bar line in the format of
            # bar_utc, open, high, low, close, vol, last_px, last_micro, vbs
            mon_map0['close_px'] = float(line[4])
            mon_map0['last_trade_px'] = float(line[6])
            mon_map0['last_upd'] = float(line[7])/1000000.0
            mon_map0['vol'] = float(line[5])
            mon_map0['vbs'] = float(line[8])
        except :
            pass

    def _upd_from_last_quote(self, mon_map0) :
        # TODO - the simple stat on the memory mapped file
        # doesn't give the latest update time.  Need a
        # python booktap
        """
        fname = mon_map0['qfname']
        try :
            mon_map0['last_upd'] = float(os.stat(fname).st_mtime)
        except :
            pass
        """
        raise RuntimeError("Not Implemented!")

    def check(self, now = None, venue_list = None, stale_sec = 900) :
        # only check during regular trading hour
        dtnow = datetime.datetime.now()
        is_trading, tday = self.tdu.is_trading_time(dtnow)
        if not is_trading:
            return True

        # reset state in case trading day has changed
        self._upd_trading_day(tday)

        cur_utc = int(dtnow.strftime('%s'))
        self.upd()
        ok = True
        for mts_sym in self.mon_map.keys() :
            if venue_list is not None and self.mts_sym_map[mts_sym]['venue'] not in venue_list:
                continue
            # get the latest bar file
            mon_map0 = self.mon_map[mts_sym]
            if cur_utc > mon_map0['sutc'] + stale_sec  and cur_utc < mon_map0['eutc'] :
                if cur_utc - mon_map0['last_upd'] > stale_sec :
                    print ("%s is more than %d second stale!"%(mts_sym, stale_sec))
                    self.logger.logError("%s is more than %d second stale!"%(mts_sym, stale_sec))
                    ok = False
                    # break
                #else :
                #    print ("%s update %d seconds ago"%(mts_sym, cur_utc - mon_map0['last_upd']))
            #else :
                #print ("%s %d %d %d"%(mts_sym, cur_utc, mon_map0['sutc'], mon_map0['eutc']))
        return ok

def check_order(fix_order_log = '/home/mts/run/log/fix/FIX.4.4-massar_fix_trading-TT_ORDER.messages.current.log'):
    try :
        ret = subprocess.check_output(['grep', '35=3', fix_order_log])
        if len(ret) > 0 :
            print("detected low level rejects, please take a look at those message and fix!", str(ret[:500]))
            return False
    except :
        #import traceback
        #traceback.print_exc()
        pass
    return True

if __name__ == "__main__":
    tpm = TPMonSnap()
    ok = tpm.check()
    if ok :
        print ("All Good!")
        sys.exit(0)
    else :
        print ("Stale Detected!")
        sys.exit(1)
