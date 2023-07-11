import numpy as np
import datetime
import os
import traceback
import subprocess
import copy

class TradingDayIterator :
    def __init__(self, start_yyyymmdd, end_yyyymmdd=None) :
        """
        Both start and end days are inclusive. 
        """
        self.start = self._remove_date_dashs(start_yyyymmdd)
        self.end = end_yyyymmdd
        if self.end is None :
            self.end = self.start
        self.end = self._remove_date_dashs(self.end)
        self.day = None
        self.dt = None

    def begin(self) :
        self.dt = datetime.datetime.strptime(self.start, "%Y%m%d")
        if self.start <= self.end :
            self._next(1)
        else :
            self._next(-1)
        return self.day

    def end(self) :
        return self.end

    def next(self, count = 1) :
        while count > 0 :
            self.dt = datetime.datetime.strptime(self.day, "%Y%m%d") + datetime.timedelta(1)
            self._next(1)
            count -= 1
        return self.day

    def prev(self, count = 1) :
        while count > 0 :
            self.dt = datetime.datetime.strptime(self.day, "%Y%m%d") + datetime.timedelta(-1)
            self._next(-1)
            count -= 1
        return self.day

    def _next(self, delta = 1) :
        # this search through weekend and holidays
        while self.dt.weekday() > 4 or TradingDayIterator.is_holiday(self.dt) :
            self.dt += datetime.timedelta(delta)
        self.day = self.dt.strftime("%Y%m%d")

    def _remove_date_dashs(self, yyyy_mm_dd):
        if len(yyyy_mm_dd) == 10:
            ymd=yyyy_mm_dd.split('-')
            if len(ymd)==3:
                return ymd[0]+ymd[1]+ymd[2]
        if len(yyyy_mm_dd) != 8:
                raise RuntimeError('unknown yyyymmdd format %s'%(yyyy_mm_dd))
        return yyyy_mm_dd

    @staticmethod
    def is_holiday(dt) :
        """
        New year and Chrismas day, to be extended
        """
        d = dt.month * 100 + dt.day
        if d == 101 or d == 1225: # or d == 704:
            return True
        return False

    @staticmethod
    def nth_trade_day(trd_day, offset) :
        """
        for next trade day, offset=1
        for prev trade day, offset=-1
        Note 1: if trd_day is not a trading day, then
                trd_day would be adjusted to the next closest
                trading day.
        Note 2: offset is counted on trading days, not calendar days
        """
        tdi = TradingDayIterator(trd_day)
        tdi.begin()
        if offset > 0 :
            tdi.next(count = offset)
        elif offset < 0 :
            tdi.prev(count = -offset)
        return tdi.day

class TradingDayUtil :
    def __init__(self, shour=-6, smin=0, ehour=17, emin=0) :
        """
        shour: the starting hour of trading time, modulo 24 hour
        smin: the starting minute of trading time, relative to starting hour
              it could be negative
        ehour: the ending hour of trading time, ehour has to be positive
        emin: the ending minute of trading time, relative to ending hour
              it could be negative
        """
        assert (ehour > 0)
        self.tmstart = int(shour*60 + smin)
        self.tmend =   int(ehour*60 + emin)
        self.trd_len = int(self.tmend - self.tmstart)

    def get_current_bar(self, dt, barsec) :
        """
        return the current bar k, k=0,1,...,n-1
        k = 0 is the first bar, while k = n-1 is the
        last bar. 
        return -1 if not currently in trading
        """
        if not self.is_trading_time(dt)[0] :
            return -1
        tm = int((dt.hour*60+dt.minute)*60+dt.second)
        return ((tm-self.tmstart*60)%(24*60*60))//barsec

    def get_start_utc(self, trade_day_yyyymmdd) :
        utc = int(datetime.datetime.strptime(trade_day_yyyymmdd, '%Y%m%d').strftime('%s'))
        return utc + self.tmstart*60

    def get_end_utc(self, trade_day_yyyymmdd) :
        utc = int(datetime.datetime.strptime(trade_day_yyyymmdd, '%Y%m%d').strftime('%s'))
        return utc + self.tmend*60

    def is_trading_time(self, dt) :
        """
        returns (True, day_yyyymmdd) if dt is a trading time, also returns a trading day in YYYYMMDD
                Otherwise, return (False, None)
        """
        tm = dt.hour*60 + dt.minute
        day = None
        if dt.weekday() == 5 :
            # Saturday
            return False, None
        elif dt.weekday() == 4 :
            # Friday
            if tm >= self.tmend :
                return False, None
        elif dt.weekday() == 6 :
            # Sunday
            if not ((self.tmstart < 0) and (tm >= (self.tmstart % (24*60)))) :
                return False, None
        else :
            # Monday to Thursday
            if not ((tm - self.tmstart)%(24*60) < self.trd_len) :
                return False, None
 
        dt_day = dt
        # trading day tomorrow if during over night session
        if dt.hour*60 + dt.minute >= self.tmend :
            dt_day += datetime.timedelta(1)

        if TradingDayIterator.is_holiday(dt_day):
            # holiday
            return False, None

        day = dt_day.strftime('%Y%m%d')
        return True, day

    def get_trading_day(self, yyyymmdd_hh_mm_ss = None, snap_forward=True) :
        """
        yyyymmdd_hh_mm_ss : in format of yyyymmdd-hh:mm:ss, use current time if None
        snap_forward : if the given time is not a trading time, look for the next
                       trading day in future.  Other look back to the previous one
        return the current trading day.

        Note Daily trading time defined in is_trading_time()
        """

        if yyyymmdd_hh_mm_ss is None :
            yyyymmdd_hh_mm_ss = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')
        if len(yyyymmdd_hh_mm_ss)==8 and ('-' not in yyyymmdd_hh_mm_ss) and (':' not in yyyymmdd_hh_mm_ss):
            yyyymmdd_hh_mm_ss = yyyymmdd_hh_mm_ss+'-00:00:00'

        dt = datetime.datetime.strptime(yyyymmdd_hh_mm_ss, '%Y%m%d-%H:%M:%S')
        dtsign = 1 if snap_forward else -1

        is_trading, trading_day = self.is_trading_time(dt)
        while not is_trading :
            dt += datetime.timedelta(hours = dtsign)
            is_trading, trading_day = self.is_trading_time(dt)

        return trading_day

    def get_trading_day_dt(self, dt = None, snap_forward=True) :
        ts = None
        if dt is not None:
            ts = dt.strftime('%Y%m%d-%H:%M:%S')
        return self.get_trading_day(yyyymmdd_hh_mm_ss = ts, snap_forward=snap_forward)

    @staticmethod
    def dt_to_utc(dt, dt_tz, machine_tz = 'US/Eastern') :
        """
        input: 
        dt: a datetime object in time zone of dt_tz, a fraction of second maybe given
        machine_ta: the timezone that the local machine uses when
                    running the dt.strftime('%s')
        return:
        a float: utc (with fraction of second if given) of dt in dt_tz
        """
        import pytz
        dtz = pytz.timezone(dt_tz)
        ttz = pytz.timezone(machine_tz)
        ddt = dtz.localize(dt)
        tdt = ttz.localize(dt)
        df = (ddt-tdt).total_seconds()
        lutc = float(dt.strftime('%s.%f'))
        return lutc+df

    @staticmethod
    def contract_month_code(month) :
        code = [ 'F','G','H','J','K','M','N','Q','U','V','X','Z']
        m = int(month)
        if m < 1 or m > 12 :
            raise RuntimeError("month out of range " + str(month))
        return code[m-1]

class MTS_Config :
    def __init__(self, cfg_file, cfgupd = 'bin/cfgupd') :
        self.cfg_file = cfg_file
        self.cfgupd = cfgupd

    def _run(self, action, key = '', value = '') :
        return subprocess.check_output([self.cfgupd, action, self.cfg_file, key, value]).decode().strip()

    def _csvLineToList(self, csv_str, type_func=str) :
        ret = []
        if len(csv_str) > 0 :
            val = csv_str.strip().split(',')
            for v in val: 
                ret.append(type_func(v.strip()))
        return ret

    def get(self, key, type_func=str):
        """
        type_func specifies the return type, i.e. int, float, str, etc
        """
        return type_func(self._run('get', key))

    def get_safe(self, key, default_val, type_func=str) :
        try :
            val = self.get(key)
        except :
            val = default_val
        return type_func(val)

    def getArr(self, key, type_func=str) :
        val = self._run('getArr', key)
        return self._csvLineToList(val, type_func=type_func)

    def listRootKeys(self) :
        val = self._run('list')
        return self._csvLineToList(val, type_func=str)

    def listSubKeys(self, key) :
        val = self._run('list', key)
        return self._csvLineToList(val, type_func=str)

    def arrayLen(self, key) :
        return self._run('count', key)

    def set(self, key, value) :
        self._run('set', key, str(value))

    def setArr(self, key, value_list) :
        val = ''
        if len(value_list) > 0 :
            val = str(value_list[0])
            for v in value_list[1:] :
                val = val + ', ' + str(v)
        self._run('setArr', key, val)

class MTS_Logger :
    def __init__(self, module, logger_file = None) :
        if logger_file is None :
            # get it from the config/main.cfg
            cfg = MTS_Config('config/main.cfg')
            logger_path = cfg.get('Logger')
            trade_day = TradingDayUtil().get_trading_day(snap_forward=True)
            logger_file = logger_path + '_' + trade_day + '.txt'

        self.lf = logger_file
        self.module = module

    def _log(self, level, module, logstr) :
        dt = datetime.datetime.now()
        line = dt.strftime('%Y%m%d-%H:%M:%S.%f') + ','+level+','+module+','+logstr+'\n'
        with open(self.lf,'a') as f :
            f.write(line)

    def logInfo(self, logstr) :
        self._log('INF', self.module, logstr)

    def logError(self, logstr) :
        self._log('ERR', self.module, logstr)

class RiskConfig :
    def __init__(self, cfg_file = None) :
        if cfg_file is None :
            cfg_file = 'config/risk.cfg'
        self.cfg = cfg_file

    def writeDefaults(self, strat_list, strat_pos = 500.0, strat_rate_limit = [3, 60], mts_pos = 1000, mts_order = 1000):
        # get all the symbols from symbol_map
        import symbol_map
        sym_list = symbol_map.SymbolMap().list_symbol()
        with open (self.cfg, "w") as f:
            f.write("strat = {\n")
            for strat in strat_list:
                f.write("    " + strat + " = {\n")
                f.write("        PaperTrading = 1\n")
                f.write("        Symbols = {\n")
                for sym in sym_list:
                    f.write("            " + sym + " = {\n")
                    f.write("                position = " + str(strat_pos) + "\n")
                    f.write("                rate_limit = " + str(strat_rate_limit) + "\n")
                    f.write("            }\n")
                f.write("        }\n")
                f.write("    }\n")
            f.write("}\n")

            f.write("mts = {\n")
            for sym in sym_list:
                f.write("    " + sym + " = {\n")
                f.write("        position = " + str(mts_pos) + "\n")
                f.write("        order_size = " + str(mts_order) + "\n")
                f.write("    }\n")
            f.write("}\n")

class StratUtil :
    @staticmethod
    def strat_code_prefix(pod_id=None, mts_code = '7000') :
        if pod_id is not None and len(pod_id) > 0 :
            return '-'.join([pod_id, mts_code, '']) 
        return '-'.join([mts_code,''])

    @staticmethod
    def strat_code(sub_strat_code, pod_id=None, mts_code = '7000') :
        return StratUtil.strat_code_prefix(pod_id=pod_id, mts_code=mts_code)+sub_strat_code

def print_pnl(sday_yyyymmdd, eday_yyyymmdd, eod_pos_file='/home/mts/run/recovery/eod_pos.csv'):
    """
    20220126-17:00:01,TSC-7000-370,WTI_202203,27,86.60617281,66336.67,1643231461049694,USD
    """
    strat_key = {'TSC-7000-335':'IDBO', 'TSC-7000-337':'IDBO', 'TSC-7000-336':'HICS', 'TSC-7000-354':'HICS', 'TSC-7000-369':'HICS', 'TSC-7000-370':'AR1', \
            'TSD-7000-370':'AR1', 'TSC-7000-380':'IDBO'}

    eod=np.genfromtxt('recovery/eod_pos.csv',delimiter=',',dtype=str,usecols=[0,1,2,5])
    pnl_dict={}
    strat_dict={}
    for line in eod:
        d=line[0].split('-')[0]
        if not (d >= sday_yyyymmdd and d<=eday_yyyymmdd):
            continue
        mn = line[1]
        if mn[-6:]=='7000-1':
            continue
        if mn[:5]=='7000-':
            mn='TSC-'+mn
        if mn not in pnl_dict.keys():
            pnl_dict[mn]=0
        pnl=float(line[3])
        pnl_dict[mn]+=pnl
        if mn in strat_key.keys():
            sn = strat_key[mn]
            if sn not in strat_dict.keys():
                strat_dict[sn] = 0
            strat_dict[sn]+=pnl

    sum = 0
    print ('Model Pnl from %s to %s\n---'%(sday_yyyymmdd, eday_yyyymmdd))
    for k in pnl_dict.keys():
        pnl=pnl_dict[k]
        sum+=pnl
        print('%s:\t%.2f'%(k,pnl))
    print ('=== Total: %.2f'%(sum))

    sum = 0
    print ('\nStrategy Pnl from %s to %s\n---'%(sday_yyyymmdd, eday_yyyymmdd))
    for k in strat_dict.keys():
        pnl=strat_dict[k]
        sum+=pnl
        print('%s:\t%.2f'%(k,pnl))
    print ('=== Total: %.2f'%(sum))

def parseExecutionLog(fn) :
    d = np.genfromtxt(fn,delimiter=',',dtype='str')
    tpos=np.r_[0,d[:,6].astype(int)]
    tsz = np.abs(tpos[1:]-tpos[:-1])
    utc = [] 
    tdu = TradingDayUtil()

    td_prev = ''
    sz0 = 0
    tday = []
    szday = []

    for t, qty in zip(d[:,0], tsz) :
        # in format of '2021-07-22 08:05:00'
        dt0 = datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S')
        td0 = tdu.get_trading_day_dt(dt0)
        utc.append(int(dt0.strftime('%s')))
        if td_prev == '':
            td_prev=td0
        if td0 == td_prev:
            sz0+=np.abs(qty)
        else :
            tday.append(td_prev)
            td_prev = td0
            szday.append(sz0)
            sz0 = 0
    tday.append(td_prev)
    szday.append(sz0)

    return np.array(tday).astype(int), np.array(szday), np.array(utc), np.array(tsz)

def parsePnlLog(fn):
    d = np.genfromtxt(fn,delimiter=',',dtype='str')
    tday = []
    for t in d[:,0]:
        tday.append(int(t[:4]+t[5:7]+t[8:10]))
    return np.array(tday), d[:,1].astype(float)

######## Execution analysis based on the execution log  #########
class ExecutionParser :
    @staticmethod
    def parseExecution(line):
        """
        line is from C++ ExecutionReport.toString(), format
            symbol=NG_202210, algo=TSC-7000-389, clOrdId=5-0-1662552013897630632, execId=78683:26078104, tag39=4, qty=0, px=0.0,execTime=20220907-08:00:13.961, tag=, recvTime=20220907-08:00:13.972623, reserved=0
        return:
            dict of {'symbol', 'algo', 'clOrdId', 'execId', 'tag39', 'qty', 'px', 'execTime'}
        """

        f = line.split(',')
        ret = {}
        k = ['symbol', 'algo', 'clOrdId', 'execId', 'tag39', 'qty', 'px', 'execTime']
        for f0 in f:
            (k0,v0) = f0.split('=')
            k0 = k0.strip()
            v0 = v0.strip()
            if k0 in k:
                ret[k0] = v0
        return ret

    @staticmethod
    def parseExecutionFile(fn):
        """read a file with lines that embed the execution in a squre bracket, 
        i.e. [symbol=NG_202210, algo=TSC-7000-389, ... ]
        to be parsded via parseExecution()
        return a list of dict
        """
        ret = []
        with open(fn, 'rt') as fp:
            l = fp.readline()
            while len(l) > 0:
                ret0 = mts_util.parseExecution(l.split('[')[-1].split(']')[0])
                ret.append(ret0)
                l = fp.readline()
        return ret

    @staticmethod
    def getOrdDict(exec_list) :
        """exec_list: list of dict returned by parseExecution(line)
        return:
            dict of clOrdId for [symbol, algo, open_utc, open_px, open_qty, fill_avg_px (None for canceled), fill_list([ fill_t, fill_qty, fill_px ]) ]
            time_lpx: {symbol: [utc, px, qty]}
        """
        od = {}
        time_lpx = {}
        for ed in exec_list:
            oid = ed['clOrdId']
            tag = ed['tag39']
            symbol = ed['symbol']
            algo = ed['algo']
            ts = ed['execTime']
            utc = float(datetime.datetime.strptime(ts, '%Y%m%d-%H:%M:%S.%f').strftime('%s.%f'))
            px = float(ed['px'])
            qty = int(ed['qty'])

            # add time_lpx series
            if tag == '1' or tag == '2':
                if symbol not in time_lpx.keys():
                    time_lpx[symbol] = []
                time_lpx[symbol].append([utc, px, qty])

            # add od - add new or fills
            if oid not in od.keys():
                if tag != '0' :
                    continue
                od[oid] = [symbol, algo, utc, float(ed['px']), int(ed['qty']), None, []]
            else:
                if tag == '1' or tag == '2':
                    od[oid][-1].append([utc, qty, px])
                if tag == '4':
                    # cancel
                    od[oid][-1].append([utc, 0, 0])

        # update the avg_fill_px
        for oid0 in od.keys():
            order0 = od[oid0]
            fills = order0[-1]
            avg_px = 0 
            qty = 0
            for f0 in fills:
                ft,fqty,fpx = f0
                qty += abs(fqty)
                avg_px += (abs(fqty) * fpx)
            if qty > 0:
                od[oid0][-2] = avg_px/qty

        # update the time series
        for sym in time_lpx.keys():
            time_lpx[sym] = np.array(time_lpx[sym])

        return od, time_lpx

    @staticmethod
    def getPassiveOrders(ordDict):
        """ get a subset of ordDict which is canceled or filled at open px
        """
        od={}
        for oid0 in ordDict.keys():
            if ordDict[oid0][-2] is None:
                od[oid0] = copy.deepcopy(ordDict[oid0])
            elif abs(ordDict[oid0][-2] - ordDict[oid0][3]) < 1e-10:
                od[oid0] = copy.deepcopy(ordDict[oid0])
        return od

    @staticmethod
    def dumpExecList(exec_list, csv_fn=None):
        """ dump the exec_list to a csv file
        input:
        exec_list: list of dict as returned from parseExecution(line)
                   dict of {'symbol', 'algo', 'clOrdId', 'execId', 'tag39', 'qty', 'px', 'execTime'}
        output:
        csv file written in format of 
        clOrdId, open_time, open_px, open_qty, passive_or_agg, fill_or_can, fill_qty, avg_fill_px, done_time

        return:
            agg_fill_list, pass_fill_list: in format of [fill_t, fill_qty, fill_px]
            porder_filled, porder_canceled: passive orders that are filled/canceled, in format of
                                            [open_utc, open_qty, open_px]
            aorder: aggressive orders, in format of [open_utc, open_qty, open_px, fill_qty, fill_avg_px]
        """
        # od_all is a dict of clOrdId for 
        #[symbol, algo, open_utc, open_px, open_qty, fill_avg_px (None for canceled), fill_list([ fill_t, fill_qty, fill_px ])]
        od_all, time_lpx = ExecutionParser.getOrdDict(exec_list)
        od_passive = ExecutionParser.getPassiveOrders(od_all)

        afills = []
        pfills = []
        porder_filled = []  # passive order filled
        porder_canceled = []  # passive orders canceled
        aorder = []  # aggressive orders

        if csv_fn is not None:
            fp = open(csv_fn, 'wt')
            fp.write('clOrdId, open_time, open_price, open_size, passive_or_aggressive, filled_or_canceled, avg_fill_price, done_time\n')
        else:
            fp = None

        for clOrdId in od_all.keys():
            (sym, algo, open_utc, open_px, open_qty, fill_avg_px, fill_list) = od_all[clOrdId]
            if clOrdId in od_passive.keys():
                is_passive = True
                if fill_avg_px is None:
                    porder_canceled.append([open_utc, open_qty, open_px])
                else:
                    porder_filled.append([open_utc, open_qty, open_px])

            else:
                is_passive = False
            if len(fill_list) == 0:
                continue
            fillqty = 0
            for fill0 in fill_list:
                fillqty += fill0[1]
            done_utc = fill_list[-1][0]
            if is_passive:
                pfills += copy.deepcopy(fill_list)
            else:
                try :
                    fill_qty = np.sum(np.array(fill_list)[:,1])
                    if fill_qty != 0:
                        aorder.append([open_utc, open_qty, open_px, fill_qty, fill_avg_px])
                        afills += copy.deepcopy(fill_list)
                except:
                    pass

            if fp is not None:
                line = '%s,%s,%f,%d,%s,%s,%s,%s,%s\n'%(clOrdId, \
                        datetime.datetime.fromtimestamp(open_utc).strftime('%Y%m%d-%H:%M:%S.%f'),\
                        open_px,\
                        open_qty,\
                        ('passive' if is_passive else 'aggressive'),\
                        ('filled' if fill_avg_px is not None else 'canceled'),\
                        str(fillqty),\
                        ('%f'%(fill_avg_px) if fill_avg_px is not None else 'NA'),\
                        datetime.datetime.fromtimestamp(done_utc).strftime('%Y%m%d-%H:%M:%S.%f'))
                fp.write(line)

        # remove 0 fills
        ix = np.nonzero(np.array(afills)[:,1]!=0)[0]
        afills = list(np.array(afills)[ix,:])
        ix = np.nonzero(np.array(pfills)[:,1]!=0)[0]
        pfills = list(np.array(pfills)[ix,:])

        return afills, pfills, porder_filled, porder_canceled, aorder

    @staticmethod
    def mergeLpx(execLpx, utc_lpx, px_lpx):
        """merge the price time series into an "all inclusive" time series
        including all points in the execution's lpx and 1-second lpx given by
        utc_lpx and px_lpx
        """
        utc0 = execLpx[:,0]
        px0 = execLpx[:,1]
        utc = np.r_[utc0, utc_lpx]
        px = np.r_[px0, px_lpx]
        ix = np.argsort(utc)
        utc = utc[ix]
        px = px[ix]
        return np.array([utc, px]).T

    @staticmethod
    def getDecay(enter_time, enter_px, enter_qty, lpx_utc, lpx_px, decay_sec=[1,2,5,10,30]) :
        """
        plot 1 second, 5 second, 10 second and 30 second decay for
        the enter_time/px/qty
        """
        t = np.array(decay_sec) + enter_time
        ix = np.clip(np.searchsorted(lpx_utc, t), 0, len(lpx_utc)-1)
        px_diff_log = (np.log(lpx_px[ix]) - np.log(enter_px)) * np.sign(enter_qty)
        px_diff = (lpx_px[ix] - enter_px) * np.sign(enter_qty)
        return px_diff, px_diff_log

    @staticmethod
    def getDecayOrders(ordDict, lpx_utc, lpx_px, decay_sec=[1,2,5,10,30]):
        """
            ordDict has [symbol, algo, open_utc, open_px, open_qty, fill_avg_px (None for canceled), fill_list([ fill_t, fill_qty, fill_px ]) ]
        """
        px_diff = []
        for od0 in ordDict.keys():
            od = ordDict[od0]
            utc = od[2]
            px = od[3]
            qty = od[4]
            px_diff0, px_diff_log = ExecutionParser.getDecay(utc, px, qty, lpx_utc, lpx_px, decay_sec=decay_sec)
            px_diff.append(px_diff0)
        return np.array(px_diff)

    @staticmethod
    def parseAlgoSeekTrd(fn):
        """UTCDate,UTCTime,LocalDate,LocalTime,Ticker,SecurityID,TypeMask,Type,Price,Quantity,Orders,Flags
            take the LocalData + LocalTime as (CST) into utc, and take the Price and Quantity, 
            take only the "Type" equals TRADE AGRESSOR ON BUY or SELL
            Example: 20220907,000000004,20220906,190000004,NGV2,196426,98,TRADE AGRESSOR ON SELL,8.037000,2,3,0
        return:
            numpy array of [utc.mmm, qty(signed), px]
        """
        trd = np.genfromtxt(fn, delimiter=',', dtype='str', skip_header=1)
        ld = trd[:,2] # 20220907
        lt = trd[:,3] # 190000004
        tp = trd[:,7]
        px = trd[:,8].astype(float)
        qty = trd[:,9].astype(int)

        ret = []
        for ld0, lt0, tp0, px0, qty0 in zip(ld, lt, tp, px, qty):
            tp0=tp0.strip()
            if tp0 == 'TRADE AGRESSOR ON BUY': 
                sn = 1
            elif tp0 == 'TRADE AGRESSOR ON SELL':
                sn = -1
            else:
                continue

            ts = '%s-%s'%(ld0, lt0)
            utc = float(datetime.datetime.strptime(ts, '%Y%m%d-%H%M%S%f').strftime('%s.%f'))
            utc += 3600 # CST to local TZ (EST)
            ret.append( [utc, sn*qty0, px0] )
        return np.array(ret)

    @staticmethod
    def matchAlgoMts(market_fill, mts_afill, mts_pfill):
        """remove mts fill from market_fill
        input:
            market_fill: from parseAlgoSeekTrd
            mts_afill, mts_pfill: from dumpExecutionList
        return:
            market_fill without mts fills in format of [utc, qty(signed), px]
            mts_fill, all mts fills in [utc, qty(signed), px]
        Note: 
            it matches with utc in millisecond, within the -10 millisecond, 
            for the fill of the same side
        """

        pfill = np.array(copy.deepcopy(mts_pfill))
        pfill[:, 1]*=-1  # fill passive direction to match with the market direction
        mts_fill = np.vstack((np.array(mts_afill), pfill))
        ix = np.argsort(mts_fill[:,0],kind='stable')
        mts_fill = mts_fill[ix,:]
        ats = mts_fill[:,0]
        done = False
        mfill = copy.deepcopy(market_fill)
        mfill = np.array(mfill)

        for forward_sign in [1, -1]:
            ts_mil_shift=0
            while len(mts_fill) > 0 and ts_mil_shift<50:
                ix = np.clip(np.searchsorted(mfill[:,0], mts_fill[:,0]-ts_mil_shift*forward_sign),0,len(mfill)-1)
                for i, ix0 in enumerate(ix):
                    mq = mfill[ix0,1]
                    aq = mts_fill[i,1]
                    if mq*aq <=0 :
                        continue
                    fq = aq
                    if np.abs(aq) >= np.abs(mq):
                        fq = mq
                    mfill[ix0,1] -= fq
                    mts_fill[i,1] -= fq
                # remove 0 qty ix

                ix0 = np.nonzero(mts_fill[:,1]!=0)[0]
                ix1 = np.nonzero(mfill[:,1]!=0)[0]
                mts_fill = mts_fill[ix0,:]
                mfill = mfill[ix1,:]
                ts_mil_shift+=1
                print('unmatched ', len(mts_fill))

        return mfill, mts_fill


    @staticmethod
    def plotMarketMtsTrade(fig, market_trd, mts_afill, mts_pfill):
        """plot the market fills together with mts aggressive fills and passive fills
        input:
            market_trd: returned by parseAlgoSeekTrd  [utc.mmm, qty (signed), px]
            mts_afill, mts_pfill: returned by dumpExecList, [utc.mmm, qty (signed), px]
        plot:
            three panals
            1) time/fill px
            2) mts cumsum vol vs market cumsum vol
            3) mts cumsum signed vol (pass+agg+mkt(self-trade removed))
            4) market cumsum signed vol 
        return:
            axlist of 4
        """
        mts_afill = np.array(mts_afill)
        mts_pfill = np.array(mts_pfill)
        mts_all_fills = np.vstack((mts_afill, mts_pfill))
        ix = np.argsort(mts_all_fills[:,0], kind='stable')
        mts_all_fills = np.array(mts_all_fills)[ix]
        #crop by afill
        market_trd = np.array(copy.deepcopy(market_trd))
        ix0 = np.searchsorted(market_trd[:,0], mts_all_fills[0,0])
        ix1 = np.searchsorted(market_trd[:,0], mts_all_fills[-1,0])-1
        market_trd = market_trd[ix0:ix1,:]
        #match mts fills
        market_trd_other, _ = ExecutionParser.matchAlgoMts(market_trd, mts_afill, mts_pfill)

        dt_afill = []
        dt_pfill = []
        dt_allfill = []
        dt_mkt = []
        dt_mkt_other = []

        for dtf, ts in zip( [dt_afill, dt_pfill, dt_allfill, dt_mkt, dt_mkt_other], \
                [mts_afill[:,0], mts_pfill[:,0], mts_all_fills[:,0], market_trd[:,0], market_trd_other[:,0]]):
            for t0 in ts:
                dtf.append(datetime.datetime.fromtimestamp(t0))
        dt_afill=np.array(dt_afill)
        dt_pfill=np.array(dt_pfill)
        dt_allfill=np.array(dt_allfill)
        dt_mkt=np.array(dt_mkt)
        dt_mkt_other=np.array(dt_mkt_other)

        ax1=fig.add_subplot(4,1,1)
        ax1.plot(dt_mkt, market_trd[:,2], label='px')

        ax2=fig.add_subplot(4,1,2,sharex=ax1)
        ax2.plot(dt_mkt, np.cumsum(np.abs(market_trd[:,1])), label='market vol')
        ax2.plot(dt_mkt_other, np.cumsum(np.abs(market_trd_other[:,1])), label='market vol (no mts)')
        ax2.plot(dt_allfill, np.cumsum(np.abs(mts_all_fills[:,1])), label='mts vol')
        ax2.plot(dt_mkt_other, np.cumsum(np.abs(market_trd_other[:,1])), label='market vol (no mts)')

        ax3=fig.add_subplot(4,1,3,sharex=ax1)
        ax3.plot(dt_afill, np.cumsum(mts_afill[:,1]), '.-', label='mts aggressive trade fills')
        ax3.plot(dt_pfill, np.cumsum(mts_pfill[:,1]), '.-', label='mts passive trade fills')
        ax3.plot(dt_mkt_other, np.cumsum(market_trd_other[:,1]), '.-', label='market fills(no mts)')

        ax4=fig.add_subplot(4,1,4,sharex=ax1)
        ax4.plot(dt_mkt, np.cumsum(market_trd[:,1]), '.-', label='market fills')
        ax4.plot(dt_mkt_other, np.cumsum(market_trd_other[:,1]), '.-', label='market fills(no mts)')
        ax4.plot(dt_allfill, np.cumsum(mts_all_fills[:,1]), '.-',label='mts fills')

        axlist=[ax1,ax2,ax3,ax4]
        for ax in axlist:
            ax.grid()
            ax.legend(loc='best')
        return axlist, mts_all_fills

    @staticmethod
    def getPassiveEnterDecay(porder_filled, porder_canceled, mfill, lpx_utc, lpx_px, decay_sec=[1,2,5,10,30,60], fig=None):
        """plot the price impact of MTS filled working order, and canceled working order, measured from the open time
           and the price impact of market fills (other than MTS fills), measured from the fill time
        input:
            porder_filled, porder_canceled: returned from dumpExecutionList
            mfill: returned from matchOrders
            lpx_utc, lpx_px: could use the all_trd returned from algoSeek
        return
            imp_mts, imp_mkt, imp_mts_filled, imp_mts_canceled: weighted average price moved of 
                    all mts, all market other than mts, mts passive filled, all passive canceled
        """

        #crop the mfill w.r.t porder
        ix0 = np.searchsorted(mfill[:,0], min(porder_filled[0][0], porder_canceled[0][0]))
        ix1 = np.searchsorted(mfill[:,0], max(porder_filled[-1][0], porder_canceled[-1][0]))-1
        print(ix0, ix1)
        mfill0 = mfill
        #mfill0 = mfill[ix0:ix1,:]

        pfilled_px, pfilled_lr, pcanceled_px, pcanceled_lr, m_px, m_lr = ([], [], [], [], [], [])
        for et, impx, imlr in zip( [porder_filled, porder_canceled, mfill0], \
                               [pfilled_px, pcanceled_px, m_px],\
                               [pfilled_lr, pcanceled_lr, m_lr] ):
            for utc, qty, px in et:
                px_diff0, px_diff_log = ExecutionParser.getDecay(utc, px, qty, lpx_utc, lpx_px, decay_sec=decay_sec)
                impx.append(list(px_diff0) +[qty])
                imlr.append(list(px_diff_log) +[qty])

        # last column is a qty - for weighted avg purpose
        pfilled_px = np.array(pfilled_px)
        pfilled_lr = np.array(pfilled_lr)
        pcanceled_px = np.array(pcanceled_px)
        pcanceled_lr = np.array(pcanceled_lr)
        m_px = np.array(m_px)
        m_lr = np.array(m_lr)

        imp = []
        p_px0 = np.vstack((pfilled_px, pcanceled_px))
        m_px0 = m_px[7919:13379,:]
        for px in p_px0,m_px0,pfilled_px,pcanceled_px:
            px0 = px[:,:-1]
            qty0 = np.abs(px[:,-1])
            px0 = np.sum((px0.T*qty0).T,axis=0)/np.sum(qty0)
            imp.append(copy.deepcopy(px0))

        imp_mts, imp_mkt, imp_mts_filled, imp_mts_canceled = imp
        if fig is not None:
            ax=fig.add_subplot(1,1,1)
            ax.plot(decay_sec, imp_mts, label='MTS Working Orders')
            ax.plot(decay_sec, imp_mkt, label='Fills other than MTS')
            ax.plot(decay_sec, imp_mts_filled, '-.', label='MTS Working Orders Filled')
            ax.plot(decay_sec, imp_mts_canceled, '-.', label='MTS Working Orders Canceled')
            ax.grid() ; ax.legend(loc='best')
            ax.set_title('price movement w.r.t. MTS working order entrance, 9/7 from 08:00 to 08:15')
            ax.set_xlabel('time in seconds')
            ax.set_ylabel('price movement')

        return imp_mts, imp_mkt, imp_mts_filled, imp_mts_canceled

    @staticmethod
    def parseAlgoL2(fn, px_mul=10):
        l2 = np.genfromtxt(fn, delimiter=',', dtype='str', skip_header=1)
        # remove the implied quotes
        ix = np.nonzero(l2[:,7] == '0')[0]
        l2 = l2[ix]
        # getting the utc out from CST
        dt = l2[:,2]
        tm = l2[:,3]
        l2t = []
        for dt0, tm0 in zip(dt, tm):
            utc = datetime.datetime.strptime(dt0+tm0, '%Y%m%d%H%M%S%f').strftime('%s.%f')
            l2t.append(float(utc) + 3600)
        # getting the bid/ask index
        bid_ask = l2[:,6]
        bix = np.nonzero(bid_ask=='B')[0]
        aix = np.nonzero(bid_ask=='S')[0]
        px_ix = np.arange(10).astype(int)*3+9
        qty_ix = np.arange(10).astype(int)*3+10
        return {'utc':np.array(l2t), 'px':  l2[:,px_ix].astype(float)/px_mul, 'qty': l2[:,qty_ix].astype(int), 'bix':bix, 'aix':aix}

    @staticmethod
    def getL2Vwap(l2_dict, utc, qty):
        if qty>0 :
            ix=l2_dict['aix']
        else:
            ix = l2_dict['bix']

        utc_l2 = l2_dict['utc'][ix]
        tix = np.searchsorted(utc_l2, utc)

        v = l2_dict['qty'][ix][tix]
        p = l2_dict['px'][ix][tix]
        ix = np.nonzero(np.abs( v*p ) > 1e-10 )[0]
        v = np.cumsum(v[ix])
        p = p[ix]
        vix = np.clip(np.searchsorted(v, abs(qty))+1, 0, len(v)-1)
        return np.sum(v[:vix]*p[:vix])/np.sum(v[:vix])

    @staticmethod
    def getFrontRunPayUps(exec_list, \
                algo_seek_trd_fn = "/mnt/mts/outage_logs/decay_analysis/data/algoseek/NGV2_trd.csv", \
                algo_seek_l2_fn  = "/mnt/mts/outage_logs/decay_analysis/data/algoseek/NGV2_L2_0800_0815.csv", \
                l2_dict = None, \
                out_csv_pass_fn = None, 
                out_csv_agg_fn = None):
        """
        input:
            exec_list: list of mts orders, such as returned by parseExecutionFile(fn), or loaded from dill
        output:
            pass_pu, agg_pu: list of [enter_utc(float), enter_qx(*), enter_qty, payup_ticks, payup_value, front_run_ords, front_run_qty(%)
        Note for agg_pu, the analysis is on two consecutive aggressive orders, and the L2-vwap price of sum of two qty
        """

        #porder_xxx in [open_utc, open_qty, open_px]
        #aorder: aggressive orders, [open_utc, open_qty, open_px, fill_qty, fill_avg_px]
        afills, pfills, porder_filled, porder_canceled, aorder = ExecutionParser.dumpExecList(exec_list)
        all_market_fills = ExecutionParser.parseAlgoSeekTrd(algo_seek_trd_fn)
        other_fills, mts_fills = ExecutionParser.matchAlgoMts(all_market_fills, afills, pfills)
        if l2_dict is None:
            l2_dict = ExecutionParser.parseAlgoL2(algo_seek_l2_fn)

        # get the passive orders table
        # for each or order, find the next aorder time, and gets the avg fill px and the other_fills in between

        pc = np.array(porder_canceled)
        utc_pc = pc[:,0]
        qty_pc = pc[:,1]
        px_pc = pc[:,2]

        # payup time and px
        ao = np.array(aorder)
        utc_ao = ao[:,0]

        pcix = np.clip(np.searchsorted(utc_ao, utc_pc+0.001),0,len(utc_ao)-1)
        utc_agg = utc_ao[pcix]
        px_ao = ao[pcix,2]
        tick_size = 0.001
        px_diff = np.abs(px_pc - px_ao)

        payup_ticks = np.round(px_diff/tick_size)
        payup_val = payup_ticks*10*np.abs(qty_pc)

        # find other_fills between utc_pc to utc_agg
        # figure out the buy fills and sell fills
        of = np.array(other_fills)
        qty_of = of[:,1]
        #bix_of = np.nonzero(qty_of>0)[0]
        #six_of = np.nonzero(qty_of<0)[0]
        utc_of = of[:,0]
        ix0 = np.clip(np.searchsorted(utc_of, utc_pc),0,len(utc_of)-1)
        ix1 = np.clip(np.searchsorted(utc_of, utc_agg),0,len(utc_of)-1)

        # from utc_pc to utc_agg on qty_pc
        front_qty = []
        for qty_pc_, ix0_, ix1_ in zip(qty_pc, ix0, ix1):
            ix1_ = min(ix1_+1, len(qty_of)-1)
            qix = np.nonzero(qty_of[ix0_:ix1_]*qty_pc_>0)[0]
            if len(qix) > 0:
                fr_qty = np.sum(np.abs(qty_of[ix0_:ix1_][qix]))
                front_qty.append([fr_qty, fr_qty/np.abs(qty_pc_)])
            else:
                front_qty.append([0, 0])
        front_qty = np.array(front_qty)
        pass_pu = np.vstack(( utc_pc, qty_pc, px_pc, payup_ticks, payup_val, front_qty.T)).T

        ############
        # get the consecutive aggressive orders payup
        ############
        oa_agg_2 = []  #at0, at1, total_sz, afx, payup_ticks, payup_value
        time_diff = 10
        i=0
        while i < len(ao)-1:
            at0, qty0, px0, fqty0, fpx0 = ao[i][:5]
            i+=1
            at1, qty1, px1, fqty1, fpx1 = ao[i][:5]
            if fqty0*fqty1 <= 0 :
                continue
            if at1-at0 > time_diff:
                continue
            while at1 - at0 < time_diff and i < len(ao)-1:
                i+=1
                at1, qty_, px_, fqty_, fpx_ = ao[i][:5]
                if fqty_ * fqty1 <= 0:
                    continue
                fpx1 = (fpx1*fqty1 + fpx_*fqty_)/(fqty1+fqty_)
                fqty1 += fqty_

            fqty = fqty0+fqty1
            fpx = (fqty0*fpx0+fqty1*fpx1)/fqty

            l2_vpx = ExecutionParser.getL2Vwap(l2_dict, at0, fqty)
            payup_ticks = np.round(max((fpx-l2_vpx)* np.sign(fqty),0)/tick_size)
            if payup_ticks <= 1e-10:
                continue
            payup_val = payup_ticks*10*abs(fqty)
            oa_agg_2.append([at0, at1, fqty0, fqty1, l2_vpx, payup_ticks, payup_val])

        # get the front run qty
        oa_agg_2 = np.array(oa_agg_2)
        at0 = oa_agg_2[:,0]
        at1 = oa_agg_2[:,1]
        ix0 = np.clip(np.searchsorted(utc_of, at0),0,len(utc_of)-1)
        ix1 = np.clip(np.searchsorted(utc_of, at1),0,len(utc_of)-1)
        qty_oag = oa_agg_2[:,2]+oa_agg_2[:,3]
        front_qty_oag = []
        for qty_, ix0_, ix1_ in zip(qty_oag, ix0, ix1):
            ix1_ = min(ix1_+1, len(qty_of)-1)
            qix = np.nonzero(qty_of[ix0_:ix1_]*qty_>0)[0]
            if len(qix) > 0:
                fr_qty = np.sum(np.abs(qty_of[ix0_:ix1_][qix]))
                front_qty_oag.append([fr_qty, fr_qty/np.abs(qty_)])
            else:
                front_qty_oag.append([0, 0])
        front_qty_oag = np.array(front_qty_oag)

        # utc_pc, qty_pc, px_pc, payup_ticks, payup_val, front_qty, front_pct
        agg_pu = np.vstack((oa_agg_2[:,0], oa_agg_2[:,2]+oa_agg_2[:,3],oa_agg_2[:,4:].T,front_qty_oag.T )).T

        for csv_fn, pu in zip( [out_csv_pass_fn, out_csv_agg_fn], [pass_pu, agg_pu] ):
            if csv_fn is not None:
                dts = []
                for t0 in pu[:,0]:
                    dts.append(datetime.datetime.fromtimestamp(t0).strftime('%Y%m%d-%H:%M:%S.%f'))
                dts=np.array(dts)
                ppu = np.vstack(( dts, pu[:,1:].T )).T
                np.savetxt(csv_fn, ppu, fmt='%s',delimiter=',',header='enter_time, enter_qty, limit_px, payup_tics, payup_value, front_run_qty, front_run_qty_pct')

        return pass_pu, agg_pu

    @staticmethod
    def mergeMtsTT_Fills(exec_list, tt_fills, out_fn = '/tmp/mf.csv', start_pos=0):
        """tt_fills loaded from decay_analysis/data/tt_fills.dill, format of [utc.milli, qty, px]
        return:
            all_fills, merged from the two, with format [utc.milli, exec_id, qty, px]
            The goal is to show the mts position, flr tgt_pos, flr twap_remain
        """
        od, time_lpx = ExecutionParser.getOrdDict(exec_list)
        od_passive = ExecutionParser.getPassiveOrders(od)
        mts_fills = [] #[utc.milli, execid, qty, px, pass/agg (0/1)]
        for e in exec_list:
            if e['tag39'] == '1' or e['tag39'] == '2':
                utc0 = float( datetime.datetime.strptime(e['execTime'], '%Y%m%d-%H:%M:%S.%f').strftime('%s.%f'))
                pa = 'agg'  #agg
                if e['clOrdId'] in od_passive.keys():
                    pa = 'pass'  #pass
                mts_fills.append( [ utc0, e['execId'], e['qty'], e['px'], pa] )
        mts_fills = np.array(mts_fills)
        utc_mts = mts_fills[:,0].astype(float)

        # mark it to tt fills
        ix0 = np.searchsorted(tt_fills[:,0], utc_mts[0])
        tt_fills = tt_fills[ix0:,:]
        #ix1 = np.searchsorted(tt_fills[:,0], utc_mts[-1])
        #tt_fills = tt_fills[:ix1,:]
        ix = np.searchsorted(tt_fills[:,0], utc_mts)

        # go through this ix
        ix_merge = []
        i0 = ix[0]
        for ix0 in ix:
            i0 = max(i0+1, ix0)
            ix_merge.append(i0)
        ix_merge = np.array(ix_merge).astype(int)
        n,_=tt_fills.shape
        mf = np.zeros((n, 5)).astype('str')
        mf[:,0] = tt_fills[:,0].astype('str')
        mf[:,2] = tt_fills[:,1].astype('str')  #qty
        mf[:,3] = tt_fills[:,2].astype('str')  #px
        mf[ix_merge,:] = mts_fills.copy()

        mf=mf[1:,:]

        qty = mf[:,2].astype(float)
        cum_qty = np.cumsum(qty)
        ix = np.nonzero(mf[:,1] == '0.0')[0]
        qty0 = qty.copy()
        qty0[ix]=0.0
        cum_qty_mts = np.cumsum(qty0)

        cum_qty += start_pos
        cum_qty_mts += start_pos

        # write to a csv for merged
        if out_fn is not None:
            dt=[]
            for t in mf[:,0]:
                dt.append(  datetime.datetime.fromtimestamp( float(t) ).strftime('%Y%m%d-%H:%M:%S.%f'))
            dt=np.array(dt)
            np.savetxt(out_fn, np.vstack((dt, mf[:,1:].T, cum_qty, cum_qty_mts)).T, fmt='%s', delimiter=',', header='fill_time, fill_exec_id, fill_qty, fill_px, fill_flag(0:pass-1:agg),tt_position, oms_position')
        return mf, mts_fills, cum_qty, cum_qty_mts

