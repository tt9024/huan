import numpy as np
import datetime
import os
import sys
import traceback
import subprocess
import glob
import pandas
import glob
import copy

import mts_util
import tickdata_parser as td_parser
import repo_util
import symbol_map

class MTS_REPO(object) :
    """
    Manages the MTS Bar in csv files for securities and dates
    Each file is one day of 1 second bar for the security in directories:
    repo_path/YYYYMMDD/tradable.csv
    """
    def __init__(self, repo_path, symbol_map_obj=None, backup_repo_path=None) :
        """
        repo_path: the root path to the MTS repo
        symbol_map_obj: the SymbolMap object for future contract definitions
        """
        self.path = repo_path
        self.backup_path = backup_repo_path
        if symbol_map_obj is None :
            symbol_map_obj = symbol_map.SymbolMap()
        self.symbol_map = symbol_map_obj


    def get_file_symbol(self, mts_symbol, contract_month, day, create_path=False, repo_path = None) :
        """
        getting file via the mts_symbol, i.e. "WTI" and a contract_month, i.e. '202101'
        """
        if repo_path is None: 
            repo_path = self.path
        p = os.path.join(repo_path, day)
        if create_path :
            os.system('mkdir -p ' + p + ' > /dev/null 2>&1')
        return os.path.join(p, mts_symbol+'_'+contract_month+'.csv')

    def get_file_tradable(self, tradable, yyyymmdd, create_path=False, repo_path = None, is_mts_symbol=None, get_holiday=True, check_file_exist=False):
        """
        get a mts bar file name from a 'tradable', could be a mts_symbol or a tradable
        """
        symn, sym, cont = (None,None,None)
        if is_mts_symbol is None:
            try:
                is_mts_symbol = tradable.split('_')[1][0]=='N'
            except:
                is_mts_symbol = False
        if is_mts_symbol is not None:
            try:
                symn, sym, cont = self.symbol_map.get_symbol_contract_from_tradable(tradable, yyyymmdd, is_mts_symbol=is_mts_symbol, add_prev_day=get_holiday)
            except:
                raise KeyError('failed to find bar file for %s on %s'%(tradable, yyyymmdd))
        else :
            try :
                symn, sym, cont = self.symbol_map.get_symbol_contract_from_tradable(tradable, yyyymmdd, is_mts_symbol=True, add_prev_day=get_holiday)
            except :
                symn, sym, cont = self.symbol_map.get_symbol_contract_from_tradable(tradable, yyyymmdd, is_mts_symbol=False, add_prev_day=get_holiday)

        bar_file = self.get_file_symbol(sym, cont, yyyymmdd, create_path=create_path, repo_path=repo_path)
        if check_file_exist:
            # bar_file could have '.gz'
            try :
                fn = glob.glob(bar_file+'*')
                assert os.stat(fn[0]).st_size > 1024
            except:
                try :
                    bar_file = self.get_backup_bar_file(bar_file)
                    fn = glob.glob(bar_file+'*')
                    assert os.stat(fn[0]).st_size > 1024
                except :
                    print( 'no bar file found (or too small) on '+yyyymmdd+ ' for ' + bar_file)
                    raise RuntimeError('no bar files found on '+yyyymmdd+ ' for ' + bar_file)
        return bar_file, sym, cont

    def get_backup_bar_file(self, bar_file) :
        yyyymmdd, sym_cont = bar_file.split('/')[-2:]
        sym, cont = sym_cont.split('.')[0].split('_')
        
        return self.get_file_symbol(sym, cont, yyyymmdd,create_path=False,repo_path=self.backup_path)

    def _get_bar(self, mts_bar_file, barsec = 1, prev_close_px=None, ref_utc=None, allow_bfill=False):
        """
        Read given mts bar file, and return generated bar using self.barsec
        prev_close_px: the last 1-sec bar of the previous day. 
                       if is given, then it is set as the open of the first bar
                       This is to get the over-night return.
        ref_utc: [sutc, eutc], one-sec bar close time as np.arange(sutc,eutc)+1
                 if None, no normalization, i.e. no gap/invalid 
        """
        print ('getting mts bar from %s'%mts_bar_file)

        # bar_file is without '.gz', genfromtxt takes care of gzip
        bar = np.genfromtxt(mts_bar_file, delimiter=',', dtype='float')

        # fix the first bars w.r.t OHLC prices, last price could be 0
        # 1. if prev_close given, set it as first open
        # 2. forward fill initial 0 bars, this could happen for tickdata repo
        utc_ix,open_ix,high_ix,low_ix,close_ix,lpx_ix,ltm_ix = repo_util.get_col_ix(['utc','open','high','low','close','lpx','ltm'])
        ixnz = np.nonzero(bar[:,close_ix]!=0)[0]
        if len(ixnz) == 0:
            print('all zero bars from ', mts_bar_file)
            raise RuntimeError('all zero bars from ' + mts_bar_file)

        nbars,ncols=bar.shape
        if ref_utc is not None:
            sutc,eutc = ref_utc
        else:
            # assuming the 1S bars from the repo bar file
            sutc = (bar[0,utc_ix] - 1)//300*300  #normalize to 5min, in case the first bars missing
            eutc = bar[-1,utc_ix]

        if prev_close_px is not None:
            assert prev_close_px[utc_ix] < bar[0, utc_ix], 'utc of prev_close more than first open %d:%d'%(prev_close_px[utc_ix],bar[0, utc_ix])

            # set the first open px as the previous day's last bar's close
            px = prev_close_px[close_ix]
            bar[0,open_ix]=px
            bar[0,high_ix]=max(bar[0,high_ix],px)
            bar[0,low_ix] =min(bar[0,low_ix], px)
            # if the first lpx/ltm is 0, set them as well
            if bar[0,ltm_ix] == 0:
                bar[0,ltm_ix] = prev_close_px[ltm_ix]
            if bar[0,lpx_ix] == 0:
                bar[0,lpx_ix] = prev_close_px[lpx_ix]

            # insert the previous bar to the begining
            pp = np.r_[prev_close_px, np.zeros(ncols)][:ncols].reshape((1,ncols))
            bar = np.vstack((pp,bar))
        bar = repo_util.daily1s(bar, sutc, eutc, backward_fill=allow_bfill)

        # all bars are valid and properly filled within ref_utc
        # now merge into the barsec
        if barsec != 1 :
            print ('merge into %d second bars'%barsec)
            bar = repo_util.mergeBar(bar, barsec)
        return bar

    def get_bar(self, mts_bar_file, barsec = 1, prev_close_px=None, ref_utc=None, allow_bfill=False):
        """
        a wrapper for get_bars, to entertain a 'backup' repo as a secondary source
        """
        bar = np.array([])
        try :
            bar=self._get_bar(mts_bar_file, barsec=barsec, prev_close_px=prev_close_px, ref_utc=ref_utc, allow_bfill=allow_bfill)
        except Exception as e:
            print(e)
        if len(bar) == 0 and self.backup_path is not None :
            mts_bar_file_backup = self.get_backup_bar_file(mts_bar_file)
            print ('problem getting ', mts_bar_file, ', trying backup ', mts_bar_file_backup)
            bar=self._get_bar(mts_bar_file_backup, barsec=barsec, prev_close_px=prev_close_px, ref_utc=ref_utc, allow_bfill=allow_bfill)
        if len(bar) == 0:
            raise RuntimeError('no bars found from %s'%(mts_bar_file))
        return bar

    def get_bars(self, tradable, sday, eday, barsec=1, cols = None, out_csv_file = None, is_mts_symbol=None, get_roll_adj=False, hours=(-6,0,17,0), get_holiday=False, allow_bfill=False):
        """ 
        tradable could be mts_symbol or a tradable
        both sday, eday are inclusive, in yyyymmdd
        cols an array for columns defined in td_parser.MTS_BAR_COL
        if get_roll_adj is True, return a second value as a dict of ['days','contracts','roll_adj']
        hours: tuple of 4 integer (start_hour, min, end_hour_min), default to be (-6, 0, 17, 0)
        get_holiday: holidays could have half day of data saved in repo. If set to True, try read if 
                     exists, otherwise, will skip
        """
        if cols is not None:
            cols_ix = repo_util.get_col_ix(cols)

        prev_bar_file = None
        prev_contract = None
        prev_close_px = None
        prev_close_px_adj = None # for keep track of roll adjust
        prev_day = None
        if get_roll_adj:
            days = []
            contracts = []
            roll_adj = []

        tdi = mts_util.TradingDayIterator(sday, eday)
        tdi.begin()
        day = tdi.prev()
        prev_cnt = 0
        while True :
            try :
                # always include half day if previous day is a holiday
                # this would use the holiday's previous day as the definition on that day
                prev_bar_file, prev_sym, prev_contract = self.get_file_tradable(tradable, day, is_mts_symbol=is_mts_symbol, get_holiday=True, check_file_exist=True)
                bar0 = self.get_bar(prev_bar_file, barsec=1, ref_utc=None, allow_bfill=True)
                prev_close_px = bar0[-1,:].copy()
                prev_day = day
                break
            except KeyboardInterrupt as e:
                print('stopped')
                return None
            except :
                print ('problem getting prev day ', day, ' bar, try previous one')
                day = tdi.prev()
                prev_cnt += 1
                if prev_cnt > 3 :
                    print ("problem getting previous day for " + sday)
                    break

        tdi = mts_util.TradingDayIterator(sday, eday)
        day = tdi.begin()
        bar = []
        colc = repo_util.get_col_ix(['close'])[0]

        md = MTS_DATA(*hours, barsec=barsec)
        while day <= eday :
            try :
                # in case we rolled today
                bar_file, sym, contract = self.get_file_tradable(tradable, day, is_mts_symbol=is_mts_symbol, get_holiday=get_holiday, check_file_exist=True)
                if prev_contract is not None and contract != prev_contract:
                    # get from that contract
                    try :
                        prev_bar_file = self.get_file_symbol(sym, contract, prev_day)
                        bar0 = self.get_bar(prev_bar_file, barsec=1, ref_utc=None, allow_bfill=True)
                        prev_close_px = bar0[-1,:].copy()
                    except :
                        print('failed to get current contract %s from previous day %s, overnight return might be lost'%(contract, prev_day))
                        prev_close_px = None

                prev_day = day
                prev_contract = contract
                if get_roll_adj:
                    # at this point, 
                    # prev_close_px_adj is previous day's close price using previous day's contract
                    # prev_close_px is the previous day's close price using today's contract
                    days.append(day)
                    contracts.append(contract)
                    if prev_close_px_adj is None:
                        #print('missed the roll adjust on day ' + day)
                        cur_adj = 0
                    else :
                        # normalize to ticksize
                        if prev_close_px is None:
                            cur_adj=0
                        else:
                            cur_adj = prev_close_px[colc]-prev_close_px_adj[colc]
                            if np.abs(cur_adj) >= 1e-10:
                                try :
                                    tick_size = float(self.symbol_map.get_tinfo(tradable, day,is_mts_symbol=True)['tick_size'])
                                except Exception as e:
                                    print('failed to get tick size ' + str(e))
                                cur_adj = np.round(cur_adj/tick_size)*tick_size
                                print ('got roll adjust %f, ticksize %f'%(cur_adj, tick_size))
                            else:
                                cur_adj = 0
                    roll_adj.append(cur_adj)

                # warn in case the prev_close_px is None, i.e. on the roll day of N3, (max_N=3)
                if prev_close_px is None and not allow_bfill:
                    print('got prev_close_px None on %s, roll of max_N contract(prev:%s-curr:%s)? consider setting allow_bfill on the day'%(day, str(prev_contract), str(contract))) 

                ref_utc = md._get_ref_utc(day)  #[sutc,eutc] from 'hours' given
                bar0 = self.get_bar(bar_file, barsec=barsec, prev_close_px = prev_close_px, ref_utc=ref_utc, allow_bfill=allow_bfill)
                prev_close_px = bar0[-1,:].copy()
                prev_close_px_adj = prev_close_px.copy()
                if cols is not None:
                    bar0 = bar0[:, cols_ix]
                if len(bar) == 0 or bar[-1].shape[1] == bar0.shape[1]:
                    bar.append(bar0)
                else :
                    raise RuntimeError('got bar different column size on %s, new shape %s, previous shape %s'%(day, str(bar0.shape), str(bar[-1].shape)))
            except KeyboardInterrupt as e:
                print ('stopped...')
                return None
            except KeyError as e:
                print('problem finding contract on %s for %s, a holiday?'%(day, tradable))
            except :
                traceback.print_exc()
                print ('problem getting day ', day, ' skipping...')
            day = tdi.next()

        if len(bar) == 0:
            raise RuntimeError('no bar found from %s to %s for %s'%(sday, eday, tradable))

        # finally, check 'hours' enforced across all days
        # TODO - remove this check as daily1s() already enforce it via ref_utc
        bar = repo_util.crop_bars(bar,cols,barsec,fail_check=True) #check 'hours' enforced for all days

        if out_csv_file is not None:
            print ('writting to %s'%out_csv_file)
            repo_util.saveCSV(bar, out_csv_file)

        if not get_roll_adj:
            return bar

        roll_adj_dict = {'days':days, 'contracts':contracts, 'roll_adj':roll_adj}
        return bar, roll_adj_dict

    @staticmethod
    def roll_adj(bar, utc_col, adj_cols, roll_adj_dict):
        """
        inputs;
            bar: shape [ndays, n, nc]
            utc_col: the column index (into nc) for utc, i.e. 0
            adj_cols: index into nc, i.e. [1,2,3,4,6] for o/h/l/c/lpx
            roll_adj_dict: dict of keys('days','contracts', 'roll_adj'), returned by get_bars()
                     days in 'YYYYMMDD', roll_adj is px diff from front to previous contract
        return: 
            bar with price adjusted with the most front contract unchanged,
                second most front adjusted from the front and the third front adjusted 
                from both second and first, etc.
            note bar is adjusted inplace
        """
        dix = 0
        bix = 0
        days = roll_adj_dict['days']
        adj = roll_adj_dict['roll_adj']
        ndays, n, nc = bar.shape
        bar_days = []
        for t in bar[:,-1,utc_col]:
            bar_days.append(datetime.datetime.fromtimestamp(t).strftime('%Y%m%d'))
        bar_days=np.array(bar_days)
        cols = np.array(adj_cols).astype(int)
        for day, diff in zip(days, adj):
            if np.abs(diff) < 1e-10:
                continue
            if day <= bar_days[0] or day > bar_days[-1]:
                continue
            # on the roll day, add to all previous day with 'diff'
            ix = np.clip(np.searchsorted(bar_days, day),0,ndays-1)
            bar[:ix,:,cols]+=diff
        return bar

    def get_next_trade_day_roll_adjust(self, mts_symbol, this_day, get_holiday=False):
        """
        Gets the next day's trading day, contract, roll_adj
        Note, this_day has to be a weekday. In case this_day 
              is a holiday, it search backwards for a trading day.
        get_holiday: if True, allow the next day to be a holiday, in which case just uses the contracts of this day, and no roll
        """
        tdi = mts_util.TradingDayIterator(this_day)
        tdi.begin()
        d0 = tdi.prev()
        tdi = mts_util.TradingDayIterator(this_day, d0)
        this_day=tdi.begin()
        while True:
            try:
                bar_file, mkt, contract = self.get_file_tradable(mts_symbol, this_day, is_mts_symbol=True, get_holiday=False, check_file_exist=True)
                break
            except:
                pass
            this_day = tdi.prev()

        tdi = mts_util.TradingDayIterator(this_day)
        tdi.begin()

        cnt = 0
        while cnt < 3:
            day = tdi.next()
            try :
                bar_file_n, _, contract_n = self.get_file_tradable(mts_symbol,day,is_mts_symbol=True, get_holiday=get_holiday, check_file_exist=False)
                break
            except :
                pass
            cnt += 1

        if cnt >= 3:
            print('next trade day not found for %s after %s'%(mts_symbol, this_day))
            raise ValueError('next trade day not found!')

        ra = 0.0
        if contract_n != contract:
            try :
                next_bar_file = self.get_file_symbol(mkt, contract_n, this_day)
                bar = self.get_bar(bar_file, barsec=1, allow_bfill=True)
                bar_nc = self.get_bar(next_bar_file, barsec=1, allow_bfill=True)

                # find out the adjust
                cix = repo_util.get_col_ix(['close'])[0]
                ra = np.median(bar_nc[-3600*1:,cix]-bar[-3600*1:,cix])

                #normalize with tinfo
                tick_size = float(self.symbol_map.get_tinfo(mts_symbol, this_day, is_mts_symbol=True)['tick_size'])
                ra = np.round(ra/tick_size)*tick_size
            except Exception as e:
                print('problem gettting roll adjust from %s to %s, set ra=0:\n%s'%(bar_file, next_bar_file, str(e)))
        return day, contract_n, ra


TDRepoPath = '/home/mts/run/repo/tickdata_prod'
MTSRepoPath = '/home/mts/run/repo/mts_live_prod'

class MTS_REPO_Live (MTS_REPO) :
    def __init__(self, symbol_map_obj=None) :
        super(MTS_REPO_Live, self).__init__(MTSRepoPath, symbol_map_obj=symbol_map_obj, backup_repo_path=TDRepoPath)

class MTS_REPO_TickData (MTS_REPO) :
    def __init__(self, symbol_map_obj=None) :
        super(MTS_REPO_TickData, self).__init__(TDRepoPath, symbol_map_obj=symbol_map_obj, backup_repo_path=MTSRepoPath)

class MTS_DATA :
    def __init__(self, start_hour, start_min, end_hour, end_min, barsec = 1) :
        """
        start_hour:      New York hour of open time of first bar, which is the first bar time minus barsec,
                         signed integer relative to trading day.
                         For example,
                         -6, meaning previous day's 18:00:00, and
                         9,  meaning current day's 09:00:00.
        start_min:       Minute of open time of first bar, which is the first bar time minus barsec,
                         always positive.
                         For example,
                         30, meaning 30 minutes into the start hour
        end_hour:        New York hour of close time of last bar, which is the last bar time, 
                         signed integer relative to trading day.
                         For example,
                         17, meaning current day's 17:00:00
        end_min:         Minute of close time of last bar, which is the last bar time,
                         always positive.  
                         For example,
                         15, meaning 15 minutes into the end hour
        barsec:          The desired bar period
        """
        self.barsec = int(barsec)
        self.sh = int(start_hour)
        self.sm = int(start_min)
        self.eh= int(end_hour)
        self.em = int(end_min)
        if self.sh > self.eh and self.sh > 0:
            self.sh -= 24
        assert self.eh > self.sh, "end hour less or equal to start hour"
        self.venue_str = ''
        self.trade_day = None

    ##
    ## MTS Bar from TickData
    ##
    def fromTickData(self, quote_file_name, trade_file_name, trading_day_YYYYMMDD, time_zone, pxmul, out_csv_file=None, extended_fields=False, overwrite_repo = False, write_optional=False, tick_size=None, ref_utc=None):
        """
        Read given quote and trade, and return generated bar using self.barsec
        quote_file_name:      full path to tickdata quote file
        trade_file_name:      full path to tickdata trade file
        trading_day_YYYYMMDD: the trading day, in YYYYMMDD
        out_csv_file:         optional output csv file of the bar
        ref_utc:              [start_utc, end_utc], if not given, use default
        """

        if not overwrite_repo and out_csv_file is not None :
            # skip this if it exists
            fa = glob.glob(out_csv_file+'*')
            if len(fa) == 1:
                try:
                    if os.stat(fa[0]).st_size > 1000:
                        if extended_fields:
                            # check number of columns to be more than BASE_COLS
                            import subprocess
                            BASE_COLS = 9  # utc,ohlc,vol,lpx,ltm,vbs
                            gz = (fa[0][-3:]=='.gz')
                            cmd_str = '%s %s | head -n 1'%('zcat' if gz else 'cat', fa[0])
                            l = subprocess.check_output(cmd_str, shell=True).decode().strip().replace('\n','')
                            if len(l.split(',')) > BASE_COLS:
                                print ('found ', fa[0], ' not writting')
                                return []
                except Exception as e:
                    print('problem checking the ' + fa[0] + ' overwriting it.  error: ' + str(e))

        print ('getting quotes from %s'%quote_file_name)
        quote = td_parser.get_quote_tickdata(quote_file_name, time_zone=time_zone, px_multiplier=pxmul)
        print ('getting trades from %s'%trade_file_name)
        trade = td_parser.get_trade_tickdata(trade_file_name, time_zone=time_zone, px_multiplier=pxmul)
        if ref_utc is None:
            start_utc, end_utc = self._get_utc(trading_day_YYYYMMDD)
        else:
            start_utc, end_utc = ref_utc
        print ('generating bars on %s'%trading_day_YYYYMMDD)
        if write_optional:
            td_parser.mts_bar_np(quote, trade, start_utc, end_utc, self.barsec, out_csv_file, tick_size)
            os.system('gzip -f ' + out_csv_file)
            return

        bar, colname = td_parser.daily_mts_bar(trade, quote, 1, start_utc, end_utc-start_utc, extended_fields = extended_fields)
        if self.barsec != 1 :
            bar = repo_util.mergeBar(bar, self.barsec)

        if out_csv_file is not None:
            print ('writting to %s'%out_csv_file)
            repo_util.saveCSV(bar, out_csv_file)
        return bar

    def fromTickDataMultiDay(self, start_day, end_day, mts_symbol, tickdata_path, repo_obj, tickdata_map_obj, extended_fields=False, overwrite_repo=False, include_spread=False, extra_N=[], write_optional=False):
        """
        continuously parse tickdata from start_day to end_day into MTS Bar.
        start_day, end_day: both start and end dyas are inclusive, in format of yyyymmdd
        mts_symbol:  mts symbols, in format of WTI
        tickdata_path: path to tickdata trade and quote
                       tickdata is supposed to be organized into
                       tickdata_path/quote/CLH21_2021_01_04_Q.asc.gz
                       tickdata_path/trade/CLH21_2021_01_04.asc.gz
        repo_obj: object of MTS_REPO, including a SymbolMap object
        tickdata_map_obj: object of TickdataMap object, for getting
                          files for quote and trade, timezone and pxmul
        include_spread:  if true includes spread contracts, i.e. WTI_N1-WTI_N2
        """

        tdi = mts_util.TradingDayIterator(start_day, end_day)
        day = tdi.begin()
        tick_size = None
        while day <= end_day :
            print ("***\nGenerating for %s"%(day))
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                contracts = repo_obj.symbol_map.get_contract_from_symbol(mts_symbol, day, add_prev_day=True, include_spread=include_spread, extra_N=extra_N)
            except:
                contracts = []
            if len(contracts) == 0 :
                print ("nothing found on %s"%(day))
            else :
                try :
                    ref_utc = None
                    try:
                        # fix a tick_size
                        mts_n = mts_symbol.split('_N')[0]+'_N1'
                        tinfo = repo_obj.symbol_map.get_tinfo(mts_n, day, is_mts_symbol=True)
                        tick_size = tinfo['tick_size']
                    except:
                        print('cannot get optional ticksize for %s, continue with parser'%(mts_symbol))

                    try:
                        sutc, eutc = self._get_utc(day,mts_sym=mts_symbol, smap=repo_obj.symbol_map)
                        ref_utc=[sutc,eutc]
                    except:
                        sutc, eutc = self._get_utc(day)
                        ref_utc=[sutc,eutc]

                    for con in contracts :
                        qfile, tfile, tzone, pxmul = tickdata_map_obj.get_tickdata_file(mts_symbol, con, day, add_prev_day=True)
                        qfile = os.path.join(tickdata_path, 'quote', qfile)
                        tfile = os.path.join(tickdata_path, 'trade', tfile)
                        out_csv_file = repo_obj.get_file_symbol(mts_symbol, con, day, create_path = True)
                        print("%s_%s"%(mts_symbol, con))
                        self.fromTickData(qfile, tfile, day, tzone, pxmul, out_csv_file=out_csv_file, extended_fields=extended_fields, overwrite_repo=overwrite_repo, write_optional=write_optional, tick_size=tick_size,ref_utc=ref_utc)
                except :
                    traceback.print_exc()
            day = tdi.next()

    ##
    ## MTS Bar from LiveTrading
    ##
    def _read_mts_bar_file(self, mts_bar_file, do_fix=False, manual_read=True):
        if not manual_read:
            try :
                bar = np.genfromtxt(mts_bar_file,delimiter=',',dtype=float)
                return bar
            except:
                traceback.print_exc()
                print('problem with the bar file %s, manual read'%(mts_bar_file))

        bar = []
        with open(mts_bar_file,'rt') as fp:
            while True:
                l = fp.readline()
                if len(l) == 0:
                    break
                lb = l.replace('\n','').split(',')
                if len(bar) > 0 :
                    if len(lb) != len(bar[-1]) :
                        ALL_COLS=16
                        if len(lb) > len(bar[-1]) and len(lb) != ALL_COLS:
                            print ('read bad line: %s, ignored'%(l))
                            continue
                        lb=lb[:len(bar[-1])]
                    if len(lb[0]) != len(bar[-1][0]):
                        print('found a bad time stamp, %s, removed'%(str(lb)))
                        continue
                    if bar[-1][0] >= lb[0] :
                        print('found a time stamp mismatch, prev(%s) - now(%s), remove both'%(str(bar[-1]), str(lb)))
                        bar.pop(-1)
                        continue
                bar.append(lb)
        bar = np.array(bar).astype(float)
        if do_fix:
            tmp_fn = mts_bar_file+'.tmp'
            np.savetxt(tmp_fn, bar, delimiter=',', fmt='%d,%.7f,%.7f,%.7f,%.7f,%d,%.7f,%d,%d')
        return bar

    def fromMTSLiveData(self, mts_bar_file, trading_day_YYYYMMDD, out_csv_file=None, write_repo=True, mts_sym=None, skip_exist=False, repo_obj=None) :
        """
        Read given mts bar file, and return generated bar using self.barsec
        if write_repo is true, it saves the generated bar to the repo using the
        out_csv_file.  If it is None, then a file name is generated from MTS_REPO
        conventions, with repo path to be mts_live
        """
        if skip_exist and out_csv_file is not None:
            try :
                import glob
                fn = glob.glob(out_csv_file+'*')
                if len(fn) == 1:
                    print('found existing file %s'%(fn[0]))
                    if os.stat(fn[0]).st_size > 1024:
                        print('%s already exists, skipping'%(out_csv_file))
                        return None
            except:
                print('out_csv not found, creating it: %s'%(out_csv_file))

        if repo_obj is None:
            repo_path = "/home/mts/run/repo/mts_live"
            repo_obj = MTS_REPO(repo_path)
        # get all the bars from mts_bar_file
        # forward/backward fill upto stuc-eutc
        sutc, eutc = self._get_utc(trading_day_YYYYMMDD,mts_sym=mts_sym, smap=repo_obj.symbol_map)
        bar = None
        file_stat = os.stat(mts_bar_file)
        if file_stat.st_size < 64:
            print("file size (%d) too small, delete %s"%(file_stat.st_size, mts_bar_file))
            os.remove(mts_bar_file)
            return None
        bar = self._read_mts_bar_file(mts_bar_file)
        n,m = bar.shape
        if bar[0,0] > sutc+1 :
            # in a unlikely situation where the first bar is missing in bar file,
            # try going to MTS Repo to retrieve previos day's bar
            venue, tradable, barsec = self._parse_bar_file(mts_bar_file)
            tdi = mts_util.TradingDayIterator(trading_day_YYYYMMDD, '19700101')
            tdi.begin()
            prev_day = tdi.prev()
            print ('%s first bar missing: try to get (%s:%s) from previous trading day %s'%(mts_bar_file, venue, tradable, prev_day))

            repo_path = "/home/mts/run/repo/mts_live"
            repo_obj = MTS_REPO(repo_path)
            try_cnt = 0
            while True:
                try :
                    prev_bar_file, sym, contract = repo_obj.get_file_tradable(tradable, prev_day)
                    prev_bar = repo_obj.get_bar(prev_bar_file)
                    print ('got %d bars'%(prev_bar.shape[0]))
                    if m0 < m:
                        prev_bar = np.hstack((prev_bar, np.zeros((n0,m-m0))))
                    bar = np.vstack((prev_bar[-1:,:m], bar))
                    break
                except :
                    print('failed to get the previous day bar', prev_day)
                    try_cnt += 1
                    if try_cnt < 7 :
                        print ('try again for previous day')
                        prev_day = tdi.prev()
                        continue
                    print ('no previous bar found, using first open as previous close')
                    bar = np.vstack((bar[0:1,:], bar))
                    bar[0,0] = sutc+1
                    break

        dbar = repo_util.daily1s(bar, sutc, eutc, min_bars = 4*3600) # this throws if less than 4 hours of data
        if write_repo: 
            repo_util.saveCSV(dbar, out_csv_file)

        return dbar

    def updateLive(self, trade_day = None, gzip_expired=True, skip_exist=False, venue_str='', main_cfg_fn='/home/mts/run/config/main.cfg') :
        """
        check the bar directory's 1 second bars and write to the repo's mts_live bars
        """
        # generate a symbol map that takes today's main.cfg about symbol/spread subscriptions
        sm_obj = symbol_map.SymbolMap(main_cfg_fn = main_cfg_fn)

        repo_path = "/home/mts/run/repo/mts_live"
        repo_obj = MTS_REPO(repo_path, symbol_map_obj=sm_obj)
        if trade_day is None :
            # today as the trading day, usually it is called at eod of a trade day
            tdu = mts_util.TradingDayUtil()
            trade_day = tdu.get_trading_day(snap_forward=True)
            tdi = mts_util.TradingDayIterator(trade_day)
            tdi.begin()
            trade_day = tdi.prev()
        print ("Update MTS Bar from Live Trading on %s!"%(trade_day))

        bar_file_glob_str = "/home/mts/run/bar/*_1S.csv"
        if venue_str is not None and len(venue_str) > 0:
            bar_file_glob_str = "/home/mts/run/bar/*"+venue_str+"*_1S.csv"

        fn = glob.glob(bar_file_glob_str)
        for bar_file in fn :
            print ("Processing " + bar_file)
            try :
                venue, tradable, barsec = self._parse_bar_file(bar_file)
                mts_sym=None
                try :
                    out_csv_file, mts_sym, contract = repo_obj.get_file_tradable(tradable, trade_day, create_path=True, get_holiday=True, check_file_exist=False)
                except KeyError as e :
                    # try the next day in case ingestion started after 18:00. Note this wont'be be run on a Sunday
                    # if the tradeable is for next day, don't gzip it (but don't parse it either)
                    print ("tradable %s not found on %s"%(tradable, trade_day))
                    tdi = mts_util.TradingDayIterator(trade_day)
                    tdi.begin()
                    next_trade_day = tdi.next()
                    try:
                        out_csv_file, mts_sym, contract = repo_obj.get_file_tradable(tradable, next_trade_day, create_path=True, get_holiday=True, check_file_exist=False)
                        print("it is for next trade day %s"%(next_trade_day))
                        if gzip_expired:
                            print('NOT gziped')
                    except KeyError as e:
                        print("it is expired!")
                        if gzip_expired:
                            # gzip all barsec files
                            bar_file_s = bar_file.split('_')
                            bar_file_all = '_'.join(bar_file_s[:-1])
                            os.system("gzip -f \'" + bar_file_all + "\'_*S.csv")
                            print ("gzip'ed")
                    continue

                self.fromMTSLiveData(bar_file, trade_day, out_csv_file=out_csv_file,mts_sym=mts_sym, skip_exist=skip_exist, repo_obj=repo_obj)
            except KeyError as e :
                print ("KeyError: " + str(e))
            except AssertionError as e :
                print (str(e))
            except KeyboardInterrupt as e :
                print("Keyboard Interrupt, exit!")
                break
            except :
                traceback.print_exc()
                print("Bar File " + bar_file + " not processed!")

    def updateLiveVenueDay(self, gzip_expired=True, skip_exist=False):
        self.updateLive(trade_day = self.trade_day, gzip_expired=gzip_expired, skip_exist=skip_exist, venue_str=self.venue_str)

    def _parse_bar_file(self, bar_file) :
        """
        bar_file is supposed to be in the format of 
            /home/mts/run/bar/NYM_CLN1_1S.csv
            /home/mts/run/bar/NYM_CLJ2-CLM2_300S.csv
        return:
            venue, tradable and barsec
            i.e. NYM, CLN1, 1
        """
        tk = bar_file.split('/')[-1].split('_')
        return tk[0], '_'.join(tk[1:-1]), int(tk[-1].split('.')[0][:-1])

    def _get_utc(self, trading_day_YYYYMMDD, mts_sym=None, smap=None, add_prev_day=True) :
        """ This is called by the bar writing process.  It gets the daily
        open/close hours from the symbol map to include those bars to the daily MTS bars.
        The number of bars in each day's MTS repo, i.e. the bar file, could be different.

        See _get_utc_alltime() for getting an all inclusive reference utc, that enforce a
        uniform daily bar count, while including all open times historically, upto 24 hours,
        such as Brent (18 to 18)
        """
        day_utc = int(datetime.datetime.strptime(trading_day_YYYYMMDD, '%Y%m%d').strftime('%s'))
        if mts_sym is not None:
            if smap is None:
                smap=symbol_map.SymbolMap(max_N=1)
            sym = mts_sym.split('_')[0] # remove the _Nx part
            tinfo=smap.get_tinfo(sym+'_N1', yyyymmdd = trading_day_YYYYMMDD, is_mts_symbol = True, add_prev_day=add_prev_day)
            sh,sm = np.array(tinfo['start_time'].split(':')).astype(int)[:2]
            eh,em = np.array(tinfo['end_time'].split(':')).astype(int)[:2]
            if sh>=eh:
                sh-=24
        else :
            sh, sm = [self.sh, self.sm]
            eh, em = [self.eh, self.em]
        sutc = day_utc + sh*3600 + sm*60
        eutc = day_utc + eh*3600 + em*60
        return sutc, eutc

    def _get_utc_alltime(self, trading_day_YYYYMMDD, mts_sym):
        """
        This gets start/end utc for mts_symbol for all the time inclusive, 
        i.e. if there were change in open/close time over the years, it gets the most inclusive range
        the purpose is to enforce a uniform shape of daily bars over the years

        It is faster than _get_utc() as it doesn't query symbol_map, but it needs a 'symbol_time_dict'
        to be loaded in the object. This object is separately created and should not be change too often.
        """
        utc0 = int(datetime.datetime.strptime(trading_day_YYYYMMDD).strftime('%s'))
        try :
            sutc,eutc = self.symbol_time_dict[mts_sym]
            return utc0+sutc, utc0+eutc
        except:
            return self._get_utc(trading_day_YYYYMMDD, mts_sym)

    def _get_ref_utc(self, trading_day_YYYYMMDD):
        utc0 = int(datetime.datetime.strptime(trading_day_YYYYMMDD, '%Y%m%d').strftime('%s'))
        sutc = utc0 + self.sh*3600 + self.sm*60
        eutc = utc0 + self.eh*3600 + self.em*60
        return [sutc, eutc]

    def fromB1S_CSV(self, fn, trading_day_YYYYMMDD):
        """
        getting from 1 second bar with format of
        utc, bsz, bp, ap, asz, bvol, avol, last_micro, bqcnt, aqcnt,_,_,lpx

        Return:
        md_dict format in [lr, vol, vbs, lpx, utc] for each 1-second bar on the trade_day
        """
        utc0, utc1 = self._get_utc(trading_day_YYYYMMDD)
        assert utc1 > utc0+1, 'wrong start/stop time'

        bar = np.genfromtxt(fn, delimiter=',',dtype=float)
        assert len(bar) > 1, 'empty bar file ' + fn
        col = {'utc':0, 'bpx':2,'apx':3,'bvol':5,'svol':6,'last_micro':7,'lpx':12}
        utc = bar[:,col['utc']]

        utcb = np.array([utc0+1,utc1])
        ixb = np.clip(np.searchsorted(utc, utcb),0, len(utc)-1)
        ix0 = ixb[0]
        ix1 = ixb[-1]
        if utc[ix1] > utc1:
            ix1-=1
        assert ix1-ix0>0, 'no bar found during this period ' + fn

        ix1+=1
        ts = utc[ix0:ix1]
        mid = (bar[ix0:ix1,col['bpx']] + bar[ix0:ix1,col['apx']])/2
        bvol = bar[ix0:ix1,col['bvol']]
        svol = bar[ix0:ix1,col['svol']]
        vol = bvol + svol
        vbs = bvol - svol
        last_micro = bar[ix0:ix1,col['last_micro']]
        lpx = bar[ix0:ix1,col['lpx']]

        # getting the previous day's mid if possible
        mid0 = mid[0]
        if ix0>0 :
            mid0 = (bar[ix0-1,col['bpx']]+bar[ix0-1,col['apx']])/2

        open_px = np.r_[mid0, mid[:-1]]
        close_px = mid
        high_px = np.max(np.vstack((open_px,close_px)),axis=0)
        low_px = np.min(np.vstack((open_px,close_px)),axis=0)
        dbar = np.vstack((ts,open_px,high_px,low_px,close_px,vol,lpx,last_micro,vbs)).T
        return repo_util.daily1s(dbar, utc0, utc1)

def get_daily_bbo(mts_symbol, start_day, end_day, start_hhmm, end_hhmm, ax_bbo=None, ax_spd=None, bars=None, barsec=300):
    """
    getting the daily avg bbo size and spd
    """
    if bars is None:
        repo = MTS_REPO_TickData()
        bars = repo.get_bars(mts_symbol, start_day, end_day, barsec=barsec, cols = ['utc','lpx','absz','aasz','aspd'])
    ix = []
    for st in [start_hhmm, end_hhmm]:
        assert len(st) == 4, 'time in format of hhmm'
        hh = int(st[:2])
        mm = int(st[2:])
        if hh>=18:
            hh -= 24
        ix.append(((hh+6)*3600 + mm)//barsec)
    ix = np.array(ix).astype(int)
    utc = bars[:,ix[1],0]
    dt = []
    for t in bars[:,ix[1],0]:
        dt.append(datetime.datetime.fromtimestamp(t))
    bsz = np.mean(bars[:,ix[0]:ix[1],2],axis=1)
    asz = np.mean(bars[:,ix[0]:ix[1],3],axis=1)
    asp = np.mean(bars[:,ix[0]:ix[1],4],axis=1)
    if ax_bbo is not None:
        ax_bbo.plot(dt, (bsz+asz)/2, '.-', label=mts_symbol+' avg bid/ask')
    if ax_spd is not None:
        ax_spd.plot(dt, asp, '.-', label=mts_symbol+' avg spread')
    return utc, bsz, asz, asp

#####################################
#  TODO - pack it into holiroll_dict
#####################################
def update_holiday_dict(holiroll_dict, end_day):
    """ holiroll_dict: the holiday and roll dict with format
        {'holiday': [yyyymmdd], 
         'last_day': yyyymmdd
         'rolls': {'roll_day': [yyyymmdd], 'front_contra': [yyyymm]}
    """
    smap=symbol_map.SymbolMap(max_N=1)
    for mkt in holiroll_dict.keys():
        s_day = holiroll_dict[mkt]['last_day']
        tdi = mts_util.TradingDayIterator(s_day)
        tdi.begin()
        day = tdi.next()
        while day <= end_day:
            symbol = mkt + '_N1'
            try :
                tinfo=smap.get_tinfo(symbol, yyyymmdd = day, is_mts_symbol = True)
            except :
                holiroll_dict[mkt]['holiday'].append(day)
            day = tdi.next()
        holiroll_dict[mkt]['last_day'] = end_day
    return holiroll_dict

def get_daily_utc(mts_symbol, barsec, start_day, end_day, holiroll_dict, hours=(-6,0,17,0)):
    symbol = mts_symbol.split('_')[0]+'_N1'
    mkt = symbol.split('_')[0]
    assert mkt in holiroll_dict.keys(), '%s not in holiroll_dict keys %s'%(mkt, str(holiroll_dict.keys()))
    holidays = holiroll_dict[mkt]['holiday']
    last_day = holiroll_dict[mkt]['last_day']
    assert end_day <= last_day, 'holiroll_dict last_day %s less than end_day%s'%(last_day, end_day)

    sh,sm, eh, em = hours
    if sh>eh:
        sh-=24
    shm = sh*3600+sm*60
    ehm = eh*3600+em*60
    assert ehm-shm <= 24*3600, 'hours %s more than 24 hours!'%(str(hours))
    assert (ehm-shm)//barsec*barsec==(ehm-shm), 'hours %s not multiple of barsec %d'%(str(hours), barsec)
    utc0 = []
    tdi = mts_util.TradingDayIterator(start_day)
    day = tdi.begin()
    while day <= end_day:
        if day not in holidays:
            utc0.append(int(datetime.datetime.strptime(day, '%Y%m%d').strftime('%s')))
        day = tdi.next()
    bt = np.arange(shm,ehm,barsec).astype(int)+int(barsec)
    return np.tile(utc0,(len(bt),1)).T+bt

