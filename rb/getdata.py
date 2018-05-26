import numpy as np
import devutil
import pdb
import traceback
import pandas
import cPickle

def read1d_csv(fn, contract_list = []) :
    """
    get the format right
          1       Line number
          2       "AcsyCode",StringType()
          3       "TradingDay",DateType()
           *4       "Datetime",TimestampType()
           *5       "Contract",StringType()
           *6       "OpenPrice",DoubleType() 
           *7       "LastPrice",DoubleType() Last Trade Price
           8       "HighestPrice",DoubleType()
           9       "LowestPrice",DoubleType()
          10       "LowerLimitPrice",DoubleType() (N/A)
          11       "UpperLimitPrice",DoubleType() (N/A)
          *12       "AccumTurnover",DoubleType()
          *13       "Turnover",DoubleType() Since Last Tick *see notes
          *14       "VWAP",DoubleType() Trade Price per share (rb is 10 per contract) = col_13/(col_16 * 10) *see notes
          15       "AccumVolume",LongType() in contracts
          *16       "Volume",IntegerType()  *see notes
          *17       "OpenInterest",LongType() holdings, in contract
          18       "Type",StringType()
          *19       "Buy1Price",DoubleType()
          *20       "Buy1Amount",IntegerType() in contract
          *21       "Sell1Price",DoubleType()
          *22       "Sell1Amount",IntegerType() in contact
    Samples:
ID,AcsyCode,TradingDay,Datetime,Contract,OpenPrice,LastPrice,HighestPrice,LowestPrice,LowerLimitPrice,UpperLimitPrice,AccumTurnover,Turnover,VWAP,AccumVolume,Volume,OpenInterest,Type,Buy1Price,Buy1Amount,Sell1Price,Sell1Amount
1190,SHFrb9000,2015-01-05,2015-01-05 09:00:00.500,rb1504,2578,2578,2578,2578,0,0,103120,103120,2578,4,4,578,FUTURELEVELONE,2500,7,2612,1
1191,SHFrb9000,2015-01-05,2015-01-05 09:00:00.500,rb1505,2550,2549,2550,2543,0,0,469675820,469675820,2549.53761806536,18422,18422,2125546,FUTURELEVELONE,2547,1,2548,1
1195,SHFrb9000,2015-01-05,2015-01-05 09:00:00.500,rb1509,2601,2608,2608,2601,0,0,416440,416440,2602.75,16,16,3486,FUTURELEVELONE,2581,27,2600,197
1196,SHFrb9000,2015-01-05,2015-01-05 09:00:00.500,rb1510,2583,2584,2584,2578,0,0,99646500,99646500,2582.85381026439,3858,3858,368002,FUTURELEVELONE,2577,30,2581,5

    *Note start of day statistics cannot reconcile with previous end of day. maybe the the sod auction?  For now, 
    - ignore the first lines with col_12 = 13
    - ignore the volume couldn't reconcile with previous.  especially sod with previous eod

    * Contract roll for rb: 
      3 contracts, rotates like this, usually  first Monday of the month
          March     July      Nov
      01  Q         T         stop
      05  stop      Q         T
      10  T         stop      Q 

    Input: 
    fn: file name of the daily csv file, expected to be the format above

    output:
    a dict ['contracts' [ 'ts', 'top_level', 'trade_sz', 'trade_px', 'buy_sz_implied', 'ask_sz_implied', 'open_interest']]
    """
    d = np.genfromtxt(fn, delimiter = ',', usecols = (1, 3, 4, 5, 6, 11, 12, 13, 15, 16, 18, 19, 20, 21), dtype=[('code', '|S12'), ('dt', '|S24'), ('con','|S8'), ('open_px', '<f8'),('last_px','<f8'), ('cum_yuan','<f8'),('yuan','<f8'),('avg_px','<f8'),('sz','<i8'), ('oi','<i8'),('bp','<f8'),('bsz','<f8'),('ap','<f8'),('asz','<f8')], skip_header = 1)
    daydict = {}
    con = contract_list
    if len(contract_list) == 0 :
        con = np.unique(d['con'])
    tz_str = '+0800'
    print con
    for c in con :
        ats = []
        atl = []
        ati = [] # trade info
        ix = np.nonzero(d['con'] == c)[0]
        prev_ts = 0
        for ix0 in ix :
            d0 = d[ix0]
            ts = devutil.str_to_utc(d0['dt'], tz_str)
            tl = [ d0['bp'], d0['bsz'], d0['ap'], d0['asz'] ]
            ti = [ d0['open_px'], d0['last_px'] ]
            avg_px = d0['avg_px']
            sz = d0['sz']
            if ts > prev_ts + 1e-9 :
                sz1s = float(sz)/float(ts - prev_ts)
                if prev_ts != 0 :
                    ti += [ sz, sz1s, avg_px, d0['oi'] ]
                    atl.append( tl )
                    ati.append( ti )
                    ats.append( ts )
                prev_ts = ts

        atl = np.array(atl)
        ati = np.array(ati)
        daydict[c] = { 'ts' : np.array(ats).astype(float),'open_px':ati[0, 0], 'last_px':ati[:, 1], 'trade_sz': ati[:, 2], 'trade_sz1s': ati[:, 3], 'trade_avg_px':ati[:, 4], 'open_int':ati[:, 5], 'top_level':atl }
    return daydict


contract_map = { '01': ['07','08','09','10'], '05':['11','12','01','02'], '10':['03','04','05','06'] }
contract_map_range = { '01': [['07','11']], '05':[['11','13'], ['00', '03']], '10':[['03','07']] }
trading_time_rb1 = [ [900, 1015], [1030, 1130], [1330, 1500] ]  # trading time in 2014 and before
trading_time_rb2 = [ [900, 1015], [1030, 1130], [1330, 1500], [2100, 2359], [0, 100] ] # trading time for 2015 and 2016-5-2
trading_time_rb3 = [ [900, 1015], [1030, 1130], [1330, 1500], [2100, 2300] ] # trading time for 2016-5-3 onwards

def get_trading_contract(day_YYYYMMDD, instrument, roll_date = '06') :
    year_str = day_YYYYMMDD[0:4]
    mon_str = day_YYYYMMDD[4:6]
    day_str = day_YYYYMMDD[6:8]
    md = mon_str+day_str
    for k, v in contract_map_range.items() :
        for v0 in v :
            smd = v0[0]+roll_date
            emd = v0[1]+roll_date
            if md >=smd and md < emd :
                # found the contract k
                cy = year_str[2:4]
                if md >= '07'+roll_date :
                    cy = str( int(cy) + 1 )
                return instrument+cy+k #rb1601
    raise ValueError('date string not recognized')

class DailyBar :
    def __init__ (self, day_str, ddict, bar_sec, TZ = 'CN', bar_period = [ [900, 1015], [1030, 1130], [1330, 1500] ]) :
        """
        in order to work with all three years 2014 to 2016, use these time period (may get back to 9:12pm)
        trading_time_rb1 = [ [900, 1015], [1030, 1130], [1330, 1500] ]  # trading time in 2014 and before
        This creates entries with fixed bar time from 900 to 1500, omitting the break periods
        This assumes ts is sorted in hhmmss, i.e. doesn't have night and morning sequeunce
        """
        self.day = day_str
        self.ddict = ddict
        self.ts = ddict['ts']
        self.bar_sec = bar_sec
        self.TZ = TZ
        self.bar_period = bar_period
        bt = []
        bt_utc = []
        bt_start_ix = []
        bt_end_ix = []
        for barp in bar_period :
            hh = '%02d'%(barp[0]/100)
            mm = '%02d'%(barp[0]%100)
            utc0 = devutil.str_to_utc(day_str+hh+mm+'00', TZ)
            bt0 = devutil.hhmmss_bar_time(barp[0]/100, barp[0] % 100, 0, barp[1]/100, barp[1]%100, 0, bar_sec)
            bt_start_ix.append(len(bt))
            bt = np.r_[ bt, bt0 ]
            bt_end_ix.append(len(bt))
            bt_utc = np.r_[ bt_utc, np.arange(len(bt0)) * bar_sec + utc0 ]
        bt = np.array(bt).astype(int)  # this is sorted
        bt_utc = np.array(bt_utc).astype(int)  # this is sorted
        btix = 0

        hhmmss = []
        yymmdd = int(day_str)
        ts = ddict['ts']
        for ts0 in ts :
            hhmmss0, yymmdd0 = devutil.hhmmss_from_utc(ts0, TZ)
            if yymmdd0 == yymmdd :  # ignore the entries from previous calendar day
                hhmmss.append(hhmmss0)
            else :
                hhmmss.append(-240000+hhmmss0)
        hhmmss = np.array(hhmmss) # assuming this is sorted w/duplicates
        tsix = np.clip(np.searchsorted(hhmmss[1:].astype(float), bt.astype(float)+1e-6), 0, len(hhmmss)-1).astype(int)
        # checking the final ones, don't mix to the next zone after, say, 1500
        i = -1
        lastix = tsix[-1]
        while i > -len(tsix):
            if hhmmss[tsix[i]] > bt[i] :
                tsix[i] -= 1
                i -= 1
            else :
                break
        if i <= -len(tsix) :
            raise ValueError('ts is all later than last bar time')
        # checking the first one, don't look ahead
        for i,j in zip(bt_start_ix, bt_end_ix):
            while i < j: 
                if hhmmss[tsix[i]] > bt[i] :
                    tsix[i] = -1
                    i += 1
                else :
                    d0 = devutil.hhmmss_diff_sec(hhmmss[tsix[i]], bt[i])
                    if d0 > max(bar_sec, 600) :
                        # stuck in the time of previous trading zone
                        # try to use the next tick
                        d1 = devutil.hhmmss_diff_sec(bt[i], hhmmss[tsix[i]+1])
                        if d1 < max(bar_sec/2, d0/10) :
                            print 'moving forward on starting bar idx'
                            tsix[i] = tsix[i]+1 
                        else :
                            break
                        i += 1
                    else :
                        break
            if i >= j :
                raise ValueError('ts is all earlier than first bar time')

        self.tsix = tsix.copy()
        self.bt = bt.copy()
        self.bt_utc = bt_utc.copy()
        self.hhmmss = hhmmss.copy()
        self.bt_start_ix = np.array(bt_start_ix)
        self.bt_end_ix = np.array(bt_end_ix)
        self.__to_bar()
        self.bar_dict['info'] = {'day': day_str, 'bar_sec': bar_sec, 'TZ': TZ, 'bar_period': bar_period }

    def validate(self, MIN_TRADE_SIZE_PERIOD) :
        for six, eix in zip (self.bt_start_ix, self.bt_end_ix) :
            sz0 = np.sum( self.bar_dict['trade_sz'][six:eix] )
            if sz0 < MIN_TRADE_SIZE_PERIOD :
                print 'validate failed for period with bar idx:  ', six, eix
                return False
        return True

    def bar_field_keep (self, d, ix) :
        """
        copy d at ix into d0 with length of ix. 
        -1 in ix will be back filled by immediate afterwards
        ix is returned by daily bar
        """
        d0 = []
        for i in ix :
            if i == -1 :
                # removed to be consistent with sum/diff
                #if ix[0] > 0 :
                #    d0.append(d[ix[0]-1])
                #else :
                d0.append(np.nan)
            else :
                d0.append(d[i])
        # backfill of na
        df = pandas.DataFrame(d0)
        pandas.DataFrame.fillna(df, method='bfill', inplace=True)
        d0 = df.as_matrix()[:, 0]
        return np.array(d0)

    def bar_field_sum(self, d, ix) :
        """
        sum of d[ix[i-1]+1 : ix[i]+1] for each bar at ix[i] 
        -1 entries in ix will be zero
        ix is returned by daily bar
        """
        d0 = []
        i0 = -1
        for i in ix :
            if i == -1 :
                d0.append(0)
            else :
                if i0 == -1 :
                    d0.append(d[i])
                else :
                    d0.append(np.sum( d[i0:i+1] ))
                i0 = i+1
        return np.array(d0)

    def bar_field_diff(self, d, ix) :
        """
        d[ix[i]] - d[ix[i-1]] for each bar at ix[i] 
        -1 entries in ix will be zero
        ix is returned by daily bar
        """
        d0 = []
        i0 = -1
        for i in ix :
            if i == -1 :
                d0.append(0)
            else :
                if i0 == -1 :
                    d0.append(0)
                else :
                    d0.append(d[i] - d[i0])
                i0 = i
        return np.array(d0)

    def bar_field_avg(self, d, wt, ix):
        """
        weighted avg of d[ix[i-1]+1 : ix[i] + 1], with corresponding weights
        -1 entries in ix will be back filled by the immediate afterwards
        ix is returned by daily bar
        """
        d0 = []
        i0 = -1
        for i in ix :
            if i == -1 :
                d0.append(np.nan)
            else :
                if i0 == -1 :
                    d0.append(d[i])
                else :
                    wt0 = np.sum(wt[i0:i+1])
                    if wt0 > 0 :
                        d0.append(np.dot(d[i0:i+1], wt[i0:i+1]) / np.sum(wt[i0:i+1]))
                    else :
                        d0.append(d[i])
                i0 = i+1
        # backfill of na
        df = pandas.DataFrame(d0)
        pandas.DataFrame.fillna(df, method='bfill', inplace=True)
        d0 = df.as_matrix()[:, 0]
        return np.array(d0)

    def __to_bar(self) :
        """
        This bins the dd with the give bars
        each entry corresponds to observation up until the bar time
        in particular, the aggregation of the following fields of a daydict:
            'hhmmss' :  the bar time of checking
            'ts' :      keep
            'open_px':  keep
            'last_px':  keep
            'trade_sz': since last bar
            'trade_sz1s': drop *see add
            'trade_avg_px': since last bar, weighted avg
            'open_int':  keep
            'top_level': drop *see add
            add:
            'buy_sz':  since last bar, get from trade_avg_px
            'sell_sz': same
            'mid':     mid of L1 quote
            'ism':     ism of L1 quote
            'ret':     ret since last bar
            'abs ret': abs ret of all ticks since last bar
            'vwap'   : vwap of last bar
            'quote_sz': bsz + asz
        """
        dd, tsix, bt, ts0 = (self.ddict, self.tsix, self.bt, self.hhmmss)
        ts = dd['ts']

        # bar-by-bar filling assuming increasing order in tsix
        bar_dict = {'hhmmss': bt.copy(), 'utc_bar': self.bt_utc.copy(), 'ts': ts0.copy()}
        # open_px, last_px
        bar_dict['open_px'] = dd['open_px']
        bar_dict['close_px'] = dd['last_px'][-1]
        bar_dict['last_px'] = self.bar_field_keep(dd['last_px'], tsix)
        # trade sz since last bar
        bar_dict['trade_sz'] = self.bar_field_sum(dd['trade_sz'], tsix)
        #open_init top_level
        bar_dict['open_int'] = self.bar_field_keep(dd['open_int'], tsix)
        bar_dict['quote_sz'] = self.bar_field_keep(dd['top_level'][:, 1] + dd['top_level'][:, 3], tsix)

        #get derived trd_sz_buy and trd_sz_sell
        bp = dd['top_level'][:, 0]
        bsz = dd['top_level'][:, 1]
        ap = dd['top_level'][:, 2]
        asz = dd['top_level'][:, 3]
        sz = dd['trade_sz']
        avg_px = dd['trade_avg_px']
        # an estimation of bsz and asz
        trd_sz_buy = np.r_[ sz[0]/2, sz[1:] * (avg_px[1:]-bp[:-1])/(ap[:-1] - bp[:-1]) ]
        ix0 = np.nonzero(np.isnan(trd_sz_buy))[0]
        if len(ix0) > 0 :
            trd_sz_buy[ix0] = 0
        trd_sz_buy = np.clip(trd_sz_buy, 0, max(trd_sz_buy))
        trd_sz_buy = np.clip(trd_sz_buy - sz, -max(sz), 0) + sz #make sure its not more 
        trd_sz_sell = sz - trd_sz_buy
        bar_dict['sz_buy'] = self.bar_field_sum(trd_sz_buy, tsix)
        bar_dict['sz_sell'] = self.bar_field_sum(trd_sz_sell, tsix)

        # mid, ism, lgret, abs_ret and vwap
        mid = (bp+ap)/2
        ism = (bp * asz + ap*bsz)/(bsz+asz)
        lgret = np.r_[0, np.log(mid[1:]) - np.log(mid[:-1])]
        absret = np.abs(lgret)
        bar_dict['mid'] = self.bar_field_keep(mid, tsix)
        bar_dict['spd'] = self.bar_field_keep(ap-bp, tsix)
        bar_dict['ism'] = self.bar_field_keep(ism, tsix)
        bar_dict['lgret'] = self.bar_field_sum(lgret, tsix)
        bar_dict['abslgret'] = self.bar_field_sum(absret, tsix) 
        bar_dict['vwap'] = self.bar_field_avg(dd['trade_avg_px'], sz, tsix)
        self.bar_dict = bar_dict

    def saveData(self, fname) :
        with open (fname, 'wb') as f :
            cPickle.dump(self.bar_dict, f)

baddays = [ '20140109', '20150721' ,'20160307', '20160509']
def get_days(start_day, end_day, bar_sec, repo_path, instrument, file_appendix_str = '_FUTURELEVELONE.csv', bar_period = [ [900, 1015], [1030, 1130], [1330, 1500] ] , TZ = 'CN', front_contract_only = True, MIN_TRADE_SIZE_PERIOD = 100, reload = False, save_bar_dict = True) :
    """
    a script to batch read in period of days. 
    daily bars are saved once they are retrieved, if not yet. set reload=True to force load
    get the rolled contract during the time period
    if front_contract_only, then the contract will be given by get_trading_contract()
    otherwise, the contract is determined by the valid contract (see later for valid) having highest trade size
    not valid if total trade size in any bar_periods is less than MIN_TRADE_SIZE
    if the front contract is not valid, it is not included in the data
    """
    ti = devutil.TradingDayIterator(start_day)
    day = ti.cur_day_str()
    daysdict = {}
    file_appendix_str = '_' + instrument+file_appendix_str
    while day <= end_day :
        try :
            if day in baddays :
                print day, ' in bad days, skipping'
                raise ValueError('bad day!')
            fn = repo_path + '/'+day[:4]+'/'+day+file_appendix_str
            print 'getting ', fn
            # try to read if possible
            dbar_dict = None
            con_max = None
            save_fn = day + '_' + str(bar_sec)+'.pkl'
            if not reload :
                try :
                    with open(repo_path+'/'+save_fn, 'rb') as f :
                        dbar_dict = cPickle.load(f)
                    con_max = dbar_dict['con']
                except:
                    #traceback.print_exc()
                    print 'could not load', save_fn, 'try to read'

            if dbar_dict is None or con_max is None:
                con_list = []
                if front_contract_only :
                    fcon = get_trading_contract(day, instrument)
                    con_list = [ fcon ]
                daydict = read1d_csv(fn, con_list)
                # figure out which contract to use
                con_sz = []
                for k, v in daydict.items() :
                    sz = np.sum(v['trade_sz'])
                    con_sz.append(sz)
                con_sz_ix = np.argsort(-np.array(con_sz))
                for i, con_max in enumerate( np.array(daydict.keys())[con_sz_ix] ) :
                    print 'trying ', con_max, ' sz: ', con_sz[con_sz_ix[i]]
                    ddict = daydict[con_max]
                    dbar = DailyBar(day, ddict, bar_sec, bar_period = bar_period, TZ = TZ)
                    dbar_dict = dbar.bar_dict
                    dbar_dict['con'] = con_max
                    # checking the validity of the dbar
                    if not dbar.validate(MIN_TRADE_SIZE_PERIOD) :
                        print con_max, ' not valid, trying a different contract'
                        dbar_dict = None
                    else :
                        # we've got one
                        if save_bar_dict :
                            print 'saving to ', save_fn
                            dbar.saveData(repo_path+'/'+save_fn)
                        break

            if dbar_dict is None :
                print '!! nothing found for day ', day
            else :
                daysdict[day] = {'con':con_max, 'data':dbar_dict}

        except KeyboardInterrupt :
            print 'control-c'
            return daysdict
        except :
            #traceback.print_exc()
            print 'problem getting for ', day, ' skipping ... '
        day = ti.next_str(1)

    return daysdict

def array_from_bdict(daysdict, col_name) :
    v = []
    for day in np.sort(daysdict.keys()) :
        v.append(daysdict[day]['data'][col_name])
    return np.array(v)


