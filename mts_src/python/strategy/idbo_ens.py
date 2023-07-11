import numpy as np
import mts_repo
import copy
import datetime
import mts_util
import strat_utils
import symbol_map
import traceback
import os
import dill
import time
from functools import partial

class DailyInd :
    def __init__(self, bar, bar_cols, bar_hour_dict, barsec=300):
        """ bar: shape [ndays,n,ncols], 
            bar_cols: list of cols in bar, i.e. ['utc','open','high','low','close']
            bar_hour_dict :dict of keys {'trd_open','trd_close','liquid_open','liquid_close','settle','daily_close'}
                           in format of "HHMM", i.e "1430".  
                           Note, HH >= 18 is taken as the previous day's hour
            barsec: bar period in seconds, i.e. 300 seconds
        """
        self.b = bar.copy()
        self.nday, self.n, self.nc = bar.shape
        self.bar_cols = copy.deepcopy(bar_cols)
        self.bar_hour_dict = copy.deepcopy(bar_hour_dict)
        self.barsec=barsec

        # get the bar_ix for the hours
        self._init()

    def _persist_to_dict(self):
        return {'bar':self.b, 'bar_cols': self.bar_cols, 'bar_hour_dict':self.bar_hour_dict, 'barsec':self.barsec}

    @staticmethod
    def retrieve(persist_dict):
        pd = persist_dict
        return DailyInd(pd['bar'], pd['bar_cols'], pd['bar_hour_dict'], barsec=pd['barsec'])

    def bar_update(self, new_bar, roll_adj=0):
        # new_bar shape [ndays, n, nc]
        self.b = np.vstack((self.b,new_bar))[-self.nday:,:,:]

        # do the roll adjustment
        ca = []
        for c in ['open','high','low','close']:
            ca.append(self._col(c))
        ca=np.array(ca).astype(int)
        self.b[:,:,ca]+=roll_adj

    def daily_vol(self, daily_open='', daily_close=''):
        """
        calculate a daily per-bar vol as square_root of sum of return squared
                return is the price difference (close-open)

        daily_open/daily_close: a string, i.e. 'liquid_open', from bar_hour_dict.keys()
                                if empty or None, take all daily bars

        NOTE: default open/close uses the first/last bar. Some market, i.e. Brent
              closes at later time of 17:00. Setting daily_close to be '' includes
              the bars from 17:00 to 18:00.  
              However, this is different with the 'daily_close', which sets to 
              thresholds of const config, see Signal.gen()
        NOTE2: when calculating 'rt', there is a subtle difference between using 'open' 
               versus using previous 'close'.  When previous day is a half-day holiday,
               the previous close is from day before holiday, since there were no bars
               on the holiday. Using open reflects the last price if half day.
        """
        bix = [0,self.n-1]
        for i, k in enumerate([daily_open, daily_close]):
            if k not in self.bar_ix.keys():
                continue
            bix[i] = self.bar_ix[k]
        rt = self.b[:,bix[0]:bix[1]+1,self._col('close')] - self.b[:,bix[0]:bix[1]+1,self._col('open')]
        return np.sum(rt**2,axis=1)**0.5

    @staticmethod
    def _ci(h, l, s, roll_days):
        d_h = np.r_[h[0], np.max(np.vstack((h[1:],s[:-1])),axis=0)]
        d_l = np.r_[l[0], np.min(np.vstack((l[1:],s[:-1])),axis=0)]
        d_r = d_h-d_l
        a_h = strat_utils.rolling_window(d_h, roll_days, np.max)
        a_l = strat_utils.rolling_window(d_l, roll_days, np.min)
        a_r = a_h - a_l
        ci = strat_utils.rolling_window(d_r,roll_days, np.sum) / a_r
        ci_signal = 100*np.log(ci)/np.log(roll_days)
        return ci_signal, d_r, a_r, ci


    def daily_ci(self, roll_days, daily_open='liquid_open', daily_close='liquid_close', settle='settle'):
        bix=[0,self.n-1,self.n-1]
        for i, k in enumerate([daily_open, daily_close, settle]):
            if k not in self.bar_ix.keys():
                continue
            bix[i] = self.bar_ix[k]

        # previous day's settle
        #s = self.b[:,bix[2],self._col('close')]
        s = self.b[:,bix[2]-1,self._col('close')] #stick to 'close', use the previous bar
        h = np.max(self.b[:,bix[0]:bix[1],self._col('high')],axis=1)
        l = np.min(self.b[:,bix[0]:bix[1],self._col('low')],axis=1)
        ci_signal, d_r, a_r, ci = DailyInd._ci(h,l,s,roll_days)
        return ci_signal

        """
        s = self.b[:,bix[2]-1,self._col('close')] #stick to 'close', use the previous bar
        s_prev = np.r_[s[0], s[:-1]]

        # get daily high/low
        d_h = np.max(np.vstack((np.max(self.b[:,bix[0]:bix[1],self._col('high')],axis=1),s_prev)),axis=0)
        d_l = np.min(np.vstack((np.min(self.b[:,bix[0]:bix[1],self._col('low')],axis=1), s_prev)),axis=0)
        d_r = d_h-d_l

        # get aggregated high/low
        a_h = strat_utils.rolling_window(d_h, roll_days, np.max)
        a_l = strat_utils.rolling_window(d_l, roll_days, np.min)
        a_r = a_h - a_l

        ci = strat_utils.rolling_window(d_r,roll_days, np.sum) / a_r
        ci_signal = 100*np.log(ci)/np.log(roll_days)
        return ci_signal
        """

    def daily_hlc(self, hl_open='liquid_open', hl_close='liquid_close', close=''):
        bix = [0, self.n-1, self.n-1]
        for i, k in enumerate([hl_open, hl_close, close]):
            if k not in self.bar_ix.keys():
                continue
            bix[i] = self.bar_ix[k]

        d_h = np.max(self.b[:,bix[0]:bix[1]+1,self._col('high')],axis=1)
        d_l = np.min(self.b[:,bix[0]:bix[1]+1,self._col('low')], axis=1)
        d_c = self.b[:,bix[2],self._col('close')]
        return d_h, d_l, d_c

    def _col(self, col_name):
        for i,cn in enumerate(self.bar_cols):
            if col_name == cn:
                return i
        raise RuntimeError("col " + col_name+ " not found!")

    def _init(self):
        # populate the bar_ix with ix for each hour, based on the bar utc given in bar
        # assuming live bar update in the SAME indexing of k.  
        # Currently fixed to start from 18:00.
        self.bar_ix={}
        utc = self.b[-1,:, self._col('utc')]
        ymd0 = datetime.datetime.fromtimestamp(utc[-1]).strftime('%Y%m%d')
        utc0 = int(datetime.datetime.strptime(ymd0, '%Y%m%d').strftime('%s'))

        for k in self.bar_hour_dict.keys():
            hhmm = int(self.bar_hour_dict[k])
            hh = hhmm//100
            mm = int(hhmm%100)
            if hh >= 18:
                hh -= 24
            t = utc0 + hh*3600 + mm*60 + self.barsec # t matching utc (bar closing time)
            ix = np.clip(np.searchsorted(utc, t),0,len(utc)-1)
            assert (utc[ix] == t) or (ix==len(utc)-1), "time " + k + " " + str(hhmm) + " not found in bar"
            self.bar_ix[k] = ix

    def get_latest_day(self):
        return datetime.datetime.fromtimestamp(self.b[-1,-1,self._col('utc')]).strftime('%Y%m%d')

class IDBO_Signal :
    def __init__(self, daily_ind):
        """
        param_dict: dict of {
                             'daily_vol': {'ma_days1', 'ma_days2', 'open_hour','close_hour'}
                             'daily_ci' : {'roll_days','ma_days1', 'ma_days2', 'open_hour','close_hour','settle_hour'}
                             }
        """
        self.d = daily_ind
        self.param = {'daily_vol': {\
                           'ma_days1':5,\
                           'ma_days2':40, \
                           'open_hour':None, \
                           'close_hour':None}, \
                      'daily_ci' : { \
                           'roll_days':14, \
                           'ma_days1':1,\
                           'ma_days2':40, \
                           'open_hour':'liquid_open', \
                           'close_hour': 'liquid_close', \
                           'settle_hour': 'settle'} \
                     }

    def gen(self) :
        """
        generate the daily_vol and ci, the related signal_tf and pos_wt
        """
        # stdv with dof accounting
        stdn = partial(np.std, ddof=1)

        # daily vol: 
        p = self.param['daily_vol']
        dv = self.d.daily_vol(daily_open=p['open_hour'], daily_close=p['close_hour'])
        dv5 = strat_utils.rolling_window(dv, p['ma_days1'], np.mean)
        dvz = (dv5 - strat_utils.rolling_window(dv, p['ma_days2'], np.mean))/strat_utils.rolling_window(dv,p['ma_days2'], stdn)*2

        # daily ci:
        p = self.param['daily_ci']
        ci = self.d.daily_ci(p['roll_days'], daily_open=p['open_hour'], daily_close=p['close_hour'], settle=p['settle_hour'])
        ciz = (strat_utils.rolling_window(ci, p['ma_days1'],np.mean) - 50)/strat_utils.rolling_window(ci,p['ma_days2'],stdn)

        # signal_tf
        signal_tf = (-np.sign(np.clip(dvz,-1,0))*np.sign(np.clip(ciz,0,1))).astype(int)
        import scipy.stats
        pos_wt = scipy.stats.norm.cdf(np.abs(dvz)+np.abs(ciz)-2) * signal_tf

        # daily high/low/close/vol
        d_h, d_l, d_c = self.d.daily_hlc(hl_open='liquid_open', hl_close='liquid_close',close='daily_close')
        day = self.d.get_latest_day()

        # debug
        #if day == '20220404' :
        #    import pdb
        #    pdb.set_trace()

        return day, dvz, ciz, signal_tf, pos_wt, d_h, d_l, d_c, dv5, dv, ci

    def update(self, bar, roll_adj=0):
        """
        bar has shape [ndays, n, ncol], new daily bar
        """
        self.d.bar_update(bar, roll_adj=roll_adj)

USDR=1.0
GBPR=1.0
EURR=1.0

class IDBO_Data() :
    def __init__(self) :
        self.mkt_hours = \
            [['WTI',1430,700,1500,300,1500,      0.01,  1000,USDR],
            ['Brent',1430,700,1500,300,1500,     0.01,  1000,USDR],
            ['NG',1430,800,1500,800,1500,        0.001, 10000,USDR],
            ['Gasoil',1130,700,1430,300,1430,    0.25,  100,USDR],
            ['HO',1430,800,1500,300,1500,        0.0001,42000,USDR],
            ['RBOB',1430,800,1500,300,1500,      0.0001,42000,USDR],
            ['HGCopper',1300,800,1300,300,1300,  0.0005,25000,USDR],
            ['Gold',1330,700,1600,300,1600,      0.1,   100,USDR],
            ['Silver',1325,700,1600,300,1600,    0.005, 5000,USDR],
            ['Platinum',1305,800,1305,300,1305,  0.1,   50,USDR],
            ['EUR',1500,700,1600,300,1600,       0.00005,125000,USDR],
            ['JPY',1500,700,1600,300,1600,       0.0000005,12500000,USDR],
            ['GBP',1500,700,1600,300,1600,       0.0001,62500,USDR],
            ['AUD',1500,700,1600,300,1600,       0.0001,100000,USDR],
            ['CHF',1500,700,1600,300,1600,       0.0001,125000,USDR],
            ['CAD',1500,800,1600,300,1600,       0.00005,100000,USDR],
            ['NZD',1500,700,1600,300,1600,       0.0001, 100000,USDR],
            ['MXN',1500,800,1600,300,1600,       0.00001,500000,USDR],
            ['SPX',1615,700,1615,300,1615,       0.25,   50.0,USDR],
            ['NDX',1615,700,1615,300,1615,       0.25,   20.0,USDR],
            ['Russell',1615,900,1615,300,1615,   0.1,    50,USDR],
            ['DJIA',1615,700,1615,300,1615,      1,      5,USDR],
            ['FTSE',1130,700,1200,300,1200,      0.5,    10, GBPR],
            ['DAX',1130,700,1200,200,1200,       1.0,    25, EURR],
            #['CAC',1130,700,1200,300,1200,       0.5,    10, EURR], #no CAC from bpipe
            ########['IBEX35',1130,700,1200,200,1200,    1,      1],
            ['EuroStoxx',1130,700,1200,200,1200, 0.1,    50, EURR],
            ['TY',1500,700,1600,300,1600,        0.015625,1000,USDR],
            ['US',1500,700,1600,300,1600,        0.031250,1000,USDR],
            ['TU',1500,700,1600,300,1600,        0.0078125,2000.0,USDR],
            ['FV',1500,700,1600,300,1600,        0.0078125,1000,USDR],
            ['Bund',1115,700,1200,200,1200,      0.01, 1000, EURR],
            ['Gilt',1115,700,1200,300,1200,      0.01, 1000, GBPR],
            ['OAT',1115,700,1200,200,1200,       0.01, 1000, EURR],
            #['BTP',1115,700,1200,200,1200,       0.01, 1000, EURR],
            ['Wheat',1415,930,1415,930,1415,     0.0025,5000,USDR],
            ['Soybeans',1415,930,1415,930,1415,  0.0025,5000,USDR],
            ['SoybeanMeal',1415,930,1415,930,1415,0.1,100,   USDR],
            ['SoybeanOil',1415,930,1415,930,1415, 0.0001,60000,USDR],
            ['Sugar',1255,800,1255,330,1255, 0.0001, 112000,USDR]]

        # for debug
        #self.mkt_hours = self.mkt_hours[:2]

        self.cols = ['utc','open','high','low','close']
        self.barsec = 300
        self.contract = '_N1'

    def init(self, start_day, end_day, md_dict_in={}) :
        """generate ind for each market
            bar_cols: list of cols in bar, i.e. ['utc','open','high','low','close']
            bar_hour_dict :dict of keys {'trd_open','trd_close','liquid_open','liquid_close','settle'}
                           in format of "HHMM", i.e "1430".  
                           Note, HH > 17 is taken as the previous day's hour
            md_dict_in: {symbol:bar} where bar shape [n,ndays,nc]. 
                        if not empty, only markets defined in this dict are initialized
            NOTE - all bar_ix counts from 18:00, i.e. bar k=0 ends at 18:05
        """
        repo_td = mts_repo.MTS_REPO_TickData()
        self.mkt={}
        daily_close = 1700  # set the daily close to be 5pm for all market
        for mh in self.mkt_hours:
            mkt, settle, trd_open, trd_close, liquid_open, liquid_close, tick_size, contract_size, fx = mh
            
            bar_hour_dict = {'trd_open':trd_open,'trd_close':trd_close,'liquid_open':liquid_open, 'liquid_close':liquid_close, 'settle':settle, 'daily_close':daily_close}
            sym = mkt+self.contract
            if sym in md_dict_in.keys():
                bars = md_dict_in[sym]
            else:
                if len(md_dict_in) > 0:
                    # only initialize symbols in md_dict_in, if given
                    continue
                bars = repo_td.get_bars(sym, start_day, end_day, cols=self.cols, barsec=self.barsec, is_mts_symbol=True)
            ind = DailyInd(bars, self.cols, bar_hour_dict, barsec=self.barsec)
            self.mkt[mkt] = {'signal':IDBO_Signal(ind), 'trd_open':trd_open, 'trd_close': trd_close, 'tick_size':tick_size, 'contract_size':contract_size, 'fx':fx}

            # get a barix for trd_open/close
            for k, v in zip(['trd_open_bix', 'trd_close_bix'], [trd_open, trd_close]):
                v = int(v)
                hh = v//100
                mm = int(v%100)
                if hh > 17: hh -= 24
                bar_ix = ((hh+6)*3600 + mm*60)//self.barsec
                assert bar_ix*self.barsec == ((hh+6)*3600 + mm*60), "%s(%s) not multiple of barsec %d"%(k,str(v),self.barsec)
                self.mkt[mkt][k] = bar_ix

    def _persist(self) :
        # create a new mkt without the signal object
        # and dump the ind dict instead
        mkt_dict = {}
        for k in self.mkt.keys():
            mkt_dict[k] = copy.deepcopy(self.mkt[k])
            signal = mkt_dict[k]['signal']
            mkt_dict[k]['signal'] = signal.d._persist_to_dict()
        return mkt_dict

    @staticmethod
    def retrieve(mkt_dict):
        d = IDBO_Data()
        d.mkt = copy.deepcopy(mkt_dict)
        for k in d.mkt.keys():
            ind = DailyInd.retrieve(d.mkt[k]['signal'])
            d.mkt[k]['signal'] = IDBO_Signal(ind)
        return d

    def update_day(self, day, md_dict_in={}, roll_adj_dict={}):
        """ this goes to the MTS Live repo, gets the market data for the day (yyyymmdd)
            for markets defined in md_dict_in and update the signal
            if md_dict_in is empty, then do it for all markets

            roll_adj_dict: {mkt: roll_px_diff}, px_diff to be added to the prices, 
                           including the day being updated
        """
        #repo_live = mts_repo.MTS_REPO_TickData()
        repo_live = mts_repo.MTS_REPO_Live()
        md_dict = {}
        for mkt in self.mkt.keys():
            sym = mkt+self.contract
            if sym in md_dict_in.keys():
                bars = md_dict_in[sym]
            else :
                # only update markets in md_dict_in
                if len(md_dict_in.keys()) > 0:
                    continue
                try :
                    bars = repo_live.get_bars(sym, day, day, cols=self.cols, barsec=self.barsec, is_mts_sybmol=True)
                except :
                    # if no data updated, the gen would still generate the same parameter
                    print('problem getting data for %s on %s, continue'%(sym, day))
                    continue
            if len(bars)==0 or len(bars.shape)!=3:
                print('no market data for %s on %s, skipping'%(sym, day))
                continue

            ra = roll_adj_dict[mkt] if mkt in roll_adj_dict.keys() else 0
            # this adjusts both this 'bar' and existing bar history in signal.indicator
            self.mkt[mkt]['signal'].update(bars, roll_adj=ra)
            md_dict[sym]= bars  # bar is not roll adjusted
        return md_dict

    def gen(self, param_dict_in={}):
        """
        generate daily parameters for the latest day
        """
        param_dict=copy.deepcopy(param_dict_in)
        for mkt in self.mkt.keys():
            day, dvz, ciz, signal_tf, pos_wt, d_h, d_l, d_c, dv5, dv, ci = self.mkt[mkt]['signal'].gen()
            if mkt not in param_dict.keys():
                param_dict[mkt] = {}
            param_dict[mkt]['ref'] = {'signal_tf':int(signal_tf[-1]) , \
                                      'pos_wt'   :pos_wt[-1], \
                                      'prev_h'   :d_h[-1], \
                                      'prev_l'   :d_l[-1], \
                                      'prev_c'   :d_c[-1], \
                                      'prev_v'   :dv5[-1], \
                                      'trd_open' :self.mkt[mkt]['trd_open_bix'], \
                                      'trd_close':self.mkt[mkt]['trd_close_bix'], \
                                      'dv'       :dv[-1], \
                                      'ci'       :ci[-1], \
                                      'day'      : day
                                     }

            ref = param_dict[mkt]['ref']
            if 'ref_hist' in param_dict[mkt].keys():
                param_dict[mkt]['ref_hist'].append( [int(ref['day']), ref['signal_tf'],  ref['pos_wt'], ref['prev_h'], ref['prev_l'], ref['prev_c'], ref['prev_v']] )

            # generate ensembles as [mkt]['ens'][ens_name]['param']
            if 'ens' not in param_dict[mkt].keys():
                param_dict[mkt]['ens']={}

            v = ref['prev_v']
            for m, hk,lk in zip([0,0.25,0.5,1], \
                              ['prev_h','prev_h', 'prev_h','prev_c'], \
                              ['prev_l','prev_l', 'prev_l','prev_c']) :
                h = ref[hk]+m*v
                l = ref[lk]-m*v
                en0 = '%.2f'%(m)
                for sl in ['sar','const']:
                    en=en0+'_'+sl
                    en_dict = copy.deepcopy(ref)
                    en_dict['thres_h']=h
                    en_dict['thres_l']=l
                    en_dict['sl']=sl
                    en_dict['cfg']=en
                    en_dict['mkt']=mkt
                    if en not in param_dict[mkt]['ens'].keys() :
                        param_dict[mkt]['ens'][en] = {}
                    param_dict[mkt]['ens'][en]['param']=en_dict

        return param_dict

    def get_td_md_dict(self, start_day, end_day, mkt_list=None):
        # TODO - 
        #        1, use lpx for ohlc
        #        2, roll adjust
        md_dict = {}
        repo_td = mts_repo.MTS_REPO_TickData()

        if mkt_list is None:
            mkt_list = self.mkt.keys()
        for mkt in mkt_list:
            try :
                sym = mkt+self.contract
                md_dict[sym] = repo_td.get_bars(sym, start_day, end_day, cols=self.cols, barsec=self.barsec, is_mts_symbol=True)
            except KeyboardInterrupt as e:
                return
            except :
                traceback.print_exc()
        return md_dict


class IDBO_Live :
    def __init__(self, idbo_data, model_name='INTRADAY_MTS_IDBO_TF_ENS'):
        """
        Manage live trading of each market, entry and exit. 
        """
        self.data = idbo_data
        self.state = {}

        # for live
        #self.backtest_path = '/home/mts/upload/ZFU_STRAT/INTRADAY_MTS_IDBO_TF_ENS'
        #self.persist_path = '/home/mts/run/recovery/strat/IDBO_TF_ENS'

        # these are for test settings, live settings are set from 
        # MODEL's retrieve() 
        self.model_name = model_name
        self.set_model_path('/tmp/idbo','/tmp/idbo')
        self.persist_fn = 'idbo_ens_obj.dill'
        self.tactical_wt = None

        self.live_dict = None
        self.sw = 1.0  #updated at run_live(), stays 1 for offline simulation
        self.strat_codes = ['TSC-7000-380']
        self.default_scale=5000  # given from 1M annual vol
        self.mts_max_trade_mkt = \
               {'WTI':80,\
                'Brent':80,\
                'NG':60,\
                'Gasoil': 60,\
                'HO':60,\
                'RBOB': 60, \
                'HGCopper': 60,\
                'Gold':80, \
                'Silver':80, \
                'Platinum': 30, \
                'EUR': 80, \
                'JPY': 80, \
                'GBP': 80, \
                'AUD': 80, \
                'CHF': 80, \
                'CAD': 80, \
                'NZD': 80, \
                'MXN': 80, \
                'SPX': 60, \
                'NDX': 40, \
                'Russell': 40, \
                'DJIA': 40, \
                'FTSE': 40, \
                'DAX': 40, \
                'CAC': 20, \
                'EuroStoxx': 80, \
                'TY':100, \
                'US': 100, \
                'TU': 100, \
                'FV': 100, \
                'Bund': 80, \
                'Gilt': 60, \
                'OAT': 30, \
                'Wheat': 20, \
                'Soybeans': 20, \
                'SoybeanMeal': 10, \
                'SoybeanOil': 10, \
                'Sugar': 20, \
                }

        self.mts_max_trade_test = \
               {'WTI':10,\
                'Brent':10,\
                'NG':10,\
                'Gasoil': 10,\
                'HO':10,\
                'RBOB': 10, \
                'HGCopper': 10,\
                'Gold':10, \
                'Silver':10, \
                'Platinum': 10, \
                'EUR': 10, \
                'JPY': 10, \
                'GBP': 10, \
                'AUD': 10, \
                'CHF': 10, \
                'CAD': 10, \
                'NZD': 10, \
                'MXN': 10, \
                'SPX': 10, \
                'NDX': 10, \
                'Russell': 10, \
                'DJIA': 10, \
                'FTSE': 10, \
                'DAX': 10, \
                'CAC': 0, \
                'EuroStoxx': 10, \
                'TY':10, \
                'US': 10, \
                'TU': 10, \
                'FV': 10, \
                'Bund': 10, \
                'Gilt': 10, \
                'OAT': 0, \
                'Wheat': 10, \
                'Soybeans': 10, \
                'SoybeanMeal': 10, \
                'SoybeanOil': 10, \
                'Sugar': 10, \
                }

        self.mts_max_trade = self.mts_max_trade_mkt

    ##########################################
    # modify _init_state() and _roll_state() #
    # for state keys addition/remomval       #
    ##########################################
    def _init_state(self) :
        """ initialize the states for each market/ensemble
            Currently the state has only the position and stop loss
        """
        param = self.data.gen()
        state_init={'pos': {'qty':0, 'k':0, 'px':0, 'pnl_h':[]}, 'sl': {'px':0, 'af': 0.01, 'cl':0.01, 'cap':0.1, 'bar_h':-1e+10, 'bar_l':1e+10, 'prev_bar_h':0, 'prev_bar_l':0,'entry_vol':0}}

        # for each of the market, initialize the state
        for mkt in param.keys():
            param[mkt]['ref_hist']=[]
            ens = param[mkt]['ens']
            for cfg in ens.keys():
                ens[cfg]['state'] = copy.deepcopy(state_init)
            # initialize the pnl, pos/pnl all tuple of (utc,pos/pnl)
            param[mkt]['pnl'] = {'pos':[], 'pnl':[(0,0)], 'daily_pos':[]}
        self.state = param

    def _roll_state(self, roll_adj_dict):
        """
           for each of the market, adjust the price in each ensemble's state
           i.e. the ['ens'][cfg]['state']['pos'] and ['sl'].  
           Note the 
                [cfg]['param'] already is adjusted from the data update and gen
                ealier in the sod

            also roll adjust the daily pnl list for proper backoffice tracking
            self.state[mkt]['pnl']['daily_pos'] - the daily pos
            self.state[mkt]['pnl']['pos']       - the cummulative of all daily pos

            input: 
            roll_adj_dict: {mkt: px_adj}, px_adj to be added to all existing prices
        """
        for mkt in self.state.keys():
            if mkt not in roll_adj_dict.keys():
                continue
            ra = roll_adj_dict[mkt]
            md = self.state[mkt]['ens']
            for cfg in md.keys():
                # adjust px in pos and sl
                pos = md[cfg]['state']['pos']
                pos['px'] += ra
                sl = md[cfg]['state']['sl']
                for k in ['px', 'bar_h', 'bar_l', 'prev_bar_h', 'prev_bar_l'] :
                    sl[k] += ra

            for pd in [self.state[mkt]['pnl']['daily_pos'], self.state[mkt]['pnl']['pos']] :
                for i in np.arange(len(pd)).astype(int):
                    #first 11 columns to be 
                    #utc, pos, cfg, dp_cfg, bar_h, bar_l, bar_c, sl_px, sl_h, sl_l, sl_af 
                    pdl = list(pd[i])
                    for col0 in [4,5,6,7,8,9] :
                        pdl[col0]+=ra
                    pd[i] = tuple(pdl)

    def sod(self, prev_day_yyyymmdd, md_dict_in={}, roll_adj_dict={}, write_backoffice=False, dump_param=True):
        """ start of day: update the parameters using previous day's market data
            md_dict_in:  a md_dict of 1-day worth of bar, that only includes market that traded in the previous day,
                         it is given in simulation, but could leave empty, as in live, when it gets from repo
            roll_adj_dict: {mkt: roll_px_diff}, px_diff to be added to the prices, 
                           including the day being updated
        """
        # update_day() updates the indicator and signal using the given md_dict
        # markets not in md_dict_prev_day is not updated, and the following gen() won't change anything (after re-gen())
        # If today is a holiday and a roll day, the roll adjust is not applied until the next trading day.
        # This is OK since we don't trade today anyway.
        md_dict_prev_day = self.data.update_day(prev_day_yyyymmdd, md_dict_in=md_dict_in, roll_adj_dict=roll_adj_dict)

        # gen() generates ensemble cfg for next day based on updated indicators,
        # but the state, (pos, stop loss), is not updated
        self.state = self.data.gen(param_dict_in=self.state)

        # dump the previous day's position/pnl and next day's parameters
        # for all mkt in the self.state.keys()
        # NOTE - this appends a eod posotion to 'daily_pos', this last
        # line with trade_qty=0, utc=eod time. It will be rolled
        # to the first line next day.  In case roll adjust, this
        # first line will be adjusted before next day, so pnl is
        # in-contract
        if write_backoffice:
            self._write_backoffice_files(prev_day_yyyymmdd, md_dict_prev_day)
        if dump_param:
            self._reset_daily_pos(prev_day_yyyymmdd)
            self._dump_param()
        self._roll_state(roll_adj_dict)
        return md_dict_prev_day

    def _write_backoffice_files(self, trade_day, md_dict_prev_day, fx_map=None):
        """output daily backoffice files, pnl,position,exeuction, from each
        market's daily_pos.  It first create a pos_dict to fill in necessary
        data to create those backoffice files, and then adjust daily_pos's 
        ending px to be the daily close, so that next day's pnl would not
        double counting if we had a position.

        pos_dict is in format of
         {mkt: {'daily_pos', 'tcost' (in price), 'contract_size', 'contract_month', 'tick_size', 'fx', 'symbol'(i.e. mkt_N1)}}
         where daily_pos:  list of tuple from state['mkt']['pnl']['daily_pos']
                            (utc, tgt_qty, trd_cfg_name, trd_qty, 
                             cur_bar_h, cur_bar_l, cur_bar_c, 
                             sl_px, sl_h, sl_l, sl_af)
        """
        pos_dict = {}
        if fx_map is None:
            fx_map = symbol_map.FXMap()
        smap = symbol_map.SymbolMap()
        for mkt in self.state.keys():
            symbol = mkt+self.data.contract
            tcost = self.data.mkt[mkt]['tick_size']
            tick_size = self.data.mkt[mkt]['tick_size']
            contract_size = self.data.mkt[mkt]['contract_size']
            contract_month = 197001  # default contract to be updated if the symbol is trading
            try :
                if symbol in md_dict_prev_day.keys():
                    bar_file, contract_month, contract_size, tick_size, start_time, end_time = strat_utils.get_symbol_day_detail(symbol, trade_day, self.data.barsec, smap=smap)
            except:
                print('failed to get contract detail on a trading day: %s, %s'%(symbol, trade_day))

            pos_dict[mkt] = {'daily_pos': self.state[mkt]['pnl']['daily_pos'], \
                             'tcost': float(tick_size), \
                             'contract_size': float(contract_size), \
                             'contract_month': int(contract_month), \
                             'tick_size': float(tick_size), \
                             'fx': fx_map.get(mkt, trade_day, default=self.data.mkt[mkt]['fx']), \
                             'symbol': symbol,  \
                             'bar_unadj': md_dict_prev_day[symbol] if symbol in md_dict_prev_day.keys() else None, \
                             'n': self.data.mkt[mkt]['signal'].d.n}
        dump_daily_backtest(trade_day, pos_dict, self.sw,\
                os.path.join(self.backtest_path, self.exec_fn), \
                os.path.join(self.backtest_path, self.pnl_fn), \
                os.path.join(self.backtest_path, self.pos_fn))

    def _reset_daily_pos(self, day):
        # write daily position to the file
        # roll 'daily_pos' rows to 'pos'
        # write backtest files
        for mkt in self.state.keys():
            pd = self.state[mkt]['pnl']
            cnt = len(pd['daily_pos'])
            if cnt >= 1:
                # new position change, write to daily_pos file
                dn = os.path.join(self.persist_path, 'pos_dump')
                os.system('mkdir -p '+ dn + ' > /dev/null 2>&1')
                fn = os.path.join(dn,'pos_'+mkt)
                with open(fn,'at') as fp:
                    for row in pd['daily_pos'][1:]:
                        utc, pos, cfg, dp_cfg, bar_h, bar_l, bar_c, sl_px, sl_h, sl_l, sl_af = row
                        fp.write('%s, %f, %s, %f, %f, %f, %f, %f, %f, %f, %f\n'%(datetime.datetime.fromtimestamp(utc).strftime('%d-%b-%Y %H:%M:%S'), pos, cfg, dp_cfg, bar_h, bar_l, bar_c, sl_px, sl_h, sl_l, sl_af))

                if len(pd['pos']) == 0:
                    pd['pos'].append(pd['daily_pos'][0])
                pd['pos'] += pd['daily_pos'][1:]
                pd['daily_pos'] = pd['pos'][-1:]

    def _dump_param(self):
        dump_keys = ['signal_tf', 'pos_wt', 'prev_h', 'prev_l', 'prev_c', 'prev_v', 'trd_open', 'trd_close', 'dv', 'ci']
        for mkt in self.state.keys():
            ref = self.state[mkt]['ref']
            day = ref['day']
            dn = os.path.join(self.persist_path, 'param_dump')
            os.system('mkdir -p ' + dn + ' > /dev/null 2>&1')
            fn = os.path.join(dn, 'param_'+mkt)
            try :
                fp=open(fn,'rt')
                fp.close()
            except:
                fp=open(fn,'at')
                fp.write('day')
                for k in dump_keys:
                    fp.write(',\t%s'%(k))
                fp.write('\n')
                fp.close

            with open(fn, 'at') as fp:
                fp.write('%d'%(int(day)))
                for k in dump_keys:
                    fp.write(',\t%f'%(float(ref[k])))
                fp.write('\n')
                fp.close()

    def _run_bar(self, k, bar_utc, bar_h, bar_l, bar_c, mkt_dict, pnl_dict):
        """
        This runs a configuration (ensemble) defined by mkt_dict, with the current bar and update the
        given pnl_dict. 
        The mkt_dict is from state[mkt]['ens'][cfg]
        The pnl_dict is from state['pnl']

        update the state with h/l/c
        return new position
        mkt_dict:{ 'param':{'thres_h','thres_l','sl', 'prev_v', 'pos_wt','signal_tf','trd_open','trd_close','mkt'}\
                   'state':{'pos': {'qty', 'k', 'px', 'pnl_h'},
                            'sl' : {'px','af','cl','cap','bar_h','bar_l','prev_bar_h','prev_bar_l','entry_vol'}
                           }
                 }
        Note - this would run by both run_live() and sim(), k always starts from previous day's 18:00,
               i.e. k = 0 refers to the bar closing at 18:05
        """
        #if bar_utc == 1649264400:
        #    import pdb
        #    pdb.set_trace()

        state = mkt_dict['state']
        param = mkt_dict['param']
        cfg_name = param['cfg']
        mkt = param['mkt']
        pos = state['pos']
        sl = state['sl']
        cur_bar = (bar_h, bar_l, bar_c)
        barsec = self.data.barsec
        fx = self.data.mkt[mkt]['fx']

        # allow trade upon open but 1 bar before close
        if k < param['trd_open'] - 1 or k >= param['trd_close']-1:
            return

        if pos['qty'] == 0:
            # look for entrance
            if param['signal_tf'] == 0:
                return
            if bar_h > param['thres_h']:
                # enter long
                enter_size = self._get_trade_size(mkt,param['pos_wt']/fx/param['prev_v'], cfg_name)
                if abs(enter_size) > 1e-10:
                    pos['qty'] = enter_size
                    pos['px']=bar_c
                    pos['k']=bar_utc

                    # set the stop loss
                    sl['px'] = bar_c - param['prev_v']
                    sl['bar_h'] = bar_h
                    sl['bar_l'] = bar_l
                    sl['af'] = 0.01
                    sl['prev_bar_h'] = bar_h
                    sl['prev_bar_l'] = bar_l
                    sl['entry_vol'] = param['prev_v']

                    # update the position
                    self._upd_pos(pnl_dict, bar_utc, pos['qty'], cfg_name, sl, cur_bar)

                    # disable entrance for the day
                    # this will be updated on the next daya
                    param['signal_tf'] = 0
                # else:
                #     self.logger.logDebug('trying to enter but not enough for 1 contract, will try later')

            elif bar_l < param['thres_l']:
                # enter short
                enter_size = self._get_trade_size(mkt,param['pos_wt']/fx/param['prev_v'], cfg_name)
                if abs(enter_size) > 1e-10:
                    pos['qty'] = -enter_size
                    pos['px']=bar_c
                    pos['k']=bar_utc
                    sl['px'] = bar_c + param['prev_v']
                    sl['bar_h'] = bar_h
                    sl['bar_l'] = bar_l
                    sl['af'] = 0.01
                    sl['prev_bar_h'] = bar_h
                    sl['prev_bar_l'] = bar_l
                    sl['entry_vol'] = param['prev_v']
                    self._upd_pos(pnl_dict, bar_utc, pos['qty'], cfg_name, sl, cur_bar)
                    param['signal_tf'] = 0
            return

        if pos['qty'] > 0:
            new_high = False
            # only update sl px and h/l after the first bar
            if bar_utc > pos['k'] + barsec:
                # update the sl_h/l first using previous bar_h/l
                if sl['prev_bar_h'] > sl['bar_h']:
                    new_high = True
                    sl['bar_h'] = sl['prev_bar_h']

                # update the sl px using existing sl_px, sl_h/l and af
                if param['sl'] == 'sar':
                    # update sl price with updated high
                    sl['px']+=((sl['bar_h']-sl['px'])*sl['af'])
                    if new_high:
                        # update af to be used for next iteration
                        sl['af'] = min(sl['af']+sl['cl'],sl['cap'])
                else :
                    # const
                    if new_high:
                        sl['px']=sl['bar_h']-sl['entry_vol']

            sl['prev_bar_h'] = bar_h
            sl['prev_bar_l'] = bar_l

            # compare current bar_l with the updated sl price
            if bar_l >= sl['px']:
                # trace when position is on
                #if (cfg_name[:5] == '0.00_' or cfg_name[:5] == '0.25_') and bar_utc > 1649023500:
                #    self._upd_pos(pnl_dict, bar_utc, 0, cfg_name, sl, cur_bar)
                return
            # sl hit, exit long, fall through
        else:
            new_low = False
            if bar_utc > pos['k'] + barsec:
                if sl['prev_bar_l'] < sl['bar_l']:
                    new_low = True
                    sl['bar_l'] = sl['prev_bar_l']
                if param['sl'] == 'sar':
                    sl['px']+=((sl['bar_l']-sl['px'])*sl['af'])
                    if new_low:
                        sl['af'] = min(sl['af']+sl['cl'],sl['cap'])
                else :
                    if new_low:
                        sl['px']=sl['bar_l']+sl['entry_vol']
            sl['prev_bar_h'] = bar_h
            sl['prev_bar_l'] = bar_l
            if bar_h <= sl['px']:
                #if (cfg_name[:5] == '0.00_' or cfg_name[:5] == '0.25_') and bar_utc > 1649023500:
                #    self._upd_pos(pnl_dict, bar_utc, 0, cfg_name, sl, cur_bar)
                return

        # close position, no scale needed, clear what's left
        qty = pos['qty']
        px0 = pos['px']
        pnl = (bar_c - px0)*qty
        pos['pnl_h'].append([qty,pos['k'],px0,bar_utc,bar_c,pnl])

        pos['qty'] = 0
        self._upd_pos(pnl_dict, bar_utc, -qty, cfg_name, sl, cur_bar)
        self._upd_pnl(pnl_dict, bar_utc, pnl)

    def _init_live_dict(self, trade_day):
        """
        setup trigger time, update the bar_file, contracts
        return: 
        live_dict{'utc0': starting utc of 18:00 current trading day, 
                  'utc1': ending utc of 17:00 current trading day
                  'utc':  last utc
                  'mkt': {mkt: {'bar_file':  the live bar file name
                                'trade_day': the trade day for bar_file
                                'contract_month': contract month
                                'contract_size': contract size, integer
                                'tick_size': float
                                'start_time' and 'end_time': string like '18:00:00'
                               }
                         }
                  }

        Note - live_dict['mkt'] only includes markets open on trade_day, 
        this has to be consistent with the sod().  Essentially the
        symbol map defines whether a market should be run on a day or not,
        by live_dict['mkt'].  run_live() iterates on it.
        """
        live_dict = {}
        # utc0, utc1: start/end utc, fixed at 18:00 to 17:00
        utc0 = int(datetime.datetime.strptime(trade_day, '%Y%m%d').strftime('%s')) - 6*3600
        utc1 = int(datetime.datetime.strptime(trade_day, '%Y%m%d').strftime('%s')) + 17*3600

        live_dict['utc0'] = utc0
        live_dict['utc1'] = utc1
        live_dict['utc'] = utc0
        ld_mkt = {}
        smap = symbol_map.SymbolMap()
        for mkt in self.state.keys():
            symbol = mkt + self.data.contract
            barsec = self.data.barsec
            try :
                bar_file, contract, contract_size, tick_size, start_time, end_time = strat_utils.get_symbol_day_detail(symbol, trade_day, barsec, smap=smap)
                ld_mkt[mkt] = {'bar_file': bar_file, 'trade_day':trade_day, 'contract_month':contract, 'contract_size':contract_size, 'tick_size':tick_size, 'start_time':start_time, 'end_time':end_time}
            except :
                print('%s not a trading day for %s, live_dict not updated'%(trade_day, mkt))
        live_dict['mkt'] = ld_mkt
        return live_dict

    def run_live(self, trade_day=None):
        """
        be on the market from 18:00 to 17:00, and run on every 5 minute bar
        Daily process:
            Call live trading stop at 17:00, and run sod at 17:50.
        Persistence: 
            It saves the states every bar time after process all markets
        Load:
            Figure out the current k, gets all bars since last k, run 
            all bars for alll market, send orders at the each market
        """

        if trade_day is None:
            tdu = mts_util.TradingDayUtil()
            cur_dt = datetime.datetime.now()
            trade_day = tdu.get_trading_day_dt(dt=cur_dt, snap_forward=True)
        utc0 = int(datetime.datetime.strptime(trade_day,'%Y%m%d').strftime('%s')) - 3600*6

        # do we need to do sod_live?
        ld = self.live_dict
        if ld is None:
            # initialization
            self.live_dict = self._init_live_dict(trade_day)
        else:
            assert ld['utc0'] <= utc0, 'Invalide utc0 in live_dict on %s(%d), live_dict:%s'%(trade_day, utc0, str(ld))
            if ld['utc0'] < utc0:
                # this does everything for live 
                # gets new live_dict
                # gets the roll adjust
                # calls the sod to update param/backoffice
                # persist afterwards
                self.live_sod(trade_day, write_backoffice=True)
                self.sw=strat_utils.get_strategy_weight(self.model_name,  '/home/mts/upload/STRATEGY_WEIGHTS.yaml')
                self._persist(fn_date = trade_day)

        self.logger = mts_util.MTS_Logger(self.model_name)
        self.should_run = True
        barsec = self.data.barsec
        contract = self.data.contract
        self.sw=strat_utils.get_strategy_weight(self.model_name,  '/home/mts/upload/STRATEGY_WEIGHTS.yaml')
        self.logger.logInfo('run_live() started - sw(%.2f), tactical_wt(%s)'%(self.sw, str(self.tactical_wt)))

        # match position with mts
        pos_matched = IDBO_Live.match_position(self, self.logger)
        if not pos_matched:
            self.logger.logInfo('not all IDBO positions match with MTS, trade will start with the IDBO position!')
        #assert pos_matched, "we cannot start trading without position matching"

        while self.should_run:
            # run the tight loop of bar
            #    * gets latest bar
            #    * run the bar
            #    * set target position
            #    * persist the entire object
            dt = datetime.datetime.now()
            cur_utc = int(dt.strftime('%s'))
            last_utc = self.live_dict['utc']
            k = (last_utc-utc0)//barsec

            if cur_utc>=last_utc+barsec:
                # due for bar update
                k = (last_utc-utc0)//barsec
                barcnt = (cur_utc-last_utc)//barsec

                # run all open markets in live_dict
                for mkt in self.live_dict['mkt'].keys():
                    self.logger.logInfo('checking %s on bar %d, barcnt(%d)'%(mkt, k, barcnt))

                    # skip if not in tradeing hours
                    trd_k0 = self.state[mkt]['ref']['trd_open']
                    trd_k1 = self.state[mkt]['ref']['trd_close']
                    if k+barcnt-1 < trd_k0-2 or k+barcnt-1 > trd_k1+1:
                        continue

                    ld = self.live_dict['mkt'][mkt]
                    bfile = ld['bar_file']
                    mkt_dict = self.state[mkt]['ens']
                    pnl_dict = self.state[mkt]['pnl']
                    symbol = mkt+contract
                    # take close as lpx
                    bars = strat_utils.get_bar_cols(bfile, barcnt, cols=[0,1,2,3,6])
                    if bars is None:
                        print('failed to get bar for %s (%s), wanted %d bars '%(mkt, bfile, barcnt))
                        continue
                    elif len(bars[:,0]) != barcnt and bars[-1,0]<last_utc+barsec*barcnt:
                        self.logger.logError('bar file (%s) have less than %d bars! Market %s not processed!'%(bfile,barcnt,mkt))
                        continue
                    cur_utc0 = last_utc  #last_utc initialized as 18:00, same as utc0

                    # update barcnt starting from k
                    for k0 in np.arange(barcnt).astype(int):
                        cur_k0 = k+k0
                        cur_utc0+= barsec
                        # find cur_utc0 in bars[:,0]
                        if len(bars) == 0:
                            self.logger.logInfo('no bars found for %s'%(str(mkt)))
                            break

                        ix = np.clip(np.searchsorted(bars[:,0], cur_utc0),0,barcnt-1)
                        if bars[ix,0] != cur_utc0:
                            # this bar cur_utc0 not found in the bar file, not started?
                            self.logger.logInfo('bar not found (%s) at %s(k=%d), using %s: %s'%(\
                                    str(mkt),\
                                    str(datetime.datetime.fromtimestamp(cur_utc0)), cur_k0,\
                                    str(datetime.datetime.fromtimestamp(bars[ix,0])), \
                                    str(bars[ix,:])))
                        bar0 = bars[ix,:]
                        bar_utc, bar_o, bar_h, bar_l, bar_c = bar0
                        for cfg in mkt_dict.keys():
                            self._run_bar(cur_k0, bar_utc, bar_h, bar_l, bar_c, mkt_dict[cfg], pnl_dict)

                    # getting the target position from the run
                    try:
                        tgt_pos = pnl_dict['daily_pos'][-1][1]
                    except:
                        self.logger.logInfo('%s not in daily_pos, set as 0'%(mkt))
                        tgt_pos = 0
                    self._set_mts_pos(symbol, tgt_pos)
                    self.logger.logInfo('set position %s to %f'%(symbol, float(tgt_pos)))

                # update live_dict
                self.live_dict['utc'] += barcnt*barsec

                # persist
                self.logger.logInfo('persisting')
                self._persist()

            if cur_utc > self.live_dict['utc1']:
                break
            else:
                time.sleep(1)

        self.logger.logInfo('exit the loop')
        cur_utc = int(datetime.datetime.now().strftime('%s'))
        if cur_utc > self.live_dict['utc1']:
            self.logger.logInfo('done with the day!')
            print('Done with the day!')

    def _set_mts_pos(self, symbol, tgt_pos):
        # the trading time not checked here
        for strat in self.strat_codes:
            strat_utils.set_target_position(strat, symbol, np.round(tgt_pos), self.logger, twap_minutes = 5)

    def live_sod(self, trade_day, write_backoffice=False):
        """
        It gets the roll adjustment from today to previous trading day, and gets unadjusted
        bars from previous trading day to call sod(), where signal and state is updated and
        backoffice files updated.

        Input :
            trade_day: the next day to start from
        """
        # gets new live_dict
        # gets the roll adjust
        # calls the sod to update param/backoffice
        prev_lh = self.live_dict
        prev_trade_day = datetime.datetime.fromtimestamp(prev_lh['utc1']).strftime('%Y%m%d')
        repo_live = mts_repo.MTS_REPO_Live()
        roll_adj_dict = {}
        for mkt in self.state.keys():
            symbol = mkt + self.data.contract
            # prev_trade_day maybe a holiday, it will search backwards for a prev_trade_day
            next_trade_day, next_contract, roll_adj = repo_live.get_next_trade_day_roll_adjust(symbol, prev_trade_day, get_holiday=False)
            if next_trade_day != trade_day:
                print('%s not a trading day on %s, next day is %s'%(symbol, trade_day, next_trade_day))
            else:
                roll_adj_dict[mkt] = roll_adj

        # getting the md_dict for prev_trade_day
        md_dict = {}
        for mkt in prev_lh['mkt'].keys():
            symbol = mkt + self.data.contract
            barsec = self.data.barsec
            cols = ['utc', 'open','high','low','lpx']
            try :
                bar = repo_live.get_bars(symbol, prev_trade_day, prev_trade_day, barsec=barsec, get_roll_adj=False, cols = cols, is_mts_symbol=True, get_holiday=False)
                md_dict[symbol] = bar
            except:
                print('%s sod not performed, no data on %s'%(symbol, prev_trade_day))

        self.sod(prev_trade_day, md_dict_in=md_dict, roll_adj_dict=roll_adj_dict, write_backoffice=write_backoffice, dump_param=True)

        #update the live_dict, needs to be updated after the sod, as the current contract detail will be used in backoffice in sod()
        self.live_dict=self._init_live_dict(trade_day)

    def sim_day_mkt(self, mkt, daily_bars):
        """
        mkt:  a string, i.e. 'WTI'
        daily_bars:  shape [n, ncol],  the cols from self.data
        """
        mkt_dict = self.state[mkt]['ens']
        # run each bar for all the configs

        # getting the starting reference 'utc0' for k
        day = datetime.datetime.fromtimestamp(daily_bars[-1,0]).strftime('%Y%m%d')
        utc0 = int(datetime.datetime.strptime(day, '%Y%m%d').strftime('%s')) - 6*3600
        utc1 = utc0 + 23*3600
        barsec = self.data.barsec
        for bar in daily_bars:
            bar_utc, bar_o, bar_h, bar_l, bar_c = bar[:5]
            if bar_utc <= utc0 or bar_utc > utc1:
                continue
            k = int(np.round((bar_utc-utc0)/barsec))-1
            for cfg in mkt_dict.keys():
                self._run_bar(k, bar_utc, bar_h, bar_l, bar_c, mkt_dict[cfg], self.state[mkt]['pnl'])

    def sim(self, md_dict, lastday_sod=True, run_live_sod=False, write_backoffice=False, persist_sod=False):
        """
        md_dict: key symbol, value shape [ndays, n, nc], 
                 note that ndays and n could be different for symbols
                 nc has to at least include first 5 cols as utc,ohlc
        lastday_sod: performs sod on lastday of a mkt. Set it false to
                     leave the object to be updated at start of day of
                     next trade day, i.e. run_live()
        run_live_sod: run live_sod() (assuming the live_dict updates) instead of run sod()

        simulate for all mkt in md_dict and for all days in each, using sim_day_mkt()
        Note, since days in mkt can be different, need to enforce the same day for all
        mkt, to simulate a day.
        
        the simulation starts with the earlierst day and end at latest day.
        """
        day_symbol = {}  # {symbol: {trd_day: nd}}, where nd is the second index of bar
        _trade_day_set = set()
        lastday_symbol = {} # {symbol: last_trd_day}, used for checking lastday_sod
        for symbol in md_dict.keys():
            ds = {}
            for i, t in enumerate(md_dict[symbol][:,-1,0].flatten()):
                td0 = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d')
                ds[td0]=i
            day_symbol[symbol] = ds
            _trade_day_set.update(ds.keys())
            lastday_symbol[symbol]=td0
        day_arr = np.sort(list(_trade_day_set))

        # for each day, simulate all market if bar available
        # skip start/end day difference and holidays
        for day in day_arr:
            pstr = 'day '+day+':'
            md_dict_day = {}  # used for calling sod()

            if run_live_sod:
                self.live_sod(day, write_backoffice=write_backoffice)
                if persist_sod:
                    self._persist(fn_date = day)

            for symbol in day_symbol:
                if day in day_symbol[symbol].keys():
                    d =day_symbol[symbol][day]
                    daily_bar = md_dict[symbol][d,:,:]
                    # populate the day dict for sod()
                    if lastday_sod or lastday_symbol[symbol] != day:
                        n0,nc0 = daily_bar.shape
                        md_dict_day[symbol] = daily_bar.reshape((1,n0,nc0)).copy() #save for sod()
                    #else:
                    #    print('%s sim sod will not run on %s'%(symbol, day))

                    mkt0 = symbol.split('_')[0]
                    self.sim_day_mkt(mkt0,daily_bar)
                    pstr += (' '+mkt0)
            print(pstr)

            if not run_live_sod:
                if len(md_dict_day.keys()) > 0:
                    self.sod(day, md_dict_in=md_dict_day,  write_backoffice=write_backoffice)

    def _persist(self, fn_date = ''):
        persist_fn = self.persist_fn+fn_date
        fn = os.path.join(self.persist_path, persist_fn)
        with open(fn,'wb') as fp:
            dill.dump({'state':self.state, 'live_dict':self.live_dict, 'data':self.data._persist(), 'sw':self.sw,'tactical_wt':self.tactical_wt}, fp)

        # backup existing running dump, and start a new one
        if fn_date != '' :
            running_fn = os.path.join(self.persist_path, self.persist_fn)
            os.system('mv %s %s > /dev/null 2>&1'%(running_fn, running_fn+'_'+fn_date))
            os.system('cp %s %s'%(fn, running_fn))

    @staticmethod
    def retrieve(obj_dict, backtest_path = '/home/mts/upload/ZFU_STRATS', \
                         persist_path = '/home/mts/run/recovery/strat', \
                         persist_fn = 'idbo_ens_obj.dill', \
                         strat_codes = ['TSC-7000-380'],\
                         model_name = None):
        data = IDBO_Data.retrieve(obj_dict['data'])
        idbo = IDBO_Live(data)
        idbo.state = copy.deepcopy(obj_dict['state'])
        idbo.live_dict = copy.deepcopy(obj_dict['live_dict'])
        if model_name is not None:
            # mainly here for setting up live object from simulation
            idbo.model_name=model_name
            idbo.set_model_path(backtest_path, persist_path)
        idbo.persist_fn = persist_fn
        idbo.strat_codes = strat_codes
        if 'sw' in obj_dict.keys():
            idbo.sw = obj_dict['sw']
        if 'tactical_wt' in obj_dict.keys():
            idbo.tactical_wt = copy.deepcopy(obj_dict['tactical_wt'])
        return idbo

    def _get_trade_size(self, mkt, size, cfg_name):
        """ this should be called for entering trade, since
        scale could change from time to time.
        """
        tw = 1.0
        if self.tactical_wt is not None:
            try:
                tw = self.tactical_wt[mkt][cfg_name]
            except:
                self.logger.logInfo('tactical weight for %s-%s not found, using 1'%(mkt,cfg_name))
        size*=(self.default_scale/self.data.mkt[mkt]['contract_size']*tw)
        size*=self.sw
        size = min(np.round(size), self.mts_max_trade[mkt])
        return float(size)

    def _upd_pos(self, pnl_dict, utc, dp, cfg_name, sl, cur_bar):
        # daily_pos always have at least one row
        # first row is the previous day's ending position
        if len (pnl_dict['daily_pos']) == 0:
            p = 0
        else :
            u, p = pnl_dict['daily_pos'][-1][:2]
        bar_h, bar_l, bar_c = cur_bar
        sl_px, sl_h, sl_l, sl_af = (sl['px'], sl['bar_h'], sl['bar_l'], sl['af'])
        pnl_dict['daily_pos'].append((utc, p+dp, cfg_name, dp, bar_h, bar_l, bar_c, sl_px, sl_h, sl_l, sl_af))

        # debug
        #if utc == 1649264400:
        #    import pdb
        #    pdb.set_trace()

    def _upd_pnl(self, pnl_dict, utc, pnl):
        u, p = pnl_dict['pnl'][-1]
        pnl_dict['pnl'].append((utc, p+pnl))

    def shutdown(self):
        print('IDBO_Live shutting down')
        self.should_run = False

    def set_tactical(self, tactical_fn):
        self.tactical_wt = None
        if tactical_fn is not None:
            tactical_fn = os.path.join(self.backtest_path,tactical_fn)
            self.tactical_wt = IDBO_Live.get_tactical_weight(tactical_fn)

    def set_model_path(self, backtest_path, persist_path):
        self.backtest_path = os.path.join(backtest_path,self.model_name)
        self.persist_path = os.path.join(persist_path, self.model_name)
        self.exec_fn = 'ExecutionLog-'+self.model_name
        self.pnl_fn = 'Pnl-'+self.model_name
        self.pos_fn = 'Position-'+self.model_name

    @staticmethod
    def get_tactical_weight(fn):
        # weight file in format of 
        # market,0.00_sar,0.25_sar,0.50_sar,1.00_sar,0.00_const,0.25_const,0.50_const,1.00_const,max_trd_sz,annaul_vol($)

        print('getting tactical weight from', tactical_fn)
        mkt_cfg_wt = {}
        cfg_keys = ['0.00_sar','0.25_sar','0.50_sar','1.00_sar','0.00_const','0.25_const','0.50_const','1.00_const']
        with open(fn,'rt') as fp:
            fp.readline() # skip the header
            l = fp.readline()
            while len(l) > 0:
                lc = l.strip().split(',')
                cfg_wt = {}
                for w,cfg in zip(lc[1:1+len(cfg_keys)], cfg_keys):
                    cfg_wt[cfg] = float(w)
                mkt_cfg_wt[lc[0]] = copy.deepcopy(cfg_wt)
                l = fp.readline()
        return mkt_cfg_wt

    @staticmethod
    def match_position(idbo_obj, logger=None):
        match = True
        for mkt in idbo_obj.state.keys():
            try :
                idbo_pos = idbo_obj.state[mkt]['pnl']['daily_pos'][-1][1] 
            except:
                print(mkt, ' no daily_pos found, skipping position matching')
                continue

                #import pdb
                #pdb.set_trace()

            for strat in idbo_obj.strat_codes:
                try :
                    pos = strat_utils.get_mts_position(strat, mkt, logger=logger)
                    assert pos is not None
                except Exception as e:
                    if logger is not None:
                        logger.logError('failed to get position for %s %s'%(mkt, strat))
                    print('%s IDBO_ENS: failed to get position for %s %s'%(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), mkt, strat))
                    continue
                if pos != idbo_pos:
                    if logger is not None:
                        logger.logInfo('position mismatch for %s, %s: (idbo:%d, mts:%d)'%(strat, mkt, idbo_pos, pos))
                    print('%s IDBO_ENS: position mismatch for %s, %s: (idbo:%d, mts:%d)'%(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), strat, mkt, idbo_pos, pos))
                    match = False
        if match:
            if logger is not None:
                logger.logInfo('ALL position Matched!')
            print('%s IDBO_ENS: all positions matched'%(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
        return match

def dump_daily_backtest(trade_day, pos_dict, sw, exec_fname, pnl_fname, position_fname,  fee=2.0):
    """
    trade_day: yyyymmdd, the day of the daily position
    pos_dict: daily position dict, i.e., from state['mkt']['pnl']['daily_pos'], 
              format: 
                  {mkt: {'daily_pos', 'tcost' (in price), 'contract_size', 'contract_month', 'tick_size', 'fx', 'symbol'(i.e. mkt_N1), 'bar_unadj'}}
                  where daily_pos:  list of tuple from state['mkt']['pnl']['daily_pos']
                                    (utc, tgt_qty, trd_cfg_name, trd_qty, 
                                     cur_bar_h, cur_bar_l, cur_bar_c, 
                                     sl_px, sl_h, sl_l, sl_af)
                  bar_unadj: the daily bar shape [ndays, n, nc], where ndays>=1, nc have ohlc as col 1,2,3,4
                             Note we assume that the current day's bar is always unadjusted.  The  Subsequent
                             operation should adjust all necessary states if next day is a new contract,
                             but not included here.
    sw: the strategy weight used for the position, used to remove the scale in backoffice files

    Note, it adjusts pos_dict's daily position's ending position to have a bar_h/l/c to be the
          last bar of the day.  This is to reflect the pnl in a daily basis so they can
          be added over time.

    exec file: 2021-10-25 14:30:00, 2021-10-25 00:00:00, WTI_N1, 0, -33, 33, 83.64
    pnl file: 2021-10-25, 1000.00
    position file: 2021-10-22, 0
    """
    # write pnl and position files
    # * generate a per market list of pnl and pos
    pos_key = [] # mkt as key
    pos = [] # ending position
    pnl_raw = [] # ending pnl without tcost
    pnl_net = [] # ending pnl with tcost
    pnl_raw_twap = [] # ending twap pnl without tcost
    pnl_net_twap = [] # ending twap pnl without tcost

    all_utc = []
    all_pos = [] # (symbol, tgt, trd, lpx)

    for mkt in pos_dict.keys():
        pos_key.append(mkt)
        pd = pos_dict[mkt]
        tcost = pd['tcost']
        contract_size = pd['contract_size']
        contract_month = pd['contract_month']
        pos0=[]  # positions in daily_pos
        lpx=[]   # entering px
        lpx_twap=[] # avg px for twap using bar time from bar_unadj
        if len(pd['daily_pos']) == 0:
            # not initailized
            pos0=[0]
            lpx=[0]
        else:
            # append the last roll of daily_pos in case of non-zero position
            utc0, tgt_pos, trd_cfg,trd_qty, bh,bl,bc = pd['daily_pos'][-1][:7]
            bar_unadj = pd['bar_unadj']
            if np.abs(tgt_pos) > 1e-12:
                # add a new line with bar_h/l/c as the bar clse
                # NOTE - this is highly dependent on the 'daily_pos' 
                # structure, assuming the first 7 columns to be
                # utc0, tgt_pos, trd_cfg,trd_qty, bh,bl,bc
                pd['daily_pos'].append(copy.deepcopy(pd['daily_pos'][-1]))
                row0 = list(pd['daily_pos'][-1])
                # update the time to be end time of trade day, trade_qty to be 0
                row0[0] = int(datetime.datetime.strptime(trade_day,'%Y%m%d').strftime('%s'))+17*3600
                row0[3] = 0 # make the trd_qty to be 0
                # bar_unadj could be None if holiday or missing days
                # repeating the last line won't change the pnl in this case
                if bar_unadj is not None:
                    u0, o, h,l,c = bar_unadj[-1,-1,:5]  # last bar
                    for c0, v in zip([4,5,6],[h,l,c]):
                        row0[c0] = v
                pd['daily_pos'][-1] = tuple(row0)
                # this last line is subject to roll adjust later in sod()

            for i, pp0 in enumerate(pd['daily_pos']):
                utc0, tgt_pos, trd_cfg,trd_qty, bh,bl,bc = pp0[:7]
                if np.abs(sw) < 1e-3:
                    sw = 1.0
                pos0.append(tgt_pos/sw)  # unscaled
                lpx.append(bc)
                # figuring out the twap price by
                # taking avg of this and next bar's close px
                # note the first position is copied from previous
                # day's last position with prices adjusted
                lpx_tw = bc
                if i>0 and bar_unadj is not None:
                    butc = bar_unadj[-1,:,0]
                    bix = np.clip(np.searchsorted(butc,utc0)+1, 0, len(butc)-1)
                    u_, o_, h_,l_,c_ = bar_unadj[-1,bix,:5]
                    if u_>utc0:
                        lpx_tw = (bc+c_)/2
                    if np.isnan(lpx_tw):
                        lpx_tw = bc
                lpx_twap.append(lpx_tw)
                if i >= 1:
                    # don't write the first (initial position) line to execution file
                    all_utc.append(utc0)
                    all_pos.append((pd['symbol'], tgt_pos, trd_qty, bc, contract_month))

        pos.append(pos0[-1])  # daily ending position
        pnl0 = [0]
        pnl1 = [0]
        pnl0_twap = [0]
        pnl1_twap = [0]
        # now the pos0 either has a single line of 0 position
        # or more than 1 line of positions
        if len(pos0) > 1:
            # the instant trading cost
            p00 = pos0[0]
            p0 = pos0[1:]
            lr0 = np.log(lpx[1]/lpx[0])
            if np.isnan(lr0):
                lr0 = 0
            pnl1, pnl0 = strat_utils.pnl_from_p0(p00, p0, lpx[1:], tcost=tcost, fee=fee, contract_size=contract_size, lr0=lr0)

            # the twap trading cost
            lr0_twap = np.log(lpx_twap[1]/lpx_twap[0])
            if np.isnan(lr0_twap):
                lr0_twap=0
            pnl1_twap, pnl0_twap = strat_utils.pnl_from_p0(p00, p0, lpx_twap[1:], tcost=tcost, fee=fee, contract_size=contract_size, lr0=lr0_twap)
            #debug 
            #if np.abs(np.sum(pnl0)) > 1000000 :
            #    import pdb
            #    pdb.set_trace()

            if np.isnan(pd['fx']):
                pd['fx'] = 1.0
            pnl0 *= pd['fx']
            pnl1 *= pd['fx']
            pnl0_twap *= pd['fx']
            pnl1_twap *= pd['fx']
        # get a sum of pnl in
        pnl_raw.append(np.sum(pnl1))
        pnl_net.append(np.sum(pnl0))
        pnl_raw_twap.append(np.sum(pnl1_twap))
        pnl_net_twap.append(np.sum(pnl0_twap))

    if len(pos) == 0:
        print('nothing to be updated!')
        return
    ix = np.argsort(pos_key)
    pos_key=np.array(pos_key)[ix]
    pos = np.array(pos)[ix]
    pnl = copy.deepcopy(np.array(pnl_net_twap)[ix])

    pnls_raw = np.sum(pnl_raw)
    pnls_net = np.sum(pnl_net)
    pnls_raw_twap = np.sum(pnl_raw_twap)
    pnls_net_twap = np.sum(pnl_net_twap)

    fd_pnl = open(pnl_fname, 'at+')
    trade_day_fmt = '%s-%s-%s'%(trade_day[:4],trade_day[4:6],trade_day[6:8])
    fd_pnl.write('%s,%.2f,%.2f,%.2f,%.2f'%(trade_day_fmt, \
                     pnls_raw, pnls_raw_twap, pnls_net, pnls_net_twap))
    fd_pos = open(position_fname, 'at+')
    fd_pos.write('%s'%(trade_day_fmt))

    for k, pos0, pnl0 in zip(pos_key, pos, pnl):
        fd_pos.write(',%f'%(pos0))
        # we don't write the per-market pnl
        #fd_pnl.write(',  %f'%(pnl0))

    for fp in [fd_pnl, fd_pos]:
        fp.write('\n')
        fp.close()

    # write exec file
    #all_utc = []
    #all_pos = [] # (symbol, tgt, trd, lpx, contract_month)
    if len(all_utc) > 0 :
        ix = np.argsort(all_utc, kind='stable')
        all_utc = np.array(all_utc)[ix]
        all_pos = np.array(all_pos)[ix]

        ex = []
        for utc0, pos0 in zip(all_utc, all_pos):
            symbol = pos0[0]
            tgt0, trd0, lpx0, contract_month = pos0[1:5].astype(float)
            ex_arr = []
            dt0 = datetime.datetime.fromtimestamp(utc0)
            ex_arr.append(dt0.strftime('%Y-%m-%d %H:%M:%S'))
            ex_arr.append(trade_day + ' 00:00:00')
            ex_arr.append(symbol)
            ex_arr.append('%d'%(int(contract_month)))
            ex_arr.append('%d'%( 0 if tgt0==0 else int(np.sign(trd0)))) # 1/-1: buy/sell, 0:close position
            ex_arr.append('%f'%(tgt0))  # tgt
            ex_arr.append('%f'%(trd0))  # trd
            ex_arr.append('%f'%(lpx0))  # lpx
            ex.append(ex_arr)
        with open(exec_fname, 'at+') as fd:
            np.savetxt(fd, np.array(ex), fmt='%s', delimiter=',')

def plot_pos_file(fn, ax, lpx=None, lpx_utc=None, ax_lpx=None, fn_shift_time = 0, ax_pnl=None, contract_size=1000, tcost=0.01, start_day = None):
    pos = np.genfromtxt(fn, delimiter=',', dtype='str',skip_header=1)

    utc0 = 0
    if start_day is not None:
        utc0 = int(datetime.datetime.strptime(start_day,'%Y%m%d').strftime('%s')) - 6*3600

    dt = []
    utc = []
    for t in pos[:,0]:
        dt0 = datetime.datetime.strptime(t, '%d-%b-%Y %H:%M:%S') 
        dt0 = datetime.datetime.fromtimestamp(int(dt0.strftime('%s')) + fn_shift_time)
        dt.append(dt0)
        utc.append(int(dt0.strftime('%s')))

    dt = np.array(dt)
    utc=np.array(utc)
    pos0 = pos[:,1].astype(float)

    # apply start_day
    ix = np.nonzero(utc>=utc0)[0]
    dt=dt[ix]
    utc=utc[ix]
    pos0=pos0[ix]

    # plot position using a double edge trigger
    pos0_ = np.vstack((pos0,pos0)).T.flatten()[:-1]
    dt_ = np.r_[np.vstack((dt[:-1],dt[1:])).T.flatten(),dt[-1]]
    ax.plot(dt_, pos0_, '.-', label=fn.split('/')[-1])

    dtx = []
    utcx = []
    if lpx is not None:
        for t in lpx_utc:
            dt0=datetime.datetime.fromtimestamp(t)
            dtx.append(dt0)
            utcx.append(int(dt0.strftime('%s')))
        dtx=np.array(dtx)
        utcx=np.array(utcx).astype(int)

        #ix = np.nonzero(utcx>=utc0)[0]
        #dtx=dtx[ix]
        #utcx=utcx[ix]
        #lpx=lpx[ix]
        if ax_lpx is not None:
            ax_lpx.plot(dtx, lpx)

    if ax_pnl is not None:
        lpx_ix = np.clip(np.searchsorted(utcx, utc+0.1)-1,0,len(utcx)-1)
        lpx_pos = lpx[lpx_ix]
        _, pnl_cst = strat_utils.pnl_from_p0(0, pos0, lpx_pos, tcost=tcost, fee = 2.0, contract_size=contract_size, lr0=0)
        ax_pnl.plot(dt, np.cumsum(pnl_cst), '.-', label='pnl_'+fn.split('/')[-1])


def gen_data(init_day_start, init_day_end, md_dict_in=None):
    ida = IDBO_Data()
    if md_dict_in is not None:
        md_dict = md_dict_in
    else :
        mkt_list=[]
        for l in ida.mkt_hours:
            mkt_list.append(l[0])
        md_dict = ida.get_td_md_dict(init_day_start, init_day_end, mkt_list=mkt_list)
    ida.init('','',md_dict_in=md_dict)
    #param = ida.gen()
    return ida

def get_md_bar_days(symbol, start_day, end_day, barsec=300):
    """
    this is IDBO way of getting market data.  Here we resample from 1S bar to 300 in order to:
    1. use lpx as the close price, (just to match the simulation)
    2. and therefore resample the ohl from the 1S bar

    """
    #repo_live = mts_repo.MTS_REPO_Live()
    repo_live = mts_repo.MTS_REPO_TickData()
    bars, roll = repo_live.get_bars(symbol, start_day, end_day, barsec=barsec, cols=['utc','open','high','low','close','lpx'], get_roll_adj=True, is_mts_symbol=True)
    bar0 = bars.copy()
    bara = mts_repo.MTS_REPO.roll_adj(bars, 0, [1,2,3,4,5], roll)
    ndays, n, nc = bara.shape
    bara = bara[:,:,np.array([0,1,2,3,5]).astype(int)]
    bara[:,:,1] = np.r_[bara[0,0,1], bara[:,:,4].flatten()[:-1]].reshape((ndays,n))  # this is not adjusted for the half day
    return bara, roll, bar0

def dump_md(bara, roll, bar0, fn_dill=None, roll_csv_fn=None, bara_csv_fn=None, bar0_csv_fn=None):
    """save bara, roll, bar0 from get_md_bar_days() for purpose of md checking
       fn_dill: dict of everything in dill format
       also save csv for roll, bar0 and bara if needed

       bara, roll, bar0: returned from get_md_bar_days(), as adjusted bar, roll dict and unadjusted bar
       roll is a dict of ['days', 'contracts', 'roll_adj'] as returned from get_bars of mts_repo
    """
    if fn_dill is not None:
        with open(fn, 'wb') as fp:
            fp.dump({'bar_unadj':bar0, 'roll':roll, 'bar_adj':bara}, fp)

    if roll_csv_fn is not None:
        rc = []
        for d, c, ra in zip(roll['days'], roll['contracts'], roll['roll_adj']):
            rc.append([d, c, ra])
        np.savetxt(roll_csv_fn, np.array(rc), delimiter=',', fmt='%s')

    for fn, b in zip([bara_csv_fn, bar0_csv_fn], [bara, bar0]):
        if fn is not None:
            dts = []
            for t in b[:,:,0].flatten():
                dts.append(datetime.datetime.fromtimestamp(t).strftime('%d-%b-%Y %H:%M:%S'))
            b = b.astype('str')
            bd = np.vstack((np.array(dts), b[:,:,1].flatten(), b[:,:,2].flatten(), b[:,:,3].flatten(), b[:,:,4].flatten(), b[:,:,0].flatten())).T
            np.savetxt(fn, bd, delimiter=',', fmt='%s')


def md_dict_from_bar5(holiroll_dict, path, fn='bar5_*', contract='_N1', barsec=300, md_dict_in = {}, start_day0 = '20100101', symbol_list=None, hours = (-6,0,17,0)):
    """
    load all given bar5_mtssymbol.txt files, which is already roll adjusted, and 
    includes the half day in case 'holiday', so the open price reflect the half day's
    last price.

    holiroll_dict: holiday and roll dict, see mts_repo
    hours: a uniform enforced start/stop time for each day, therefore a 'n' for all days
           Brent may use a (-6,0,18,0)
    """
    md_dict = copy.deepcopy(md_dict_in)
    import glob
    fna = glob.glob(os.path.join(path, fn))
    print('got ', len(fna), ' bar5_* files')

    for f0 in fna:
        mkt = f0.split('/')[-1].split('_')[-1].split('.')[0]
        symbol = mkt+contract
        if symbol_list is not None and symbol not in symbol_list:
            continue
        if symbol in md_dict.keys():
            continue
        print('getting ', symbol, ' from ', f0)
        b = np.genfromtxt(f0, delimiter=',', dtype='str', skip_header=1)

        # getting the start-end day
        start_day = max(datetime.datetime.strptime(b[0,0], '%d-%b-%Y %H:%M:%S').strftime('%Y%m%d'), start_day0)
        end_day = datetime.datetime.strptime(b[-1,0], '%d-%b-%Y %H:%M:%S').strftime('%Y%m%d')

        # getting the utc-ref
        utc_ref = mts_repo.get_daily_utc(symbol, barsec, start_day, end_day, holiroll_dict, hours=hours)
        ndays, n = utc_ref.shape

        # using the existing md
        #bar = load_md(f0, utc_ref=utc_ref.flatten(), b=b)
        #nndays,nc = bar.shape
        #ndays0 = nndays//n
        #assert ndays0*n == nndays
        #md_dict[symbol] = bar.reshape((ndays0,n,nc))

        # using the new way
        bar = load_md2(f0, utc_ref)
        md_dict[symbol] = bar

    return md_dict

def load_md(fn, utc_ref=None, ohlc_col=[2,3,4,5], b=None):
    """ load a dumped md csv file from Ji's 5m bar file
    time, contract, o, h, l, c, ...
    if utc_ref is given, the bar is normalized using the utc, with forward filling
    """
    if b is None:
        b = np.genfromtxt(fn, delimiter=',', dtype='str', skip_header=1)
    utc = []
    for t in b[:,0]:
        utc.append(int(datetime.datetime.strptime(t, '%d-%b-%Y %H:%M:%S').strftime('%s')))
    utc=np.array(utc)
    bar = np.vstack((utc, b[:,np.array(ohlc_col).astype(int)].astype(float).T)).T

    if utc_ref is not None:
        ix = np.clip(np.searchsorted(utc, utc_ref+0.1)-1,0,len(utc)-1)
        # get the open as previous bar's close.
        # NOTE - can't use previous close as this open, due to half days
        # If half days in the bar file, but not in utc_ref, then we should
        # still use last half day's price as first open of next day
        ix_prev = np.clip(ix-1,0,len(utc)-1)
        opx = bar[ix_prev,4] # get the open as previous bar's close in bar file
        bar = np.vstack((utc_ref, bar[ix,1:].T)).T
        bar[1:,1] = opx[1:]  # rewrite the open
    else :
        # replace open with previous close
        bar[1:,1] = bar[:-1,-1]
    return bar

def load_md2(fn, utc_ref, ohlc_col=[2,3,4,5], b=None, barsec=300):
    """ load a dumped md csv file from Ji's 5m bar file
    time, contract, o, h, l, c, ...
    This also fixes the open being the previous open, in particularly,
    in case a half day, the open is previous half day's close

    utc_ref: shape [ndays, n], bar is normalized using the utc with normalize()
    return:
        shape [ndays0, n, 5] bar, ndays0 may be less than ndays
    """
    if b is None:
        b = np.genfromtxt(fn, delimiter=',', dtype='str', skip_header=1)
    utc = []
    for t in b[:,0]:
        utc0 = int(datetime.datetime.strptime(t, '%d-%b-%Y %H:%M:%S').strftime('%s'))
        utc0 = int(np.round(float(utc0)/float(barsec)) * barsec)
        utc.append(utc0)
    utc=np.array(utc)
    bar = np.vstack((utc, b[:,np.array(ohlc_col).astype(int)].astype(float).T)).T

    #get the open to be the previous bar's close
    close_ix = 4
    open_ix = 1
    cpx = bar[:,close_ix]
    bar[1:,open_ix]= cpx[:-1]

    dbar = []
    if utc_ref is not None:
        for ur in utc_ref:
            try :
                dbar.append(mts_repo.normalize_ref_utc(bar, ur, bar_cols = ['utc','open','high','low','close'], verbose=False))
            except:
                import traceback
                traceback.print_exc()
                print('problem getting day %s'%(str(datetime.datetime.fromtimestamp(ur[-1]))))
    return np.array(dbar)

def run_sim_md_bar(symbol, md_bar, idbo_dill_fn=None, idbo_data=None):
    """
    run a single mkt using market data (md_bar, shape [ndays, n, ncol])
    this was used primarily for matching purpose, using the
    given bar5m bar that already is roll adjusted
    """
    ndays, n, ncol = md_bar.shape
    mkt = symbol.split('_')[0]

    if idbo_dill_fn is not None:
        sim = dill.load(open(idbo_dill_fn,'rb'))
        # in this case the sim already done the sod before persistence
        # so no need to do the sod
    else :
        # create new
        if idbo_data is None:
            init_days=70
            assert ndays > init_days, 'ndays not enough for initialize idbo_data'
            md_dict={symbol:md_bar[:init_days,:,:]}
            idbo_data = gen_data('','', md_dict_in=md_dict)
            md_bar=md_bar[init_days:,:,:]
            ndays-=init_days

        sim = IDBO_Live(idbo_data)
        sim._init_state()

        # get the first day 
        md_dict = {symbol:md_bar[:1,:,:]}
        trade_day = datetime.datetime.fromtimestamp(md_bar[0,-1,0]).strftime('%Y%m%d')
        sim.sod(trade_day, md_dict_in=md_dict)
        md_bar=md_bar[1:,:,:]
        ndays-=1

    for nd in np.arange(ndays).astype(int):
        md_dict = {symbol:md_bar[nd:nd+1,:,:]}
        trade_day = datetime.datetime.fromtimestamp(md_bar[nd,-1,0]).strftime('%Y%m%d')
        print('running for day %s'%(trade_day))
        sim.sim_day_mkt(mkt, md_bar[nd,:,:])
        sim.sod(trade_day, md_dict_in=md_dict)

    return sim

###
# The process of creating an idbo objec
# 1. create an adjusted bar in md_dict, using either 
#    given bar5_* (see md_dict_from_bar5())
#    from repo (see mts_repo.get_bars())
# 2. run_sim(), this brings the object to the end of last day
#    with lastday_sod=False
# 3. setup_live(), this prepares the the live_dict and save to 
#    and object using given file name
# 4. subsequent days could be run by sim_live(), which
#    simulates days using sod_live
# 
# Note - Live trading would load the object and run_live()
#        see idbo_MODEL
###

def run_sim(md_dict, init_days = 100, run_days=100, lastday_sod=True, write_backoffice=False, sim_tactical_fn=None):
    """ input:
    init_days: number of days immediately before run_days to be used in init
               this affects the ind.b, shape [nday, n, nc], ndays=init_days
     run_days: number of days to run sim
     lastday_sod: upon the last of each symbol, whether to run the 'sod'
                  set it to False if the object is ready to run live

     runs simulations for all markets in md_dict for all days provided
     this creates idbo object from the initial days and
     calls the idbo_ens.sim() with remaining days in md_dict
     """

    md_dict_init={}
    sday_dict = {}
    for sym in md_dict.keys():
        ndays, n, nc = md_dict[sym].shape
        sday = max(0, ndays-run_days-init_days)
        md_dict_init[sym] = md_dict[sym][sday:sday+init_days,:,:]
        sday_dict[sym]=sday

    idbo_data = gen_data('','', md_dict_in=md_dict_init)
    idbo = IDBO_Live(idbo_data, model_name='Tactical')
    idbo._init_state()

    if sim_tactical_fn is not None:
        idbo.set_tactical(sim_tactical_fn)

    md_dict_run={}
    for sym in md_dict.keys():
        md_dict_run[sym] = md_dict[sym][sday_dict[sym]+init_days:,:,:]
    idbo.sim(md_dict_run, run_live_sod=False, lastday_sod=lastday_sod, write_backoffice=write_backoffice)
    return idbo

def setup_live(idbo, trade_day):
    """setup live object and persist to be used by next sim_live()
       trade_day should be the last day of the run_sim(), 
       on which the sod was not run.  the next sim_live()
       should be run on the next trade day of the trade_day, 
       when the sod would perform the sod() for the last
       day of the run_sim().
    """
    idbo.live_dict = idbo._init_live_dict(trade_day)
    idbo.live_dict['utc'] = idbo.live_dict['utc1']
    idbo._persist(fn_date = trade_day)

    # steps to setup model object and backoffice files
    # create backtest file in backtest_path/model_name
    # where backtest_path is the parameter in "retrieve()"
    # i.e. /home/mts/upload/ZFU_STRATS
    # copy the exsiting Pnl-, Exec-, Pos- files to
    # the backtest_path/model_name with names changed to
    # Pnl-model_name, etc
    # finally 
    #
    # idbo_dict = dill.load(open(persist_fn, 'rb'))
    # idbo = IDBO_Live.retrieve(idbo_dict, strat_codes = [xxx], model_name = mts_model_name)
    # return idbo
    return idbo

def sim_live(idbo, sday, eday, write_backoffice=True, symbol_list = None, run_live_sod=True, lastday_sod=False, persist_sod=False):
    if symbol_list is None:
        symbol_list = []
        for mkt in idbo.state.keys():
            symbol_list.append(mkt+idbo.data.contract)

    repo_live = mts_repo.MTS_REPO_Live()
    tdi = mts_util.TradingDayIterator(sday)
    day=tdi.begin()
    barsec = idbo.data.barsec
    while day <= eday:
        # get the md_dict
        print('running for day %s'%(day))
        md_dict = {}
        for symbol in symbol_list:
            try :
                print('getting %s'%(symbol))
                bars = repo_live.get_bars(symbol, day, day, barsec=barsec, cols=['utc','open','high','low','lpx'], get_roll_adj=False, is_mts_symbol=True)
                md_dict[symbol] = bars
            except:
                traceback.print_exc()

        lastday_sod_day = lastday_sod
        if not run_live_sod:
            # in case running offline sod at the end of day
            # we need to make sure the days that are not lastday_sod
            # do run the sod()
            lastday_sod_day = lastday_sod or (day != eday)

        idbo.sim(md_dict, lastday_sod=lastday_sod_day, run_live_sod=run_live_sod, write_backoffice=write_backoffice, persist_sod=persist_sod)
        day = tdi.next()
    
    if run_live_sod:
        try :
            idbo.live_dict['utc'] = idbo.live_dict['utc1']
        except :
            print('failed to set live_dict for the last day, try setting it manually')



########################
# Load the bar5 md_dict
# run sim upto the 5/24
# setup_live
# run live 
# persist, this is the live object!
#########################
def create_live_object():
    # this is a very hack script, purpose is to provide a reference on how the initial
    # object was created

    # getting the offline roll adjusted 5m bar
    holiroll_dict = dill.load(open('/home/mts/run/config/symbol/holiroll.dill','rb'))
    md_dict = md_dict_from_bar5(holiroll_dict, '/mnt/mts/idbo/bar', fn='bar5_*', contract='_N1', barsec=300, start_day0='20150101')
    idbo = run_sim(md_dict, init_days=100, lastday_sod=False, write_backoffice=True)

    # md_dict last day is 20220524
    setup_live(idbo, '20220524')

    # persist here to save a debug point for live
    # idbo._persist()
    # obj_dict = dill.load(open('/tmp/idbo/idbo_ens_obj.dill20220524','rb'))
    # idbo2 = IDBO_Live.retrieve(obj_dict, '/tmp/idbo', '/tmp/idbo')

    run_live(idbo, '20220525','20220608')
    # above won't persist object, since persist is really in run_live()

def adjust_pos(idbo):
    """
    this is just adjust raw position in the state's position into 
    actual contract.  Used to bring the old object upto new,
    not to be used again.
    """
    for mkt in idbo.state.keys():
        cfg = idbo.state[mkt]['ens'].keys()
        for c in cfg:
            pos=idbo.state[mkt]['ens'][c]['state']['pos']
            pos['qty'] = idbo._get_trade_size(mkt, pos['qty'])

