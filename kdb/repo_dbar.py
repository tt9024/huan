import numpy as np
import l1
import copy

# This is the repository for storing
# daily bars of all assets.  Each asset
# such as CL, lives in a directory, such as
# dbar/CL and maintais a global index file
# idx.npz.  This file stores global attributes
# of that asset as well as for each
# day the attributes of the bars of that day.
# currently the global attributes include:
#     tick size, start/stop utc. (CME and ICE have
#     different start/stop time, which is important)
# Per day attributes include:
#     columes, bar_sec
# When a new hist file downloaded from IB (subject
# to all kinds of errors due to crappiness of their
# API and my code), they are added to the repo 
# day by day with the option:
# * Add:  
#   only write to the repo if that day is not there
#   This is typcially used in normal new history download
# * Update
#   Same as Add, except on an existing day, add columns
#   if not there or rows (utc of a bar) not there. 
#   This is typically used in getting new features in,
#   such as derived from L1/L2.
#   Note the bar_sec and number of rows have to match
#   number of rows equals total TradingHour * 3600 / bar_sec
#   where TradingHour is defined by venue from l1.py
#
# * Overwrite:
#   write column to the repo, overwirte if the columns 
#   exist on that day, but keep the rest of the columns 
#   This is typically used to patch missing days or
#   periods of days
#
# Internally, it stores the daily bars into files
# of a day, with filename of yyyymmdd
# It will retrive and save
# 
# When retrieving, it allows an easy interface to 
# get bars:
# * getDailyBar(sday, eday, cols, barsec)
#   returns a nparray of the bars, with sample time at 
#   barsec, returned as 3-d array of day/bar/col
# * getWeeklyBar(sday, eday, cols, barsec)
#   Same, return as 3-d array of week/bar/col
# 
# plotting: 
# * allow plotting to check availablity
# 

repo_col={'utc':0, 'lr':1, 'vol':2, 'vbs':3, 'lrhl':4, 'vwap':5, 'ltt':6, 'lpx':7}
utcc=repo_col['utc']
lrc=repo_col['lr']
volc=repo_col['vol']
vbsc=repo_col['vbs']
lrhlc=repo_col['lrhl']
vwapc=repo_col['vwap']
lttc=repo_col['ltt']
lpxc=repo_col['lpx']
ALL_COL_RAW=len(repo_col.keys())  # KDB and IB hist columns 
weekday=['mon','tue','wed','thu','fri','sat','sun']
kdb_ib_col = [utcc, lrc, volc, vbsc, lrhlc, vwapc, lttc, lpxc]
def col_name(col) :
    if isinstance(col, list) :
        return [ col_name(c0) for c0 in col ]
    if col < len(repo_col.keys()) :
        return repo_col.keys()[np.nonzero(np.array(repo_col.values())==col)[0][0]]
    raise ValueError('col ' + str(col) + ' not found!')

def col_idx(colname) :
    if isinstance(colname, list) :
        return [ col_idx(c0) for c0 in colname ]
    for k, v in repo_col.items() :
        if colname == k :
            return v
    raise ValueError('col ' + col_name + ' not found')

def ci(carr, c) :
    for i, c0 in enumerate (carr) :
        if c0 == c :
            return i
    raise ValueError(col_name(c) + ' not found in ' + col_name(carr))

class RepoDailyBar :
    @staticmethod
    def make_bootstrap_idx(symbol) :
        venue = l1.venue_by_symbol(symbol)
        tick_size = l1.SymbolTicks[symbol]
        start_hour, end_hour = l1.get_start_end_hour(venue)

        idx = { 'global': \
                       { 'symbol': symbol, \
                         'venue' : venue,  \
                         'sehour': [start_hour, end_hour], \
                         'ticksz': tick_size 
                       },  \
                'daily' : {\
                    #  '19700101' : \
                    #             { 'bar_sec': 1, \
                    #               'cols'   : [] \  # columns available
                    #             }  \
                          }\
              }
        return idx

    def __init__(self, symbol, repo_path='repo', bootstrap_idx=None, venue=None) :
        """
        idx.npz stores global as well as daily configurations
        global:
           start hour
           end hour
           symbol
        daily: key by 'YYYYMMDD'
           bar_sec
           cols
        """
        self.symbol = symbol
        self.path = repo_path+'/'+symbol
        if venue is not None :
            # this is to entertain multiple venue for FX
            # i.e. EUR.USD/Hotspot, etc
            self.path += '/'+venue
        self.idxfn = self.path+'/idx.npz'
        if bootstrap_idx is not None :
            if l1.get_file_size(self.idxfn) > 0 :
                raise ValueError('idx file exists: ' + self.idxfn)
            print 'saving the given bootstrap idx'
            np.savez_compressed(self.idxfn, idx=bootstrap_idx)

        try :
            self.idx = np.load(self.path+'/idx.npz')['idx']
        except :
            raise ValueError('idx.npz not found from ' + self.path)

        self.venue = self.idx['global']['venue']
        self.sh,self.eh = self.idx['global']['sehour']

    def update(self, bar_arr, day_arr, col_arr, bar_sec) :
        """
        input
          bar_arr: list of daily bar, increasing in time stamp but may have holes
                   daily bar is a 2-d array shape [totbars, col]
          day_arr: list of trading days of each bar
          col_arr: list of columns on each day
          bar_sec: the raw bar second from hist/bar files
        return
          None
        Updates the repo with the following rules:
        * if the day doesn't exist, add to repo
        * if the colums of the day doesn't exist, add to repo
        *    NOTE the bar_sec has to match.
        See also overwrite() where the columns and rows are written to to day
        overwriting the existing. 
        """

        totbars = self._get_totalbars(bar_sec)
        for b, d, c in zip(bar_arr, day_arr, col_arr) :
            print 'update bar: ', d, ' bar_cnt: ', len(b), '/',totbars, ' ... ',
            rb, col, bs = self._load_day(d)
            if len(rb) == 0 :  # add all the columns
                print ' a new day, add all col: ', c
                if len(b) != totbars :
                    print '!!! len(b) != ', totbars, ' skipped...'
                    continue
                rb = b
                bs = bar_sec
                c = col
            else :             # add only new columns
                print ' found ', len(rb), ' bars (', bs, 'S), col: ', col_name(col), 
                if bs != bar_sec :
                    print ' !! bar_sec mismatch, skip'
                    continue
                if len(rb) != len(b) :
                    print ' !!! number of bars mismatch, skip'
                    continue
                nc = []
                ncbar = []
                for i, c0 in enumerate(c) :
                    if c0 not in col :
                        col.append(c0)
                        nc.append(c0)
                        ncbar.append( b[:, i] )
                if len(nc) == 0 :
                    print ' no new columns, done '
                    continue
                print ' adding ', col_name(nc)
                rb = np.r_[rb.T, np.array(ncbar)].T

            # writing back to daily
            self._dump_day(d, rb, col, bs)
        print 'Done'

    def overwrite(self, bar_arr, day_arr, col_arr, bar_sec, rowix = None) :
        """
        This writes each of the bar to the repo, overwrite existing columns
        if exist, but leave other columns of existing day unchanged. 
        optional rowix, if set is a two dimensional array, each for a day
        about the specific rows to be updated.  Used for adding l1 data
        where gaps are common
        Note in case adding columns to existing bars, the rowix has to
        aggree with the totbars, just as update() above. 
        Refer to update()
        """
        totbars = self._get_totalbars(bar_sec)
        if rowix is None :
            rowix = np.arange(totbars).tile((len(bar_arr), 1))
        assert len(bar_arr) == len(rowix), 'len(bar_arr) != len(rowix)'
        for b, d, c, rix in zip(bar_arr, day_arr, col_arr, rowix) :
            print 'overwrite bar: ', d, ' bar_cnt: ', len(b), '/', totbars ' ... ',
            rb, col, bs = self.load_day(d)
            # just write that in
            if len(rb) != 0 :
                print len(rb)
                if bar_sec != bs :
                    print 'barsec mismatch, skipping '
                    continue
                if len(rb) != totbars :
                    raise ValueError('repo bar has incorrect row count???')

                for i, c0 in enumerate(c) :
                    print i, ', ', col_name(c0), 
                    if c0 in col :
                        print ' overwriting existing repo '
                        rb[rix, ci(col, c0)] = b[:, i]
                        continue
                    else :
                        print 'adding to repo '
                        # but in this case, the rowix has to match totbars
                        if len(rix) != totbarss :
                            raise ValueError('rix not equal to totbars for adding column ' + str(len(rix)))
                        col.append(c0)
                        rb = np.r_[rb.T, [b[:, i]]].T
            else :
                print ' NO bars found, adding all columns as new! '
                if len(rix) != totbars :
                    raise ValueError('rix not equal to totbars for adding column ' + str(len(rix)))
                col = c
                rb = b

            self._dump_day(d, rb, col, bs)

    def load_day(self, day) :
        """
        read a day from repo, files are stored in the day directory
        each day's index is stored as a key in the repo's symbol 
        index
        """
        col = []
        bs = 0
        if day in self.idx['daily'].keys() :
            bs = self.idx['daily'][day]['bar_sec']
            col= copy.deepcopy(self.idx['daily'][day]['cols'])
            try :
                bfn = self.path+'/daily/'+day+'/bar.npz'
                bar = np.load(bfn)['bar']
            except :
                raise ValueError(bfn+' not found but is in the repo index')

        assert self._get_totalbars(bs) == len(bar), bfn + ' wrong size: '+str(len(bar)) + ' should  be  ' + str(self._get_totalbars(bs)))
        return bar, bol, bs

    def _dump_day(self, day, bar, col, bar_sec) :
        """
        update self.index
        write bar to repo
        """
        assert self._get_totalbars(bar_sec) == len(bar), bfn + ' wrong size: '+str(len(bar)) + ' should be ' + str(self._get_totalbars(bar_sec)) 
        self.idx['daily'][day] = {'bar_sec':bar_sec, 'cols':copy.deepcopy(col)}
        bfn = self.path+'/daily/'+day+'/bar.npz'
        np.savez_compressed(bfn, bar=bar)
        np.savez_compressed(self.idxfn, idx=self.idx)

    def _get_totalbars(self, bar_sec) :
        return  (self.eh - self.sh) * (3600 / bar_sec)

class RepoReader  :
    def __init__(self, repo) :
        """
        repo: an instance of RepoDailyBar
        """
        self.repo = repo

    def get_daily(self, sday, eday  
