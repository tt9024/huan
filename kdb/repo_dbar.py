import numpy as np
import l1
import copy
import os
import traceback
import pandas as pd
import datetime

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


###  Be very careful aboout the columns, 
###  ONLY ADD, NEVER DELETE, NOR CHANGE EXISTING ORDER
hist_col=['utc', 'lr', 'vol', 'vbs', 'lrhl', 'vwap', 'ltt', 'lpx']
l1bar_col=['spd', 'bs', 'as', 'qbc', 'qac', 'tbc', 'tsc', 'ism1']
all_col=hist_col+l1bar_col
def make_repo_col() :
    col={}
    for i, c in enumerate(all_col) :
        col[c]=i
    return col
repo_col=make_repo_col()

### Special columns from history
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
    raise ValueError('col ' + colname + ' not found')

def ci(carr, c) :
    for i, c0 in enumerate (carr) :
        if c0 == c :
            return i
    raise ValueError(str(c) + ' not found in ' + col_name(carr))

def ix_by_utc(u0, utc, verbose=True) :
    """
    get an index of utc into the day's existing daily_utc u0
    For example, u0 = self._make_daily_utc(day, bar_sec)
    return: ix0: an index of utc into u0, used for overwirte
            zix: an index into utc, for matching entries
            
    """
    ix0 = np.clip(np.searchsorted(u0, utc), 0, len(u0)-1)
    zix = np.nonzero(u0[ix0] - utc == 0)[0]
    if verbose :
        print 'got %d (out of %d total) bars, mismatch %d'%(len(utc), len(u0), len(utc)-len(zix))
    if len(zix) != len(utc) :
        if verbose :
            print 'missing: ', np.delete(np.arange(len(u1)), ix0[zix])
            print 'not used:', np.delete(np.arange(len(ix0)), zix)
        ix0=ix0[zix]
    return ix0, zix

def sync_lr_lpx(dbar, day, upd_col) :
    if upd_col is None :
        print 'upd_col is none for sync_lr_lpx!'
        return
    sync_lr_by_lpx(dbar,day,upd_col=upd_col)
    sync_lpx_by_lr(dbar,day,upd_col=upd_col)

def sync_lr_by_lpx(dbar, day, upd_col=None) :
    """
    when lpx is updated, the LR is updated accordingly. 
    It is called after the upd_col has been written to the repo
    Note 1 the first LR is the over-night
    LR, and is NOT updated. 
    Note 2 repo will call this when only lpx is updated or overwritten without lr.
    upd_col: if None, always recalculate lr based on lpx of the day, except the first lr
             if not None, then it checks if 'lpx' in upd_col without 'lr'. 
             Only recalculate lr if the condition is true
    """
    if upd_col is not None:
        if 'lpx' not in col_name(upd_col) or 'lr' in col_name(upd_col) :
            return
        else :
            print 'lpx updated but not lr, lr to be recalculated',
    print 'update lr based on lpx!'
    bar, col, bs = dbar.load_day(day)
    if lpxc not in col :
        print 'sync_lr_by_lpx without lpx on ', day, ' ', dbar.symbol, ' col: ', c
        return

    lpx_hist = bar[:, ci(col, col_idx('lpx'))]
    u0 = bar[:, ci(col, col_idx('utc'))]

    # get the overnight lr if exists
    # assuming it was from KDB or IB hist
    if col_idx('lr') in col :
        lr0 = bar[:, ci(col, col_idx('lr'))][0]
    else :
        lr0 = 0
    # don't update the over-night lr
    lgpx=np.log(lpx_hist)
    lr = np.r_[lr0, lgpx[1:]-lgpx[:-1]]
    try :
        #lr = lr.reshape((len(u0), 1))
        col = col_idx(['lr','lpx'])
        dbar.overwrite([np.vstack((lr,lpx_hist)).T], [day], [col], bs)
    except :
        traceback.print_exc()

def sync_lpx_by_lr(dbar, day, upd_col=None) :
    """
    when lr is updated but not lpx, then 
    reconstruct a lpx based on lr. 
    This is useful for overwriting by 
    different contracts
    Note 1 the first lpx is used to reconstruct the lpx
    """
    if upd_col is not None:
        if 'lr' not in col_name(upd_col) or 'lpx' in col_name(upd_col) :
            return
        else :
            print 'lr updated but not lpx, lpx to be recalculated',
    print 'construct lpx based on lr!'
    bar, col, bs = dbar.load_day(day)
    if lrc not in col or lpxc not in col :
        print 'sync_lpx_by_lr on ', day, ' ', dbar.symbol, ' without lrc + lpxc '
        return

    lpx0 = bar[0, ci(col, col_idx('lpx'))]
    if np.abs(lpx0) < 1e-10 :
        raise ValueError('sync_lpx_by_lr on ', day, ' ', dbar.symbol, ' no lpx0 to calculate lpx from lr!')

    lr = bar[:, ci(col, col_idx('lr'))]
    lpx=np.r_[lpx0,np.exp(np.log(lpx0)+np.cumsum(lr[1:]))]

    try :
        #lpx = lpx.reshape((len(lr), 1))
        col = col_idx(['lr','lpx'])
        dbar.overwrite([np.vstack((lr,lpx)).T], [day], [col], bs)
    except :
        traceback.print_exc()

def fwd_bck_fill(d0, v=0, fwd_fill=True, bck_fill=True) :
    """
    backward and forward fill value in d that equals to v
    Such as d = lpx, v = 0 for missing values
    It works only for float values
    NOTE if both fills are true, fwd_fill is always BEFORE
    bck_fill.
    """
    if d0.dtype == np.dtype('int') :
        d = d0.astype(float)
    else :
        d = d0
    if v is not None :
        ix = np.nonzero(d==v)[0]
        d[ix]=np.nan
    df=pd.DataFrame(d)
    if fwd_fill:
        df.fillna(method='ffill',inplace=True)
    if bck_fill:
        df.fillna(method='bfill',inplace=True)
    if d0.dtype == np.dtype('int') :
        d0[:] = d.astype(int)

def nc_repo_path(repo_path) :
    """
    get the repo_path of the back contracts of a future symbol
    repo_path: the path to RepoDailyBar of the front contract.
               i.e. /cygdrive/e/research/kdb/repo
    Return: 
    the path to RepoDailyBar of the next contract (back contract).
               i.e. /cygdrive/e/research/kdb/repo_nc
    """
    s = repo_path.split('/')
    s[-1] += '_nc'
    v=''
    for s0 in s[:-1] :
        v+=s0+'/'
    v+= s[-1]
    return v

class RepoDailyBar :
    @staticmethod
    def make_bootstrap_idx(symbol) :
        venue = l1.venue_by_symbol(symbol)
        tick_size, contract_size = l1.asset_info(symbol)
        start_hour, end_hour = l1.get_start_end_hour(symbol)

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
        
    def __init__(self, symbol, repo_path='/cygdrive/e/research/kdb/repo', bootstrap_idx=None, venue=None, create=False) :
        """
        boostrap_idx: an optional idx to use if the idx file doesn't exist (idx.npz)
        venue:   an optional venue for the symbol, i.e. EBS.  path will append the venue name
        create:  if the idx does not exist, create a new (empty) idx

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
            self.idx = np.load(self.path+'/idx.npz')['idx'].item()
        except Exception as e:
            if create :
                os.system('mkdir -p '+self.path)
                self.idx=RepoDailyBar.make_bootstrap_idx(symbol)
                np.savez_compressed(self.path+'/idx.npz',idx=self.idx)
            else :
                import traceback
                traceback.print_exc()
                raise ValueError('idx.npz not found from ' + self.path)

        self.venue = self.idx['global']['venue']
        #self.sh,self.eh = self.idx['global']['sehour']
        self.sh,self.eh = l1.get_start_end_hour(symbol)

    def update(self, bar_arr, day_arr, col_arr, bar_sec) :
        """
        input
          bar_arr: list of daily bar, increasing in time stamp but may have holes
                   daily bar is a 2-d array shape [totbars, col]
          day_arr: list of trading days of each bar
          col_arr: list of columns on each day. The col_name's idx go with bar's column
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
            rb, col, bs = self.load_day(d)
            if len(rb) == 0 :  # add all the columns
                print ' a new day, add all col: ', c
                if len(b) != totbars :
                    print '!!! len(b) != ', totbars, ' skipped...'
                    continue
                rb = b
                bs = bar_sec
                col = copy.deepcopy(c)
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
            sync_lr_lpx(self, d, upd_col=c)
        print 'Done'

    def overwrite(self, bar_arr, day_arr, col_arr, bar_sec, rowix = None, utcix = None) :
        """
        This writes each of the bar to the repo, overwrite existing columns
        if exist, but leave other columns of existing day unchanged. 
        optional rowix, if set, is a two dimensional array, each for a day
        about the specific rows to be updated.  Used for adding l1 data
        where gaps are common. elements of rowix can be none for all. 
        If rowix is None, utcix can be spceified similarly on per-day basis. 
        utcix is a list of daily utc upon when to apply the column data.
        In this case utcix is converted to rowix. 
        Note in case adding columns to existing bars, the rowix has to
        aggree with the totbars, just as update() above. 
        Refer to update()
        """

        for b,c, d in zip(bar_arr,col_arr, day_arr) :
            if len(b.shape) < 2 :
                raise ValueError('bar array of ' +  d + ' need to have 2D')
            if b.shape[1] != len(c) :
                raise ValueError('bar array of ' + d + ' dim mismatch with col_arr' + str(c))

        totbars = self._get_totalbars(bar_sec)
        if rowix is None :
            if utcix is None :
                rowix = np.tile( np.arange(totbars), (len(bar_arr), 1))
            else :
                rowix = []
                barr = []
                for day, uix, ba in zip(day_arr, utcix, bar_arr) :
                    u0 = self._make_daily_utc(day, bar_sec)
                    ix0, zix = ix_by_utc(u0, uix)
                    rowix.append(ix0)
                    barr.append(ba[zix,:])
                bar_arr = barr

        assert len(bar_arr) == len(rowix), 'len(bar_arr) != len(rowix)'
        for b, d, c, rix in zip(bar_arr, day_arr, col_arr, rowix) :
            if rix is None :
                rix = np.arange(totbars)
            print 'overwrite!  day: ', d, ' bar_cnt: ', len(b), '/', totbars, ' ... ',
            rb, col, bs = self.load_day(d)
            # just write that in
            if len(rb) != 0 :
                print ' loaded ', len(rb), ' bars'
                if bar_sec != bs :
                    print 'barsec mismatch, skipping ', d
                    continue
                if len(rb) != totbars :
                    raise ValueError('repo bar has incorrect row count???')

                for i, c0 in enumerate(c) :
                    print 'column: ', col_name(c0),
                    if c0 in col :
                        if c0 == utcc :
                            print ' cannot overwrite utc timestamp! skipping ...'
                        else :
                            print ' overwriting existing repo '
                            rb[rix, ci(col, c0)] = b[:, i]
                        continue
                    else :
                        print 'a new column! adding to repo '
                        # but in this case, the rowix has to match totbars
                        if len(rix) != totbars :
                            raise ValueError('rix not equal to totbars for adding column ' + str(len(rix)))
                        col.append(c0)
                        rb = np.r_[rb.T, [b[:, i]]].T
            else :
                print ' NO bars found, adding all columns as new! '
                if len(rix) != totbars :
                    raise ValueError('rix not equal to totbars for adding column ' + str(len(rix)))
                col = copy.deepcopy(c)
                rb = b
                bs = bar_sec

            self._dump_day(d, rb, col, bs)
            sync_lr_lpx(self, d, upd_col=c)

    def daily_bar(self, start_day, day_cnt, bar_sec, end_day=None, cols=[utcc,lrc,volc,vbsc,lpxc], group_days = 5) :
        """
        return 3-d array of bars for specified period, with bar period, column, 
        grouped by days (i.e.e daily, weekly, etc)
        start_day: the first trading day to be returned
        day_cnt :  number of trading days to be returned, can be None to use end_day
        bar_sec :  bar period to be returned
        end_day :  last trading day to be returned, can be None to use day_cnt
        cols    :  columns to be returned
        group_days: first dimension of the 3-d array, daily: 1, weekly=5, etc
        """
        if end_day is not None :
            print 'end_day not null, got ',
            ti=l1.TradingDayIterator(start_day)
            day_cnt=0
            day=ti.yyyymmdd()
            while day <= end_day :
                day_cnt+=1
                ti.next()
                day=ti.yyyymmdd()
        else :
            ti=l1.TradingDayIterator(start_day)
            ti.next_n_trade_day(day_cnt-1)
            end_day = ti.yyyymmdd()

        # getting the day count, removing initial and final missing days
        ti = l1.TradingDayIterator(start_day)
        day = ti.yyyymmdd()
        darr = []
        inarr = []
        while day <= end_day :
            darr.append(day)
            if self.has_day(day) :
                inarr.append(True)
            else :
                inarr.append(False)
            ti.next()
            day = ti.yyyymmdd()

        ix = np.nonzero(inarr)[0]
        if len(ix) == 0 :
            raise ValueError('no bars found in repo! %s to %s!'%( start_day, end_day))
        start_day = darr[ix[0]]
        end_day = darr[ix[-1]]
        day_cnt = ix[-1]-ix[0]+1
        if day_cnt / group_days * group_days != day_cnt :
            print '( Warning! group_days ' + str(group_days) + ' not multiple of ' + str(day_cnt) + ' adjustint...)', 
            day_cnt = day_cnt/group_days * group_days
            start_day = darr[ix[-1] - day_cnt +1]
        print day_cnt, ' days from ', start_day, ' to ', end_day

        ti=l1.TradingDayIterator(start_day)
        day=ti.yyyymmdd()
        bar = []
        day_arr=[]
        while day <= end_day :
            print "reading ", day, 
            b, c, bs = self.load_day(day)
            if len(b) == 0 :
                print " missing, filling zeros"
                bar.append(self._fill_daily_bar_col(day,bar_sec,cols))
                day_arr.append(day)
            else :
                bar.append(self._scale(day, b, c, bs, cols, bar_sec))
                day_arr.append(day)
                print " scale bar_sec from ", bs, " to ", bar_sec
            ti.next()
            day=ti.yyyymmdd()

        bar = np.vstack(bar)
        
        # process missing days if any
        for c in [lpxc, lttc] + col_idx(['ism1']) :
            if c in cols :
               self._fill_last(bar[:, ci(cols,c)])

        d1 = day_cnt / group_days
        bar = bar.reshape((d1, bar.shape[0]/d1, bar.shape[1]))
        return bar

    def has_day(self, day) :
        return day in self.idx['daily'].keys()

    def load_day(self, day) :
        """
        read a day from repo, files are stored in the day directory
        each day's index is stored as a key in the repo's symbol 
        index
        return bar, col, bs
        Note: The start/stop time may be different from time to time.
              Especially KDB and IB hist.
              self.sh and self.eh initialized from l1.start_stop_hour()
              Scale to satisfy this
        """
        col = []
        bs = 0
        if self.has_day(day) :
            bs = self.idx['daily'][day]['bar_sec']
            col= list(copy.deepcopy(self.idx['daily'][day]['cols']))
            try :
                bfn = self.path+'/daily/'+day+'/bar.npz'
                bar = np.load(bfn)['bar']
            except KeyboardInterrupt as e :
                print 'Keyboard interrupt!'
                raise e
            except :
                print bfn+' not found but is in the repo index'
                self.remove_day(day, check=False)
                return [], [], 0
        else :
            return [], [], 0

        #assert self._get_totalbars(bs) == len(bar), bfn + ' wrong size: '+str(len(bar)) + ' should  be  ' + str(self._get_totalbars(bs))
        if self._get_totalbars(bs) != len(bar) :
            print bfn + ' wrong size: '+str(len(bar)) + ' should  be  ' + str(self._get_totalbars(bs))
            utc=bar[:, ci(col,utcc)]
            u0 = self._make_daily_utc(day, bs)
            ix0, zix = ix_by_utc(u0, utc, verbose=False)
            bar = bar[zix, :]
            if len(zix) != len(u0) :
                bar0 = np.zeros((len(u0), len(col)))
                bar0[:, ci(col, utcc)] = u0
                bar0[ix0, :] = bar[:, :]
                # fill forward and backward for ltt, lpx, ism1, spd
                for i, c in enumerate(col) :
                    if c in [lttc, lpxc] + col_idx(['ism1', 'spd']) :
                        repo.fwd_bck_fill(bar0[:,i],  v=0)
                bar = bar0
        return bar, col, bs

    def remove_day(self, day, match_barsec=None, check=True) :
        """
        if match_barsec is not None then only remove the day if
        the existing barsec matches with match_barsec
        """
        if check :
            bar, col, bs = self.load_day(day)
            if len(bar) == 0 :
                print day, ' not exist... no need to remove'
                return
            if match_barsec is not None:
                if bs != match_barsec :
                    print 'barsec not matched, day not removed!'
                    return

        print 'repo removing %s on %s'%(self.symbol, day)
        try :
            ret=self.idx['daily'].pop(day)
            if ret is not None :
                np.savez_compressed(self.idxfn, idx=self.idx)
        except :
            print self.symbol, ' on ', day, ': not found in index, NOT REMOVED'

        try :
            os.system('rm -fR ' + self.path+'/daily/'+day)
        except :
            traceback.print_exc()
            print self.symbol, ' on ', day, ': not found in daily file, NOT REMOVED '

    def all_days(self) :
        """
        get all the days in the repo
        """
        return self.idx['daily'].keys()

    def upd_overnight_lr(self, day, lr0) :
        """
        just update the lr[0]
        """
        b, c, bs = self.load_day(day)
        if len(b) > 0 and lrc in c :
            b[0,ci(c,lrc)]=lr0
            self._dump_day(day, b,c,bs)
        else :
            print 'upd_overnight_lr: ', day, ' not exist or lr not in ', c

    def clear_cols(self, day, cols, bar_sec) :
        """
        fill col with zeros on day
        """
        assert utcc not in cols, "cannot clear the utc field"
        b, c, bs = self.load_day(day)
        assert bs == bar_sec, "cannot clear cols because bar second does not match"
        for c0 in cols :
            if c0 in c :
                 b[:,ci(c, c0)]=0
            else :
                 print "col ", c0, " not found in bar of ", day
        self._dump_day(day, b, c, bar_sec)


    def _dump_day(self, day, bar, col, bar_sec) :
        """
        update self.index
        write bar to repo
        """
        assert self._get_totalbars(bar_sec) == len(bar), bfn + ' wrong size: '+str(len(bar)) + ' should be ' + str(self._get_totalbars(bar_sec)) 

        if utcc in col :
            # check on the utc against the daily range
            u0 = self._make_daily_utc(day, bar_sec)
            utc0 = bar[:, ci(col,utcc)]
            self._check_utc(u0, utc0)

        self.idx['daily'][day] = {'bar_sec':bar_sec, 'cols':copy.deepcopy(col)}
        bfn = self.path+'/daily/'+day+'/bar.npz'
        # check for existance of self.path in case the
        # program is not run under correct directory
        try :
            st = os.stat(self.idxfn)
        except :
            print 'running at a wrong directory? ', self.idxfn, ' not found!'
            raise ValueError('file not found')

        try :
            os.system('mkdir -p ' + self.path+'/daily/'+day)
            np.savez_compressed(bfn, bar=bar)
            np.savez_compressed(self.idxfn, idx=self.idx)
            # copy to a backup file once done, for some strange problem in index
            if os.stat(self.idxfn).st_size > 1024 :
                os.system('cp '+self.idxfn+' '+self.idxfn+'_bk')
        except KeyboardInterrupt as e :
            raise e
        except :
            traceback.print_exc()
            print 'problem dumping bar file or idx file on ', day, ' ', self.path

    def _get_totalbars(self, bar_sec) :
        return  (self.eh - self.sh) * (3600 / bar_sec)

    def _make_daily_utc(self, day, bar_sec) :
        totbar = self._get_totalbars(bar_sec)
        u1 = int(l1.TradingDayIterator.local_ymd_to_utc(day))+self.eh*3600
        u0 = u1 - totbar*bar_sec + bar_sec
        return np.arange(u0, u1+bar_sec, bar_sec)
 
    def _check_utc(self, utc_ref, utc_bar) :
        """
        check if the utc_bar is within the range of utc_ref
        and utc_bar is strickly increasing
        """
        assert utc_bar[-1] >= utc_ref[0] and utc_bar[0] <= utc_ref[-1], ' wrong time stamp: utc_ref(%d-%d), utc_bar(%d-%d)'%(utc_ref[0], utc_ref[-1], utc_bar[0], utc_bar[-1])
        d0 = np.nonzero(utc_bar[1:] - utc_bar[:-1] <= 0)[0]
        assert len(d0) == 0 , 'update time stamp not strickly: utc_bar(%d:%d, %d:%d)'%(d0[0], utc_bar[d0[0]], d[0]+1,utc_bar[d0[0]+1])

    def _fill_daily_bar_col(self, day, bar_sec, col_arr) :
        """
        first bar starts at the previous day's start_hour+bar_sec, last bar ends at this day's last bar
        before or equal end_hour
        """
        ca = []
        tb = self._get_totalbars(bar_sec)
        for c in col_arr :
            if c == utcc :
                ca.append(self._make_daily_utc(day, bar_sec))
            elif c in [lttc, lpxc] + col_idx(['ism1']) :
                # fill in nan, process it later
                ca.append(np.array( [np.nan]*tb ))
            else :
                # all other numbers can be zero for missing
                ca.append(np.array( [  0   ]*tb ))
        return np.array(ca).T

    def _fake_scale(self,day,b,c,bs,tgt_cols,tgt_bs) :
        """
        scale the bs down towards tgt_bs, and jam
        everything onto the last bar of the tgt_bs. 
        For example, to sacle volume from 5S to 1S, 
        assume the trade happens at last 1S bar, 
        same for the price movements.  This at least
        conserves information not look ahead, but
        it does skew short term behavior.  This is needed
        for KDB trade 1S bar patched with quote 5S bar
        on Sunday nights and missing days. Should
        really not do that in general

        Procedure:
        for columns with last snapshot: 
            defer the changes to the last bar within original bar,
            except the very first bar, the value is back-filled. 

        snapshot (i.e. lpx)
        utc 0  1  2  3  4  5  6  7  8  9  10  11
        5S                 v1             v2
        1S     v1 v1 v1 v1 v1 v1 v1 v1 v1 v2  v2 
        
        cumsum (i.e. lr, vbs)
        utc 0  1  2  3  4  5  6  7  8  9  10  11
        5S                 v1             v2
        1S     v1 0  0  0  0  0  0  0  0  v2  0 

        bix    0  1  2  3  4  5  6  7  8  9  10

        """
        assert(bs>tgt_bs)
        assert(bs/tgt_bs*tgt_bs==bs)
        utc0 = b[:, ci(c,utcc)]
        utc1 = self._make_daily_utc(day, tgt_bs)
        self._check_utc(utc0, utc1)
        fct=bs/tgt_bs

        # utc0 is larger bar, utc1 is smaller bar
        ix=np.searchsorted(utc1,utc0)
        nb=[]
        for c0 in tgt_cols :
            if c0 == utcc :
                nb.append(utc1)
                continue
            assert c0 in c, 'column ' + col_name(c0) + ' not found in '+ str(col_name(c))
            v0 = b[:, ci(c,c0)]
            if c0 in [lttc, lpxc, vwapc] +  col_idx(['ism1','spd','bs','as']) :
                # needs to get the latest snap
                nb.append(np.r_[np.tile(v0[0],fct-1),np.tile(v0[:-1],(fct,1)).T.flatten(),v0[-1]])
            elif c0 in [lrc, volc, vbsc, lrhlc] + col_idx(['qbc','qac','tbc','tsc']):
                # needs to put it into the last small bar and fill others zero
                v1=np.zeros(len(utc1))
                v1[np.arange(2*fct-1,len(utc1),fct)]=v0[1:]
                v1[0]=v0[0]
                nb.append(v1.copy())
            else :
                raise ValueError('unknow col ' + str(c0))

        return np.array(nb).T

    def _scale(self,day,b,c,bs, tgt_cols, tgt_bs) :
        if tgt_bs < bs :
            print 'scale into a smaller bar than original! ', bs, ' to ', tgt_bs
            return self._fake_scale(day,b,c,bs,tgt_cols,tgt_bs)

        utc0 = b[:, ci(c,utcc)]
        utc1 = self._make_daily_utc(day, tgt_bs)
        self._check_utc(utc0, utc1)

        ix = np.searchsorted(utc0, utc1)
        assert len(np.nonzero(utc0[ix]-utc1 != 0)[0]) == 0, 'problem scaling: utc mismatch on ' + day
        nb = []
        for c0 in tgt_cols :
            assert c0 in c, 'column ' + col_name(c0) + ' not found in '+ str(col_name(c))
            v0 = b[:, ci(c,c0)]
            if c0 in [utcc, lttc, lpxc] +  col_idx(['ism1']) :
                # needs to get the latest snap
                nb.append(v0[ix])
            elif c0 in col_idx(['spd','bs','as']) :
                # needs to get an average
                v1=np.r_[0,np.cumsum(v0)[ix]]
                nb.append((v1[1:]-v1[:-1])/(tgt_bs/bs))
            elif c0 in [lrc, volc, vbsc, lrhlc] + col_idx(['qbc','qac','tbc','tsc']):
                # needs aggregate
                v1=np.r_[0,np.cumsum(v0)[ix]]
                nb.append(v1[1:]-v1[:-1])
            elif c0 in[vwapc] :
                # needs aggregate in abs
                # WHY???
                v1=np.r_[0,np.cumsum(np.abs(v0))[ix]]
                nb.append(v1[1:]-v1[:-1])
                """
                # this is handled at daily_bar to get multi-day fillna
                # for missing days
                # intra-day nan has been handled by IB_bar.py's write_daily_bar
            elif c0 in [lpxc] :
                # needs to get the latest snapshot, but fill
                # in zero at begining and ending
                import pandas as pd
                lpx=v0
                ix0=np.nonzero(lpx==0)[0]
                if len(ix0) > 0 :
                    lpx[ix0]=np.nan
                    df=pd.DataFrame(lpx)
                    df.fillna(method='ffill',inplace=True)
                    df.fillna(method='bfill',inplace=True)
                nb.append(lpx.copy())
                """
            else :
                raise ValueError('unknow col ' + str(c0))

        return np.array(nb).T

    def _delete_rows(self,b,c,ix_arr) :
        """
        remove rows indexed as ix_arr from b. 
        b and c is returned by
        b,c,bs=self.load_day()

        fill-in missing values.  set 0 or fill previous values. 
        Note 1: in case there is no previous tick, then next tick is used to fill in 
        """
        for ix in ix_arr :
            # find ref tick
            ix0=ix
            while ix0 >= 0 and ix0 in ix_arr :
                ix0-=1
            if ix0<0:
                ix0=0
                while ix0 in ix_arr :
                    ix0+=1
            try :
                r0 = b[ix0,:]
            except :
                print 'cannot find a row after deleteion!'
                return []

            # fill in removing row
            for i,c0 in enumerate(c) :
                if c0 == utcc :
                    continue
                if c0 in [lttc, lpxc] +  col_idx(['ism1','spd','bs','as']) :
                    # needs to get the latest snap
                    b[ix,i]=r0[i]
                elif c0 in [lrc, volc, vbsc, lrhlc] + col_idx(['qbc','qac','tbc','tsc']):
                    # needs fill in 0
                    b[ix,i]=0
                elif c0 in[vwapc] :
                    # set it equal to previous lpx is possible
                    if lpxc in c :
                        b[ix,i]=r0[ci(c,lpxc)]
                    else :
                        b[ix,i]=r0[i]
                else :
                    raise ValueError('unknow col ' + str(c0))
        return b

    def _fill_last(self, v) :
        """
        fill nan in v with the previous number, fill the initial nan with
        the subsequent number. 
        Used for ltt and lpx
        """
        import pandas as pd 
        df=pd.DataFrame(v)
        df.fillna(method='ffill',inplace=True)
        df.fillna(method='bfill',inplace=True)



#################################################
# Some utility functions
#################################################
def fix_lr_nan(dbar, day_arr) :
    for d in day_arr :
        b, c, bs = dbar.load_day(d)
        if len(b) == 0 :
            continue
        # filling in the lr
        ixn=np.nonzero(np.isfinite(b[:,ci(c,lrc)])==False)[0]
        if len(ixn) > 0 :
            print 'fixing ', len(ixn), ' nan lr bars'
            b[ixn, ci(c,lrc)]=0
            # recalc everything
            lpx = np.log(b[:,ci(c,lpxc)])
            b[1:,ci(c,lrc)]=lpx[1:]-lpx[:-1]
            dbar.remove_day(d)
            dbar.update([b],[d],[c],bs)

def fix_lpx_from_lr(dbar, day, lpx0=None, dbar_ref=None) :
        b, c, bs = dbar.load_day(day)
        if len(b) == 0 :
            return
        # regenerate lpx based on lr
        lr = b[:, ci(c, lrc)]
        lpx0_ = b[0, ci(c,lpxc)]
        if lpx0 is None :
            b0,c0,_=dbar_ref.load_day(day)
            lpx0=b0[0,ci(c0,lpxc)]
        if min(lpx0_,lpx0)/max(lpx0_,lpx0)<0.5 :
            print 'updating lpx using lr for ', dbar.symbol, ' on ', day
            b[:,ci(c,lpxc)]=np.r_[lpx0, np.exp(np.log(lpx0)+np.cumsum(lr[1:]))]
            dbar.remove_day(day)
            dbar.update([b],[day],[c],bs)

def trd_cmp_func(b,c,b0,c0) :
    try :
        v1 = np.sum(b[:,ci(c,volc)])
    except :
        v1 = 0
    try :
        v2 = np.sum(b0[:,ci(c0,volc)])
    except :
        v2 = 0

    return v1>v2

def lr_cmp_func(b,c,b0,c0) :
    try :
        v1 = np.sum(np.abs(b[:,ci(c,lrc)]))
    except :
        v1 = 0
    try :
        v2 = np.sum(np.abs(b0[:,ci(c0,lrc)]))
    except :
        v2 = 0
    return v1>v2

def UpdateFromRepo(sym_arr, day_arr, repo_path_write, repo_path_read_arr, bar_sec, keep_overnight='onzero', should_upd_func=None) :
    """
    copy all contents from read to write, with target bar_sec.  keep_overnight to save the original lr0
    repo_path_read_arr is a prioritized array for reading non-empty days.

    keep_overnight=['yes'|'onzero'|'no']
       'ues' : always keep original (write_repo)'s overnight lr
       'onzero': take read repo's overnight lr if its not zero, otherwise, keep 
       'no'  : always take read repo's overnight lr

    should_upd_func is a function 
        should_upd_func(b,c,b0,c0) ==> b,c from read repo, b0, c0 from write repo
        return: true/false
    """
    for sym in sym_arr :
        try :
            dbar=RepoDailyBar(sym, repo_path=repo_path_write)
        except :
            dbar=RepoDailyBar(sym, repo_path=repo_path_write, create=True)

        dbar_read=[]
        for path in repo_path_read_arr :
            try :
                dbar_read.append(RepoDailyBar(sym, repo_path=path))
            except :
                print path, ' excluded for ', sym
        for d in day_arr :
            print 'copying ', sym, ' on ', d
            b=[]
            for dr in dbar_read :
                b, c, bs = dr.load_day(d)
                if len(b) > 0 and np.sum(np.abs(b[:,ci(c,lrc)])) > 0 :
                    break
                print d, ' not found from ', dr.path
            if len(b)==0 :
                print d, ' not found from ALL Sources, skipping...'
                continue

            if bs != bar_sec :
                b = dr._scale(d,b,c,bs,c,bar_sec)

            b0, c0, bs0 = dbar.load_day(d)
            if should_upd_func is not None and not should_upd_func(b,c,b0,c0) :
                print 'should_upd_func FALSE from ', dr.path
                continue
            if len(b0) > 0 :
                if keep_overnight in ['yes','onzero'] :
                    lr0 = b0[0,ci(c0,lrc)]
                    lr  = b[0,ci(c,lrc)]
                    if keep_overnight == 'onzero' and lr != 0 :
                        print 'using new lr ', lr, ' replacing ', lr0
                        lr0 = lr
                    else :
                        # keep the overnight lr unchanged 
                        print 'keep over-night lr %f, ignoring %f'%(lr0, lr)
                    b[0,ci(c,lrc)]=lr0
                dbar.remove_day(d)
            else :
                print d, ' not found from Dest ', repo_path_write, ' overnight not adjusted.'
            
            dbar.update([b],[d],[c],bar_sec)
            fix_lr_nan(dbar, [d])

def copy_from_repo(symarr, repo_path_write='./repo', repo_path_read_arr=['./repo_cme'], bar_sec=1, sday='20170601', eday='20171231', keep_overnight='onzero') :
    """
    simply copy days from one repo to another, with overnight lr options.
    """
    tdi=l1.TradingDayIterator(sday)
    d=tdi.yyyymmdd()
    while d <= eday :
        UpdateFromRepo(symarr, [d], repo_path_write, repo_path_read_arr, bar_sec, keep_overnight=keep_overnight)
        tdi.next()
        d=tdi.yyyymmdd()

def remove_outlier_lr(dbar, sday, eday, outlier_mul=500) :
    tdi=l1.TradingDayIterator(sday)
    d=tdi.yyyymmdd()
    while d <= eday :
        b,c,bs=dbar.load_day(d)
        if len(b) > 0 and bs == 1 :
            lr = b[:, ci(c,lrc)]
            vol= b[:,ci(c,volc)]
            lrmax=max(np.std(lr)*outlier_mul,0.0001)
            volm=np.mean(vol)
            ix=np.nonzero(np.abs(lr)>lrmax)[0]
            if len(ix) > 0 :
                ix0=np.nonzero(vol[ix]<volm)[0]
                if len(ix0) > 0 :
                    print 'outlier ', len(ix0), ' ticks!'
                    ix0=ix[ix0]
                    t=b[:,ci(c,utcc)]
                    ix1=[]
                    for ix0_ in ix0 :
                        dt=datetime.datetime.fromtimestamp(t[ix0_])
                        if not l1.is_pre_market_hour(dbar.symbol, dt) :
                            ix1.append(ix0_)
                        else :
                            print 'NOT removing 1 tick (pre_market=True: ', dbar.symbol, ', ', dt

                    dbar._delete_rows(b,c,ix1)
                    # remove lpx and overwrite the day
                    # to be reconstructed from lr
                    if lpxc in c :
                        b=np.delete(b,ci(c,lpxc),axis=1)
                        c.remove(lpxc)
                    dbar.overwrite([b],[d],[c],1)
        tdi.next()
        d=tdi.yyyymmdd()

################################################
#  Some test procedures for verifying repo data#
################################################

def getdt(utcarr) :
    dt = []
    for utc in utcarr :
        dt.append(datetime.datetime.fromtimestamp(utc))
    return dt

def plot_repo(repo_path_arr, symbol_arr, sday, eday, bsarr=None, plotdt=True) :
    """
    For each symbol in symbol_arr :
        create a new figure
        for each repo in repo_arr :
            plot lr, lpx and vbs
    
    The goal is to visually inspect any problem for the repo data
    """
    import pylab as pl
    if bsarr is None :
        bsarr = [300]*len(repo_path_arr)
    for sym in symbol_arr :
        tstr = '%s from %s to %s'%(sym, sday, eday)
        fig = pl.figure()
        ax1 = fig.add_subplot(3,1,1)
        ax1.set_title(tstr)
        ax2 = fig.add_subplot(3,1,2,sharex=ax1)
        ax3 = fig.add_subplot(3,1,3,sharex=ax1)
        ax1.grid() ; ax2.grid() ; ax3.grid()
        for rp, bs in zip(repo_path_arr, bsarr) :
            rpstr = rp.split('/')[-1]
            try :
                dbar = RepoDailyBar(sym, repo_path=rp)
                bar5m = dbar.daily_bar(sday, 0, bs, end_day=eday, group_days=1); 
                bar5m = np.vstack(bar5m)
                if plotdt :
                    dt = getdt(bar5m[:, 0])
                else :
                    dt = bar5m[:,0]

                lr = bar5m[:, 1]
                vol = bar5m[:,2]
                vbs = bar5m[:, 3]
                lpx = bar5m[:, 4]
             
                ax1.plot(dt, lpx, label=rpstr)
                ax2.plot(dt, np.cumsum(lr), label=rpstr)
                ax3.plot(dt, np.cumsum(vbs), label=rpstr+" vbs")
                ax3.plot(dt, np.cumsum(vol), label=rpstr+" vol")
            except :
                import traceback
                traceback.print_exc()
                print 'problem with ', rp

        pl.gcf().autofmt_xdate()
        pl.legend(loc='best')

def plot_weekly(wbar, axlr, axvbs, label='', dt=None) :
    """
    wbar is in shape of [week, bar, col]
    """

    v=[]
    for c in [1,3] :
        lr=wbar[:,:, c]
        s=np.std(lr, axis=1)
        m=np.mean(lr, axis=1)
        lr=lr.T-m
        lr/=s
        lr=lr.T
        lr0=np.sum(lr,axis=0)/lr.shape[0]
        v.append(lr0.copy())

    if dt is None :
        dt=[]
        for d in wbar[-1, :,0] :
            dt.append(datetime.datetime.fromtimestamp(d))

    axlr.plot(dt, np.cumsum(v[0]), label=label)
    axvbs.plot(dt, np.cumsum(v[1]), label=label)
    axlr.legend() ; axvbs.legend();
