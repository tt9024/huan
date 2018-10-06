import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo

class L1Bar :
    def __init__(self, symbol, bar_file, dbar_repo) :
        """
        A class for reading IB's L1 bars with the columes as
        UTC         bs    bp         ap            as  bv  sv  utc_at_collect   qbc qac bc sc ism_avg
        --------------------------------------------------------------------------------------------------
        1535425169, 5, 2901.5000000, 2901.7500000, 135, 5, 17, 1535425169000056, 1, 2, 1, 2, 2901.5062609
        ...
        Where
        UTC is the bar ending time
        qbc is best bid change count
        qac is best ask change count
        bc  is buy trade counts 
        sc  is sell trade counts
       
        Parser will get from the file in 
        bar/NYM_CL_B1S.csv
        NOTE 1: utc offset:
        From 201805301800 to 201806261700, utc + 1 matches with history
        From 201806261800 to 201808171700, utc + 2 matches with history
        Good afterwards

        NOTE 2:
        Extended columns starts from 20180715-20:39:55, but may have problem
        for first few days
       
        NOTE 3:
        Next contract bar starts from 20180802-18:12:30
        Same as the IB_Hist, separate dbar_repo for the same symbol's next contract,
        i.e. dbar_repo_next_contract for bars of next contract
       
        NOTE 4:
        Be prepared for any data losses and errors!
        zero prices, zero sizes
        
        Repo update rules :
        Read those l1 bar files and update repo with appropriate columns of 1s bars
        1. overwrite the vol and vbs based on bv and sv, whenever exist (use an index)
        2. add columns of bs, as, spd qbc qac tbc tsc ism1, fill-in on missing

        """
        self.symbol = symbol
        self.hours = l1.get_start_end_hour(symbol)
        self.bar_file = bar_file
        self.f = open(bar_file, 'r')
        self.dbar = dbar_repo

        # the time shifting start/stops, see Note 1
        self.utc10 = l1.TradingDayIterator.local_ymd_to_utc('20180530', 18, 0, 0)
        self.utc11 = l1.TradingDayIterator.local_ymd_to_utc('20180626', 17, 0, 0)
        self.utc20 = l1.TradingDayIterator.local_ymd_to_utc('20180626', 18, 0, 0)
        self.utc21 = l1.TradingDayIterator.local_ymd_to_utc('20180817', 17, 0, 0)
        self.bar_sec = 1  # always fixed as 1 second bar for C++ l1 bar writer

    def read(self) :
        """
        read all days from the bar file, and update dbar_repo with basic columns and
        extended columns if available.
        Repo update rules :
        1. overwrite the vol and vbs based on bv and sv, whenever exist (use an index)
        2. add columns of bs, as, spd qbc qac tbc tsc ism1, fill-in on missing
        """

        while True :
            day, utc, bcols, ecols = self._read_day(self.f)
            if day is not None :
                print 'read day ', day, ' ', len(utc), ' bars.', ' has ext:', ecols is not None
                self._udp_repo(day, utc, bcols, ecols)
            else :
                break

    def _udp_repo(self, day, utc, bcols, ecols) :
        """
        update day to the daily bar repo
        overwrite the vol and vbx
        update the rest 8 columns
        """
        ow_cols = repo.col_idx(['vol', 'vbs'])
        ow_arr = bcols[:, :2]
        repo.overwrite([ow_arr], [day], [ow_cols], self.bar_sec, utcix=utc)
        upd_arr = bcols[:, 2:]
        upd_cols = repo.col_idx(['spd', 'bs','as'])
        if ecols is not None:
            upd_arr = np.vstack((upd_arr, ecols))
            upd_cols+=repo.col_idx(['qbc','qac','tbc','tsc','ism1'])

        # need to fill-in zeros on missing numbers
        utc0 = self.dbar._make_daily_utc(day, self.bar_sec)
        ix0 = np.searchsorte(utc0, utc)

        upc = np.zeros((len(utc0), len(upd_cols)))
        upc[ix0, :] = upd_arr
        self.repo.update([upc], [day], [upd_cols], self.bar_sec)

    def _adjust_time(self, utc) :
        if utc >= self.utc10 and utc <= self.utc11 :
            return utc+1
        if utc >= self.utc20 and utc <= self.utc21 :
            return utc+2
        return utc

    def _basic_cols(self, cols) :
        """
        cols is in the form of 
        [UTC, bs, bp, ap, as, bv, sv, utc_at_collect, qbc, qac, bc, sc, ism_avg]

        returns the basic columns of a barline :
        ['vol', 'vbs', 'spd', 'bs', 'as']

        Note the line could be invalid, for zero prices.
        if not valid, return None
        """
        # validate
        if abs(cols[1]*cols[2]) > 1e-12 :
            return None
        return [cols[5] + cols[6], col[5]-col[6], col[3]-col[2], col[1], col[4]]

    def _ext_cols(self, cols) :
        """
        cols is in the form of 
        [UTC, bs, bp, ap, as, bv, sv, utc_at_collect, qbc, qac, bc, sc, ism_avg]

        returns the extended columns of a barline :
        ['qbc', 'qac', 'tbc', 'tsc', 'ism1']

        if the extended columns are not available,  return None
        """
        if len(cols) < 13:
            return None

        return [cols[8], cols[9] cols[10], cols[11], cols[12]]

    def _parse_line(self, bline, parse_ext = True) :
        """
        utc, basic, ext = parse_line(bline)

        read a line in text format into utc, basic ext fields
        utc: the bar time
        basic: the basic fields without utc: ['vol', 'vbs', 'spd', 'bs', 'as']
        ext: the extended fields:            ['qbc', 'qac', 'tbc', 'tsc', 'ism1']
             otherwise None

        Note 1: adjust time
        From 201805301800 to 201806261700, utc + 1 matches with history
        From 201806261800 to 201808171700, utc + 2 matches with history
        Good afterwards

        Note 2: check for errors in the numbers

        Note 3: The ext columns may not be available
        """
        cols=bline.replace('\n','').split(',')
        utc = self._adjust_time(int(cols[0]))
        cols = np.array(cols).astype(float)
        basic = self._basic_cols(cols)
        ext = None
        if parse_ext and basic is not None :
            ext = self._ext_cols(cols)
        return utc, basic, ext

    def _read_day(self, bfp) :
        """
        day, utc, bcols, ecols = read_next_day(self, bfp)

        read a day worth of bars in, normalize to either basic or ext format
        bfp: the file descriptor for read on the next lines.  
        day: the day obtained
        utc: an array of time stamp (in second) for each bar
        bcols and ecols: 2D arraies of 5 columns (basic) and 5 columns (ext)
        
        Upon return the file descriptor can be read for the next day
        Note, the rule for whether a day has basic or ext is the following:
        1. day before 7/17 don't have ext
        2. 13 columns for all lines in the day
        """

        day = None
        tcol = []
        bcols = []
        ecols = []
        parse_ext = True
        while True :
            l = bfp.readline()
            if len(l) > 20 : # some minimum size
                utc, basic, ext = self._parse_line(l, parse_ext=parse_ext)
                d0 = l1.TradingDayIterator.utc_to_local_trading_day(utc)
                if day is None :
                    day = d0
                elif day != d0 :
                    bfp.seek(-len(l), 1)
                    break
                bcols.append(basic)
                if ext is not None :
                    ecols.append(ext)
                else :
                    parse_ext = False
                tcol.append(utc)
            else :
                break
        bcols = np.array(bcols)
        ecols = np.array(ecols) if parse_ext else None
        return day, np.array(tcol), bcols, ecols

def read_l1(bar_file) :
    b = np.genfromtxt(bar_path, delimiter=',', use_cols=[0,1,2,3,4,5,6])
    # I need to get the row idx for each day for the columes of vbs and ism
    # which one is better?
    # I could use hist's trade for model, and l1/tick for execution
    pass

