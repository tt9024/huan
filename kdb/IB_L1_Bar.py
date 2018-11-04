import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo
import pandas as pd
import copy

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

        Based on a line in the bar file, the parsing returns the following two arrays
        bcol_arr: array of basic columns for each day.  
                 ['vol', 'vbs', 'spd', 'bs', 'as', 'mid']
        ecol_arr: array of extended columns for each day
                 ['qbc', 'qac', 'tbc', 'tsc', 'ism1']

        if dbar_repo is not None, it will update repo by the following rule:
        1. overwrite the [lrc,volc,vbsc,lpxc], whenever exist (indexing using the utcc)
        2. add columns of bs, as, spd qbc qac tbc tsc ism1, fill-in on missing
           (see NOTE 5)


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
        

        Note 5:
        There are 1~2 second drift on the hist's mid and L1's mid before 8/18/2018.
        Since the L1 is the live trading one, it is given more emphasis. 
        To be consistent, the lr also is overwritten together with vol and vbs.

        But when constructing lr to override, due to the first lr being
        calculated with previous trading day on the same contract, 
        BE SURE to use the hist data on the first index

        Weekend ingestion process for front/back future contract:
        1. collect and ingest hist file, handling missings
        2. read and ingest bar files
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

        return :
        day_arr, utc_arr, bcol_arr, ecol_arr.
        day_arr: array of days
        utc_arr: array of utc for each day
        bcol_arr: array of basic columns for each day.  
                 ['vol', 'vbs', 'spd', 'bs', 'as', 'mid']
        ecol_arr: array of extended columns for each day
                 ['qbc', 'qac', 'tbc', 'tsc', 'ism1']
        """

        darr = []
        uarr = []
        barr = []
        earr = []
        with open(self.bar_file, 'r') as f:
            while True :
                day, utc, bcols, ecols = self._read_day(f)
                if day is not None :
                    print 'read day ', day, ' ', len(utc), ' bars.', ' has ext:', ecols is not None
                    if self.dbar is not None:
                        self._upd_repo(day, utc, bcols, ecols)
                    darr.append(day)
                    uarr.append(utc)
                    barr.append(bcols)
                    earr.append(ecols)
                else :
                    break

        return darr, np.array(uarr), np.array(barr), np.array(earr)

    def _best_shift_multi_seg(self, lpx_hist0, lpx_mid0, startix=0, endix=-1, verbose = True) :
        """
        This is similar (and uses) _best_shift(), but it tries to detect
        if the shift changes during a day, which (I don't know why) happens
        during 7/24/2018.  
        This needs to be robust, to allow noisy but it uses the hist data
        as a reference against the live data.  

        It tries to find/merge a shift segment iteratively until converge
        """

        totcnt = len(lpx_hist0)
        if endix == -1 :
            endix = totcnt
        lpx_hist = lpx_hist0[startix:endix]
        lpx_mid  = lpx_mid0[startix:endix]
        thiscnt = len(lpx_hist)

        if verbose :
            print 'checking %d - %d'%(startix, endix)
        sh0 = self._best_shift(lpx_hist, lpx_mid, order=0, verbose=verbose)
        MINSAMPLE = 300
        if thiscnt < MINSAMPLE :
            return [[startix, endix]], [sh0]

        # divide equally and find shift for both segments
        cnt = thiscnt/2
        sh1 = self._best_shift(lpx_hist[:cnt], lpx_mid[:cnt], order=0, select_margin=0.5, verbose=verbose)
        sh2 = self._best_shift(lpx_hist[cnt:], lpx_mid[cnt:], order=0, select_margin=0.5, verbose=verbose)
        if verbose :
            print '<<< ', startix, ', ', startix+cnt, ', ', sh1
            print '>>> ', startix+cnt, ', ', endix, ', ', sh2

        if sh1 == sh2 :
            if sh1 != sh0 :
                if cnt < 2*MINSAMPLE :
                    if verbose :
                        print 'segment shift disagree with total shift! go with total shift ' + str(sh0) + ' ' + str(sh1) + ' ' + str(cnt)
                    sh1=sh0
                else :
                    if verbose :
                        print 'segment shift disagree with total shift! go with segment shift ' + str(sh0) + ' ' + str(sh1) + ' ' +str(cnt)
            if verbose :
                print 'CONFIRMED! ', sh1
            return [[startix, endix]], [sh1]
        
        ixarr = []
        sharr = []
        ix1, sh1 = self._best_shift_multi_seg(lpx_hist0, lpx_mid0, startix, startix+cnt, verbose=verbose)
        ix2, sh2 = self._best_shift_multi_seg(lpx_hist0, lpx_mid0, startix+cnt, endix,   verbose=verbose)
        if verbose :
            print '!!! ', ix1+ix2, sh1+sh2
        return ix1+ix2, sh1+sh2

    def _best_shift(self, lpx_hist, lpx_mid, order=0.5, select_margin=None, verbose=False) :
        # lpx_hist is the lpx from IB_hist, with correct clock shift
        # lpx_mid is the lpx after the overwritten by Bar_L1's mid, with clock skew
        # returns the best shift base on the MSE
        # order is the exponent to the absolute of difference, default squre root to
        #       put more focus on consistent difference while avoid noise during breakout
        #       set order = 0 to count exact matching points
        # if select_margin is not None, True, then the best has to be this much better than the second
        #       by the fraction of the value or if both are very small
        # NOTE: this shift matches the lpx_mid's shift, when adjusting, 
        # it should be componsated, i.e. use negative of shift to apply to utc
        assert len(lpx_hist) == len(lpx_mid), 'lpx_hist and lpx_mid length mismatch'
        sharr=np.array([-2, -1,0,1, 2])
        stsh = 5
        ixnz=np.nonzero(np.abs(lpx_hist[1:]-lpx_hist[:-1])>1e-10)[0]
        nzc=len(ixnz)
        if nzc < 2*stsh+10 :
            return 0

        mse = []
        y = lpx_hist[stsh:-stsh]
        for shift in sharr :
            x = lpx_mid[stsh + shift : -stsh + shift]
            absdiff = np.abs(y-x)
            if order == 0 :
                # get the count
                mct = np.nonzero(absdiff > 1e-10)[0]
                #v = float(len(mct))/float(len(absdiff))
                v = float(len(mct))/float(nzc)
            else :
                v=np.mean(absdiff**order)
            mse.append(v)
        ix = np.argsort(mse)
        if verbose :
            print 'got mse ', mse, ' ix ', ix, ' shift', sharr[ix[0]]

        # considering the select_margin
        if select_margin is not None:
            v0 = mse[ix[0]]
            v1 = mse[ix[1]]
            if v0 > 1e-10 and (v1-v0)/v0 < select_margin :
                return np.nan

        return sharr[ix[0]]
        
    def _upd_repo(self, day, utc, bcols, ecols) :
        """
        update day to the daily bar repo
        overwrite the vol and vbx, also the lr based lr (see NOTES)
        update the rest 8 columns

        Note 1:
        There are 1~2 second drift on the hist's mid and L1's mid.
        Since the L1 is the live trading one, it is given more emphasis. 
        To be consistent, the lr also is overwritten together with vol and vbs.

        Note 2:
        when constructing lr to override, due to the first lr being
        calculated with previous trading day on the same contract, 
        BE SURE to use the hist data on the first index

        Note 3:
        since there could be missings in the live bar, the matching applies
        to the lpx/mid, the index could change.  Overwrite the mid/lpx/vol/vbs
        and reconstruct the LR based on the result mid
        """
        ow_cols = repo.col_idx(['vol', 'vbs', 'lpx'])
        mid=bcols[:,-1]
        ow_arr = np.vstack((bcols[:, :2].T, mid)).T
        upd_cols = repo.col_idx(['spd','bs','as'])
        upd_arr = bcols[:, 2:5]
        if ecols is not None:
            upd_cols+=repo.col_idx(['qbc','qac','tbc','tsc','ism1'])
            upd_arr = np.vstack((upd_arr.T, ecols.T)).T

        bar, col, bs = self.dbar.load_day(day)
        u0 = bar[:, repo.col_idx('utc')]
        utcix, zix = repo.ix_by_utc(u0, utc, verbose=False)
        utc=utc[zix]
        mid=mid[zix]
        ow_arr=ow_arr[zix,:]
        upd_arr=upd_arr[zix,:]

        # figure out the best time shift (due to collection machine's clock problem) 
        # during 20180530 to 20180818
        utc0 = utc[0]
        if utc0 != self._adjust_time(utc0) :
            ## Save the original lpx history for possible second shift adjustments
            lpx_hist = bar[:, repo.col_idx('lpx')]
            ixa, sha = self._best_shift_multi_seg(lpx_hist[utcix], mid, verbose = False)
            print 'got shift of ', ixa, -np.array(sha),  ', reapply and overwrite!'
            for ix, shift in zip(ixa, sha) :
                if shift != 0 :
                    utc[ix[0]:ix[1]]-=shift

            # need to make sure the utc monotically increase, update ow_arr and upd_arr
            inc_ix = l1.get_inc_idx2(utc, time_inc=True)
            if len(inc_ix) != len(utc) :
                print 'got ', len(inc_ix), ' increasing utc out of ', len(utc)
                utc=utc[inc_ix]
                ow_arr=ow_arr[inc_ix,:]
                upd_arr=upd_arr[inc_ix,:]

            # readjust the utcix and zix
            utcix, zix = repo.ix_by_utc(u0, utc, verbose=False)
            if len(zix) != len(utcix) :
                print 'removing ix after adjusting utc, ',
                print 'some ix moved out of daily utc: len(utc)=%d, len(zix)=%d'%(len(utcix), len(zix))
                utc=utc[zix]
                ow_arr=ow_arr[zix, :]
                upd_arr=upd_arr[zix,:]

            """
            ixbad = np.nonzero( utc[1:]-utc[:-1] < 1)[0]
            if len(ixbad) > 0 :
                print 'removing duplicated ix ', ixbad
                utc=np.delete(utc, ixbad)
                ow_arr = np.delete(ow_arr, ixbad, axis=0)
                upd_arr = np.delete(upd_arr, ixbad, axis=0)
            """
        #self.dbar.overwrite([ow_arr[1:, :]], [day], [ow_cols], self.bar_sec, utcix=[utc[1:]])
        self.dbar.overwrite([ow_arr[1:, :]], [day], [ow_cols], self.bar_sec, rowix=[utcix[1:]])

        # need to fill-in zeros on missing bars of the day
        upc = np.zeros((len(u0), len(upd_cols)))
        upc[utcix, :] = upd_arr
        # 'spd' and 'ism1' needs to fill back/forward
        for i in [0, -1] :
            d = upc[:, i]
            ix = np.nonzero(d==0)[0]
            d[ix]=np.nan
            df=pd.DataFrame(d)
            df.fillna(method='ffill',inplace=True)
            df.fillna(method='bfill',inplace=True)

        self.dbar.overwrite([upc], [day], [upd_cols], self.bar_sec)
        #self.dbar.update([upc], [day], [upd_cols], self.bar_sec)

    def _adjust_time(self, utc) :
        """
        NOTE 1: utc offset:
        From 201805301800 to 201806261700, utc + 1 matches with history
        From 201806261800 to 201808171700, utc + 2 matches with history
        Good afterwards
        """
        if utc >= self.utc10 and utc <= self.utc11 :
            #print 'adding bar utc by 1'
            return utc+1
        if utc >= self.utc20 and utc <= self.utc21 :
            #print 'adding bar utc by 2'
            return utc+2
        return utc

    def _basic_cols(self, cols) :
        """
        cols is in the form of 
        [UTC, bs, bp, ap, as, bv, sv, utc_at_collect, qbc, qac, bc, sc, ism_avg]

        returns the basic columns of a barline :
        ['vol', 'vbs', 'spd', 'bs', 'as', 'mid']

        Note the line could be invalid, for zero prices.
        if not valid, return None
        """
        # validate
        if abs(cols[1]*cols[2]) <= 1e-12 or \
           abs(cols[3]*cols[4]) <= 1e-12 :
            print 'problem with the cols {0}'.format(cols)
            return None
        return [cols[5] + cols[6], cols[5]-cols[6], cols[3]-cols[2], cols[1], cols[4], (cols[2] + cols[3])/2]

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

        return [cols[8], cols[9], cols[10], cols[11], cols[12]]

    def _is_missing(self, utc, ext) :
        try :
            ism = ext[-1]
            cnts = np.sum(ext[:-1])
            ismdiff = np.abs(ism-self.prev_ism)
            if ismdiff < 1e-10 and cnts == 0 :
                eq_dur = utc - self.last_eq
                if eq_dur > 60 : # if it's more than 1 minutes of eq
                    return True
        except :
            pass

        self.last_eq = utc
        self.last_ism = ism
        return False

    def _parse_line(self, bline, parse_ext = True) :
        """
        utc, basic, ext = parse_line(bline)

        read a line in text format into utc, basic ext fields
        utc: the bar time
        basic: the basic fields without utc: ['vol', 'vbs', 'spd', 'bs', 'as', 'mid']
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
                if basic is None : 
                    # parsing error, next
                    continue
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
        tcol = np.array(tcol)

        # this is slightly complicated, as the bar will repeat
        # old price in case of disconnection (i.e. missing)
        # so we shouldn't include those stucked ticks in
        if parse_ext is not None :
            # check for the missing by detecting equals in ism and counts
            last_ism = 0
            last_eq = None
            miss_arr=[]
            missing = False
            marr=[]
            for u0, ext in zip(tcol, ecols) :
                ism = ext[-1]
                cnts = np.sum(ext[:-1])
                if cnts == 0 and np.abs(ism-last_ism) < 1e-10:
                    if last_eq is None :
                        last_eq = u0
                    eq_sec = u0 - last_eq
                    if eq_sec > 30 :
                        # mark missing
                        if not missing :
                            missing = True
                            marr.append(last_eq)
                            print 'missing data detected starting on ', datetime.datetime.fromtimestamp(last_eq)
                else :
                    if missing :
                        missing = False
                        marr.append(u0)
                        miss_arr.append(copy.deepcopy(marr))
                        print 'missing data end at ', datetime.datetime.fromtimestamp(u0)
                        marr = []
                    last_ism = ism
                    last_eq = None

            if missing :
                marr.append(u0+1)
                miss_arr.append(copy.deepcopy(marr))

            # remove indexes in miss_arr
            remove_ix = []
            for [s, e] in miss_arr :
                ix = np.searchsorted(tcol, [s, e])
                remove_ix = np.r_[remove_ix, np.arange(ix[0], ix[1])]

            tcol = np.delete(tcol, remove_ix)
            bcols = np.delete(bcols, remove_ix, axis=0)
            ecols = np.delete(ecols, remove_ix, axis=0)

        return day, tcol, bcols, ecols

def read_l1(bar_file) :
    b = np.genfromtxt(bar_path, delimiter=',', use_cols=[0,1,2,3,4,5,6])
    # I need to get the row idx for each day for the columes of vbs and ism
    # which one is better?
    # I could use hist's trade for model, and l1/tick for execution
    pass

