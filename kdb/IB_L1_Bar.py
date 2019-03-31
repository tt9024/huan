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
        self.venue = l1.venue_by_symbol(symbol)
        self.hours = l1.get_start_end_hour(symbol)
        self.bar_file = bar_file
        if bar_file[-3:] == '.gz' :
            os.system('gunzip -f ' + bar_file)
            self.bar_file = bar_file[:-3]
            self.gzip = True
        else :
            self.gzip = False
        self.f = open(self.bar_file, 'r')
        self.dbar = dbar_repo

        # the time shifting start/stops, see Note 1
        self.utc10 = l1.TradingDayIterator.local_ymd_to_utc('20180530', 18, 0, 0)
        self.utc11 = l1.TradingDayIterator.local_ymd_to_utc('20180626', 17, 0, 0)
        self.utc20 = l1.TradingDayIterator.local_ymd_to_utc('20180626', 18, 0, 0)
        self.utc21 = l1.TradingDayIterator.local_ymd_to_utc('20180817', 17, 0, 0)
        self.bar_sec = 1  # always fixed as 1 second bar for C++ l1 bar writer

    def read(self, noret=False) :
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
            lastpx=None
            while True :
                day, utc, bcols, ecols = self._read_day(f,lastpx=lastpx)
                if day is not None :
                    print 'read day ', day, ' ', len(utc), ' bars.', ' has ext:', ecols is not None
                    if len(utc) > 0 :
                        if self.dbar is not None:
                            self._upd_repo(day, utc, bcols, ecols)
                            repo.remove_outlier_lr(self.dbar, day, day)
                        lastpx=bcols[-1,5]
                        if not noret :
                            darr.append(day)
                            uarr.append(utc)
                            barr.append(bcols)
                            earr.append(ecols)
                else :
                    break

        if self.gzip :
            os.system('gzip -f ' + self.bar_file)
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
        
    def _copy_to_repo(self, day, utc, ow_arr, ow_cols, upd_arr, upd_cols) :
        """
        in case when repo doesn't have that day when L1 update is applied. 
        This could happen if that day was not given by IB_hist nor KDB_hist. 
        In this case, put everything in, fill in missing
        Note 1 lpx in ow_arr,  spd and ism in upd_arr need to be forward 
               and backward filled, other fields
               can be filled with zero
        Note 2 lr will be filled by repo automatically.  The overnight lr is 
               lost.
               TODO : set the overnight lr
        Note 3 Holidays, may have constant prices and zero volumes
               Don't update the day except:
               1) number of nonzero volumes are more than 1
               2) number of lpx changes are more than 1
               Half days, would be fine
        """

        if len(utc) < 10 or\
           len(np.nonzero(ow_arr[:, 0]               !=0)[0])<=1 or \
           len(np.nonzero(ow_arr[1:, 2]-ow_arr[:-1,2]!=0)[0])<=1 :
            print day, ' has too few updates, skipping '
            return
        u0 = self.dbar._make_daily_utc(day, self.bar_sec)
        utcix, zix = repo.ix_by_utc(u0, utc, verbose=False)

        # write a utc and empty lr in first just to make col_idx in order
        if repo.col_idx('utc') not in ow_cols :
            self.dbar.overwrite([np.vstack((u0, np.zeros(len(u0)))).T], [day], [repo.col_idx(['utc','lr'])], self.bar_sec)

        # basic arr: [vol, vbs, lpx]
        # lpx fwd_bck_fill, others fill 0
        ow_arr0 = np.zeros((len(u0), len(ow_cols)))
        ow_arr0[utcix, :] = ow_arr[zix, :]
        ix = np.nonzero(np.array(ow_cols) == repo.col_idx('lpx'))[0]
        # forward/backward fill lpx and ism1
        if len(ix) == 1:
            #forward and backward fill ix
            repo.fwd_bck_fill(ow_arr0[:, ix[0]], v=0)
        self.dbar.overwrite([ow_arr0], [day], [ow_cols], self.bar_sec)

        # ext arr: [spd, qbc, qac, tbc, tsc, ism1], where
        # spd and ism1 fwd_bck_fill, others fill 0
        upd_arr0 = np.zeros((len(u0), len(upd_cols)))
        upd_arr0[utcix, :] = upd_arr[zix, :]
        for cn in ['spd', 'ism1'] :
            ix = np.nonzero(np.array(upd_cols) == repo.col_idx(cn))[0]
            if len(ix) == 1 :
                repo.fwd_bck_fill(upd_arr0[:, ix[0]], v=0)
        self.dbar.overwrite([upd_arr0], [day], [upd_cols], self.bar_sec)

    def _upd_repo(self, day, utc, bcols, ecols) :
        """
        update day to the daily bar repo
        overwrite the vol and vbx, also the lr based lr (see NOTES)
        update the rest 8 columns

        bcol_arr: array of basic columns for each day.  
                 ['vol', 'vbs', 'spd', 'bs', 'as', 'mid']
        ecol_arr: array of extended columns for each day
                 ['qbc', 'qac', 'tbc', 'tsc', 'ism1']

        ow_col: ['vol', 'vbs', 'lpx']
        upd_col:['spd','bs','as','qbc','qac','qbc','tbc','tsc','ism1']

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

        Note 4:
        There could be mismatch in contract of L1 and IB_hist, or other
        general data error that causes a consistent difference between
        L1 price and IB_hist price.  This diff is defined with 
        abs(mean(lpx-mid)) > 5 * ticks 
        In this case, manual operation needed to resolve the conflict

        Note 5:
        EUX and ICE has first bar's buy volume trashed with big number.
        Simply remove this bar if the buy volume is 100 times more
        than median
        """
        ow_cols = repo.col_idx(['lr','vol','vbs','lpx'])
        mid=bcols[:,-1]
        lm=np.log(mid)
        lr=np.r_[0,lm[1:]-lm[:-1]]
        ow_arr = np.vstack((lr,bcols[:, :2].T,mid)).T
        upd_cols = repo.col_idx(['spd','bs','as'])
        upd_arr = bcols[:, 2:5]
        if ecols is not None:
            upd_cols+=repo.col_idx(['qbc','qac','tbc','tsc','ism1'])
            upd_arr = np.vstack((upd_arr.T, ecols.T)).T

        if self.venue in ['ICE', 'EUX'] :
            if ow_arr[0,repo.ci(ow_cols,repo.volc)] > np.median(ow_arr[:,repo.ci(ow_cols,repo.volc)])*100 :
                print 'wrong volume on first bar of ICE/EUX, removing!'
                utc=np.delete(utc,0)
                mid=np.delete(mid,0)
                ow_arr=np.delete(ow_arr, 0, axis=0)
                upd_arr=np.delete(upd_arr,0,axis=0)

        bar, col, bs = self.dbar.load_day(day)
        if len(bar) == 0 or repo.col_idx('utc') not in col:
            print 'nothing found on ', day, ' write as a new day!'
            self._copy_to_repo(day, utc, ow_arr, ow_cols, upd_arr, upd_cols)
            return

        u0 = bar[:, repo.ci(col, repo.col_idx('utc'))]
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
            lpx_hist = bar[:, repo.ci(col, repo.col_idx('lpx'))]
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
            if len(zix) != len(utc) :
                print 'removing ix after adjusting utc, ',
                print 'some ix moved out of daily utc: len(utc)=%d, len(zix)=%d'%(len(utc), len(zix))
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

        #########################################################################
        # check for data errors
        # raise exception if l1 data is off with lpx of the IB_hist by 5 ticks
        # This is usually caused by contract mismatch in l1 and IB_hist data
        # Unfortunately this needs attention to resolve the conflict

        # Note: use lr to reconstruct lpx, this could solve the issue
        #########################################################################
        if repo.col_idx('lpx') in col :
            # check for potential mismatch
            lpx = bar[utcix, repo.ci(col, repo.col_idx('lpx'))]
            mid = ow_arr[:, repo.ci(ow_cols,repo.lpxc)]
            diff = np.abs(np.mean(lpx-mid))
            if diff > l1.asset_info(self.symbol)[0] * 5 :
                #raise ValueError('contract mismatch on '+ day + ' for ' + self.symbol + ' diff ' + str(diff) + ' cnt ' + str(len(mid)))
                print 'contract mismatch on '+ day + ' for ' + self.symbol + ' diff ' + str(diff) + ' cnt ' + str(len(mid))

                # use lr to update, repo should recontruct lpx
                ix=repo.ci(ow_cols, repo.lpxc)
                ow_arr=np.delete(ow_arr,ix,axis=1)
                ow_cols.remove(repo.lpxc)
            else :
                # use lpx to update, repo should reconstruct lr
                ix=repo.ci(ow_cols, repo.lrc)
                ow_arr=np.delete(ow_arr,ix,axis=1)
                ow_cols.remove(repo.lrc)
        else :
            # we have a day but no lpx in columns, that's an error
            # if this happens a lot, then consider write a new day
            print 'ERROR! no lpx on ' + day + ' for ' + self.symbol + ' write as new!'
            self.dbar.remove_day(day)
            self._copy_to_repo(day, utc, ow_arr, ow_cols, upd_arr, upd_cols)
            return

        ix0=1 if utcix[0]==0 else 0
        self.dbar.overwrite([ow_arr[ix0:, :]], [day], [ow_cols], self.bar_sec, rowix=[utcix[ix0:]])

        # upd_arr col: ['spd', 'bs', 'as', 'qbc', 'qac', 'tbc', 'tsc', 'ism1']
        # 'spd' needs to fill back/forward
        # ism1 needs to fill with mid
        # all other columns need to fill-in zeros on missing bars of the day
        upc = np.zeros((len(u0), len(upd_cols)))
        upc[utcix, :] = upd_arr

        # 'spd' needs to fill back/forward
        repo.fwd_bck_fill(upc[:,0],v=0)

        # ism1 needs to fill with mid
        bar, col, bs = self.dbar.load_day(day)
        lpx = bar[:, repo.ci(col, repo.col_idx('lpx'))]
        ix = np.nonzero(upc[:, -1]==0)[0]
        upc[ix, -1] = lpx[ix]

        # use overwriting onto the columns for the whole day
        # because update doesn't allow onto existing columns
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
        bcol=[cols[5] + cols[6], cols[5]-cols[6], cols[3]-cols[2], cols[1], cols[4], (cols[2] + cols[3])/2]
        if self.venue == 'ETF' :
            # IB's ETF size round to 100.  update
            # vol,vbs,bs,as to be consistent with KDB
            for c in [0,1,3,4] :
                bcol[c]*=100
                bcol[c]+=50
        return bcol

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

        ism1 = cols[12]
        if ism1 > 1e+7 or ism1 < 0 or \
            cols[8] < 0 or cols[9] < 0 or \
            cols[10] < 0 or cols[11] < 0 or \
            cols[8] > 1e+7 or cols[9] > 1e+7 or cols[10]>1e+7 or cols[11]>1e+7 :
                print 'Found bad value in ext column, set to 0: ', [cols[8], cols[9], cols[10], cols[11], cols[12]]
                return [0,0,0,0,0]
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

    def _read_day(self, bfp, lastpx=None) :
        """
        day, utc, bcols, ecols = read_next_day(self, bfp)
        basic: the basic fields without utc: ['vol', 'vbs', 'spd', 'bs', 'as', 'mid']
        ext: the extended fields:            ['qbc', 'qac', 'tbc', 'tsc', 'ism1']
        read a day worth of bars in, normalize to either basic or ext format
        bfp: the file descriptor for read on the next lines.  
        day: the day obtained
        utc: an array of time stamp (in second) for each bar
        bcols and ecols: 2D arraies of 5 columns (basic) and 5 columns (ext)
        
        Note 1:
        Upon return the file descriptor can be read for the next day
        Note, the rule for whether a day has basic or ext is the following:
        1. day before 7/17 don't have ext
        2. 13 columns for all lines in the day

        Note 2:
        lastpx is used to remove the first stuck-up px of each day. 
        The tp upon starting could use the old price from shm for a
        while before new price in.  Remove the first px that is 
        same with lastpx.
        lastpx can be last tick of previous day, or in case the
        first day in bar directory, the first px of first day
        """

        day = None
        tcol = []
        bcols = []
        ecols = []
        parse_ext = True
        first_tick=True
        ticks=0
        while True :
            l = bfp.readline()
            if len(l) > 20 : # some minimum size
                utc, basic, ext = self._parse_line(l, parse_ext=parse_ext)
                if basic is None : 
                    # parsing error, next
                    continue
                if ext is None :
                    parse_ext = False
                d0 = l1.TradingDayIterator.utc_to_local_trading_day(utc)
                if day is None :
                    day = d0
                elif day != d0 :
                    bfp.seek(-len(l), 1)
                    break

                if first_tick :
                    # check stuck up first tick
                    thispx=basic[5]
                    if lastpx is None :
                        lastpx = thispx
                    if thispx == lastpx :
                        ticks+=1
                        continue
                    else :
                        first_tick=False
                        print 'removed first ', ticks, ' ticks for stuck'

                bcols.append(basic)
                if ext is not None :
                    ecols.append(ext)
                tcol.append(utc)
            else :
                break
        bcols = np.array(bcols)
        ecols = np.array(ecols) if parse_ext else None
        tcol = np.array(tcol)

        # this is slightly complicated, as the bar will repeat
        # old price in case of disconnection (i.e. missing)
        # so we shouldn't include those stucked ticks in
        if parse_ext  :
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
                            #print 'missing data detected starting on ', datetime.datetime.fromtimestamp(last_eq)
                else :
                    if missing :
                        missing = False
                        marr.append(u0)
                        miss_arr.append(copy.deepcopy(marr))
                        #print 'missing data end at ', datetime.datetime.fromtimestamp(u0)
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


###
# Some verification scripts
###
def verify_lpx_lr_vol_vbs_ism(bar, bar5, uarr, barr, earr, day):
    """
    bar is the repo.dbar.load_day(day) on only IB_hist data
    bar5 is the repo.dbar.load_day(day) after applying the l1 updates
    uarr, barr, earr is the l1bar.read() without updating the repo

    This is to plot the differences of IB history, l1 bars and
    the repo after l1 bar updates

    NOTE: repo.col_idx() assumes the bar has everything (by dbar.load_day)
    and repo.ci(col, idx) is omitted
    """
    f=5

    import matplotlib.pyplot as pl
    fig=pl.figure()
    ax1=fig.add_subplot(f,1,1)
    ax1.plot(bar[:, 0], bar[:, repo.col_idx('lpx')],'.-', label='hist lpx')
    ax1.plot(uarr, barr[:, -1],                            label='l1 mid'  )
    ax1.plot(bar5[:, 0], bar5[: , repo.col_idx('lpx')],   label='repo lpx')
    ax1.plot(bar5[:, 0], bar5[:, repo.col_idx('ism1')],  label='repo ism1')
    ax1.legend() ; ax1.grid()
    ax1.set_title('verify ' + day)
    
    ax2 = fig.add_subplot(f,1,2,sharex=ax1)
    ax2.plot(bar[:, 0], np.cumsum(bar[:, repo.col_idx('vol')]), '.-',    label='hist vol')
    ax2.plot(bar5[:, 0], np.cumsum(bar5[:, repo.col_idx('vol')]), 'g-',  label='repo vol')
    ax2.legend() ; ax2.grid()
    
    ax3 = fig.add_subplot(f,1,3,sharex=ax1)
    ax3.plot(bar[:, 0], np.cumsum(bar[:, repo.col_idx('vbs')]), '.-',    label='hist vbs')
    ax3.plot(uarr, np.cumsum(barr[:, 1]), '.-',                          label='l1 vbs')
    ax3.plot(bar5[:, 0], np.cumsum(bar5[:, repo.col_idx('vbs')]), 'g-',  label='repo vbs')
    ax3.legend() ; ax3.grid()
    
    ax4 = fig.add_subplot(f,1,4,sharex=ax1)
    ax4.plot( uarr, np.cumsum(barr[:, 3]-barr[:, 4]), '.-',                                     label='l1 cumsum(bs-as)')
    ax4.plot(bar5[:, 0], np.cumsum( bar5[:, repo.col_idx('bs')] - bar5[:, repo.col_idx('as')]), label='repo cumsum(bs-as)')
    ax4.legend() ; ax4.grid()

    ax5 = fig.add_subplot(f, 1,5,sharex=ax1)
    ax5.plot(uarr,np.cumsum(earr[:, 2]-earr[:,3]), '.-',                                 label='l1 cumsum(tbc-tas)')
    ax5.plot(uarr,np.cumsum(earr[:, 0]-earr[:,1]), '.-',                                 label='l1 cumsum(qbc-qac)')
    ax5.plot(bar5[:, 0], np.cumsum( bar5[:, repo.col_idx('tbc')] - bar5[:, repo.col_idx('tsc')]), label='repo cumsum(tbc-tsc)')
    ax5.plot(bar5[:, 0], np.cumsum( bar5[:, repo.col_idx('qbc')] - bar5[:, repo.col_idx('qac')]), label='repo cumsum(qbc-qac)')
    ax5.legend() ; ax5.grid()

def test_l1(bar_file='bar/20180727/NYM_CL_B1S.csv', hist_load_date = None, symbol = 'CL', repo_path='repo_test', bs=1) :
    """
    need to run at the kisco root path 
    if hist_load_date is not None, it should be a [start_day, end_day] for the repo to be loaded with IB Histroy
       This is only needed for initialization.  Typically you can save a directory of repo and use
       rm -fR and cp -fR to achieve this
    """
    if hist_load_date is not None :
        print 'create repo ', repo_path, ' and load history dates: ', hist_load_date
        import os
        import IB_hist as ibhist
        os.system('mkdir -p ' + repo_path + ' > /dev/null 2>&1')
        dbar = repo.RepoDailyBar(symbol, repo_path=repo_path, create=True)
        try :
            ibhist.gen_daily_bar_ib(symbol, hist_load_date[0], hist_load_date[1],bs,dbar_repo=dbar, get_missing=False)
        except :
            pass
    else :
        print 'using existing repo at ', repo_path
        dbar = repo.RepoDailyBar(symbol, repo_path=repo_path)

    # read l1 updates from L1 Bar file
    l1bar = L1Bar(symbol,bar_file, None)
    darr, uarr, barr, earr = l1bar.read()

    # save history bar without l1 updates
    bars = []
    for d in darr :
        bar, col, bs = dbar.load_day(d)
        bars.append(copy.deepcopy(bar))

    # update repo with l1 and save to bar5
    bar5s=[]
    l1bar2 = L1Bar(symbol,bar_file, dbar)
    darr2, uarr2, barr2, earr2 = l1bar2.read()
    for d in darr :
        bar5, col, bs = dbar.load_day(d)
        bar5s.append(copy.deepcopy(bar5))

    # ready to go
    for bar, bar5, d, ua, ba, ea in zip(bars, bar5s, darr, uarr, barr, earr) :
        verify_lpx_lr_vol_vbs_ism(bar, bar5, ua, ba, ea, d)

#bar_dir = [20180629,20180706,20180713,20180720,20180727,20180803,20180810,20180817,20180824,20180907,20180914,20180921,20180928,20181005,20181012,20181019,20181026,20181102,20181109,20181116,20181123,20181207,20181214,20181221,20190104,20190111,20190118,20190125,20190201,20190208,20190215,20190222,20190301,20190308,20190315,20190322]
bar_dir = [20180629,20180706,20180713,20180720,20180727,20180803,20180810,20180817,20180824,20180907,20180914,20180921,20180928,20181005,20181012,20181019,20181026,20181102,20181109,20181116,20181123,20181207,20181214,20181221,20190104]

def gzip_everything(bar_path='./bar') :
    os.system('for f in `find ' + bar_path + ' -name *.csv -print` ; do echo "gzip $f" ; gzip -f $f ; done ')

def ingest_all_l1(bar_date_dir_list=None, repo_path='./repo', sym_list=None, bar_path='./bar') :
    """
    ingest all the symbols in bar_date_dir, including the future, fx, etf and future_nc
    for each *_B1S.csv* file: 
    read l1bar for symbol from bar_date_dir, i.e. NYM_CL_B1S.csv.gz
    if bar_date_dir_list is not none, it should be a list of bar_date_dir, i.e. [20180629,20180706]
       otherwise, all dates in bar_path
    if repo_path is not None, update the repo for that symbol. 
    if sym_list is not None, then only these symbols are updated,
       otherwise, all symbols found in the bar directory will be updated
    Note 1: future_nc has *_B1S_bc.csv*,  i.e. NYM_CL_B1S_bc.csv.gz
            and have different repo_path than front contract, 
            obtained by repo.nc_repo_path(repo_path), i.e. repo_nc
    """
    repo_path_nc = repo.nc_repo_path(repo_path) if repo_path is not None else None
    #gzip_everything(bar_path)
    if bar_date_dir_list is None :
        b = glob.glob(bar_path+'/*')
        bar_date_dir_list=[]
        for b0 in b :
            bar_date_dir_list.append(b0.split('/')[-1])
        print 'got ', len(bar_date_dir_list), ' directories to update'

    for bar_date_dir in bar_date_dir_list :
        fs_front = bar_path+'/'+str(bar_date_dir)+'/*_B1S.csv*'
        fs_back  = bar_path+'/'+str(bar_date_dir)+'/*_B1S_bc.csv*'
        for fs, rp in zip([fs_front, fs_back], [repo_path, repo_path_nc]) :
            fn = glob.glob(fs)
            print 'found ', len(fn), ' files for ', rp
            for f in fn :
                sym = f.split('/')[-1].split('_')[1]
                if sym_list is not None and sym not in sym_list :
                    print sym, ' not in ', sym_list, ' ignored. '
                    continue
                print 'getting ', sym, ' from ', f, ' repo_path ', rp
                dbar = None
                if rp is not None :
                    dbar = repo.RepoDailyBar(sym, repo_path=rp, create=True)
                l1b = L1Bar(sym, f, dbar)
                try :
                    l1b.read(noret=True)
                except :
                    import traceback
                    traceback.print_exc()

def fix_eux_ice_first_bar_volume(repo_l1='./repo_l1', repo_hist='./repo', sday=None, eday=None) :
    """
    EUX and ICE had the first bar's buy volume wrong, try to get it from 
    repo_hist, or set to 0
    This is included in the L1Bar.read(), so it's not needed.
    """
    if sday is None :
        sday = '00000000'
    if eday is None :
        eday = '99999999'

    eur_sym = l1.ven_sym_map['EUX']
    ice_sym = l1.ven_sym_map['ICE']

    #for sym in eur_sym + ice_sym :
    for sym in ['LCO'] :
        print 'symbol: ', sym
        dbarl1=repo.RepoDailyBar(sym, repo_path=repo_l1)
        dbar=repo.RepoDailyBar(sym, repo_path=repo_hist)

        days=dbarl1.all_days()
        for d in days :
            if d < sday or d > eday:
                continue

            print 'day ', d, 
            b1,c1,bs1=dbarl1.load_day(d)
            b,c,bs=dbar.load_day(d)
            changed=False
            ix0 = np.nonzero(b1[:, repo.ci(c1,repo.volc)]>1e-10)[0]
            if len(ix0)==0 :
                print 'all zero volume! '
                continue
            ix0=ix0[0]
            vol0=b1[ix0, repo.ci(c1,repo.volc)]
            vbs0=b1[ix0, repo.ci(c1,repo.vbsc)]
            print vol0, vbs0,
            if bs == bs1 and len(b1)>0 and len(b)>0 :
                # use b's first bar volume
                vol0=b[ix0,repo.ci(c,repo.volc)]
                vbs0=b[ix0,repo.ci(c,repo.vbsc)]
                changed=True
                print 'using repo! ', vol0, vbs0
            else :
                vbs1=b1[:,repo.ci(c1,repo.vbsc)]
                if np.abs(vbs1[ix0]) > 100 * np.median(np.abs(vbs1)) :
                    # set to 0
                    vol0=0
                    vbs0=0
                    changed=True
                    print 'setting to 0!'
            if changed :
                b1[ix0, repo.ci(c1,repo.volc)]=vol0
                b1[ix0, repo.ci(c1,repo.vbsc)]=vbs0
                dbarl1._dump_day(d, b1,c1,bs1)
            else :
                print 'all good!'

def backup_to_repo(sym_arr, sday, eday, repo_l1='./repo_back', repo_hist='./repo') :
    """
    backup the content of repo_hist to repo_l1 before ingestion
    """
    repo.copy_from_repo(sym_arr,repo_path_write=repo_l1, repo_path_read_arr=[repo_hist], bar_sec=1, sday=sday, eday=eday, keep_overnight='no')

def get_from_back(sym_arr, sday, eday, repo_l1='./repo_back', repo_hist='./repo') :
    """
    recover from backup
    """
    repo.copy_from_repo(sym_arr,repo_path_write=repo_hist, repo_path_read_arr=[repo_l1], bar_sec=1, sday=sday, eday=eday, keep_overnight='no')

def remove_outlier(sym_arr,repo_path, sday, eday) :
    """
    sym_arr=['6Z','6M','6R'], front and back
    """
    for sym in sym_arr :
        print sym
        dbar = repo.RepoDailyBar(sym, repo_path=repo_path)
        repo.remove_outlier_lr(dbar, sday, eday)



