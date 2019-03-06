import numpy as np
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo
import os

import KDB_hist as kh

def bar_by_file(fn) :
    """
    adapt the cme tick-by-tick data into a KDB trd file
    So the CME file has format of
    1496275455186063751,CLN7,Y,0,48.86,1,2,5

    utc, contract, side [B/A/E/F/Y/Z/O], level, px, sz, cnt, act, type(32/42/43) OID, QUE
    utc: in nano-second
    contract: the future contract
    side:  B: bid, A: ask, E: implied bid, F: implied ask, Y: buy, Z: sell, O: from spread
    level: quote level of px, 1-10, 0 for trade
    px: price
    sz: size
    act: 0(New), 1(change), 2(del), 3(del thru), 4(del from), 5(trd)
    cnt: number of orders
    OID/QUE: order id and queue priority

    Return:
    [utc, px, bsvol]
    """

    tmp_f='/tmp/'+fn.split('/')[-1]+'_'+datetime.datetime.now().strftime('%s')+'.tmp'
    gpstr="grep \"\\,Y\\,\\|\\,Z\\,\" " + fn + " > " + tmp_f
    print gpstr
    os.system(gpstr)

    # read into the ts format
    bar_raw=np.genfromtxt(tmp_f,delimiter=',',usecols=[0,2,4,5],dtype=[('utc','i8'),('dir','|S2'),('px','f8'),('sz','f8')])

    utc=bar_raw['utc']/1000
    utc=utc.astype(float)/1000000.0 # microseconds
    px=bar_raw['px']
    d=np.ones(len(utc))
    ix=np.nonzero(bar_raw['dir']=='Z')[0]
    sz=bar_raw['sz']
    sz[ix]*=-1
    return np.vstack((utc, px, sz)).T

def get_fn(cme_path, symbol, day, con) :
    return cme_path+'/OutRights/'+day[:6]+'/'+day+'/'+symbol+'/'+day+'_'+con+'.csv'
    
def gen_bar(sym_array, sday, eday, repo_cme_path='./repo_cme', cme_path='./cme', bar_sec=1, nc=False) :
    """
    getting from the ts [utc, px, signed_vol]
    output format bt, lr, vl, vbs, lrhl, vwap, ltt, lpx

    repo_cme_path: repo to store the 1S trd bars

    return : None
        update (remove first) dbar with bar_arr, days, col_arr
    """

    if nc :
        assert repo_cme_path[-2:]=='nc', 'repo_cme_path='+repo_cme_path+' not ending with nc'
    for symbol in sym_array :
        try :
            dbar = repo.RepoDailyBar(symbol, repo_path=repo_cme_path)
        except :
            print 'repo_trd_path failed, trying to create'
            dbar = repo.RepoDailyBar(symbol, repo_path=repo_cme_path, create=True)

        start_hour, end_hour = l1.get_start_end_hour(symbol)
        TRADING_HOURS=end_hour-start_hour
        # sday has to be a trading day
        it = l1.TradingDayIterator(sday)
        tday = it.yyyymmdd()
        if tday != sday :
            raise ValueError('sday has to be a trading day! sday: '+sday + ' trd_day: ' + tday)

        lastpx=0
        prev_con=''
        while tday <= eday :
            eutc = it.local_ymd_to_utc(tday,h_ofst=end_hour)
            sutc = eutc - (TRADING_HOURS)*3600
            if nc :
                con=l1.FC_next(symbol, tday)[0]
            else :
                con=l1.FC(symbol, tday)

            con=symbol+con[-2:]
            try :
                bar = bar_by_file(get_fn(cme_path, symbol, tday, con))
            except (KeyboardInterrupt) :
                print 'interrupt!'
                return
            except :
                print 'problem getting ', symbol, tday
                bar=[]

            if len(bar) == 0 :
                lastpx=0
                prev_con=''
            else :
                # this is the good case, prepare for the bar
                # 1) get bar with start/stop, 2) contract updated 3) lastpx
                # need to allow for entire content being in one ta, i.e. some
                # days having tds==2 but all contents in one ta, due to gmt_offset

                # have everything, need to get to
                # output format bt, lr, vl, vbs, lrhl, vwap, ltt, lp
                if lastpx==0 or prev_con!=con:
                    lastpx=bar[0,1]
                bt=np.arange(sutc+bar_sec,eutc+bar_sec,bar_sec)
                tts=np.r_[sutc,bar[:,0]]
                pts=np.r_[bar[0,1],bar[:,1]]
                vts=np.r_[0,bar[:,2]]
                pvts=np.abs(vts)*pts

                pxix=np.clip(np.searchsorted(tts[1:],bt+1e-6),0,len(tts)-1)
                lpx=pts[pxix]
                lr = np.log(np.r_[lastpx,lpx])
                lr=lr[1:]-lr[:-1]

                # tricky way to get index right on volumes
                btdc=np.r_[0,np.cumsum(vts)[pxix]]
                vbs=btdc[1:]-btdc[:-1]
                btdc=np.r_[0,np.cumsum(np.abs(vts))[pxix]]
                vol=btdc[1:]-btdc[:-1]

                # even tickier way to get vwap/ltt right
                ixg=np.nonzero(vol)[0]
                btdc=np.r_[0, np.cumsum(pvts)[pxix]]
                vwap=lpx.copy()  #when there is no vol
                vwap[ixg]=(btdc[1:]-btdc[:-1])[ixg]/vol[ixg]
                ltt=np.zeros(len(bt))
                ltt[ixg]=tts[pxix][ixg]
                repo.fwd_bck_fill(ltt, v=0)

                # give up, ignore the lrhl for trd bars
                lrhl=np.zeros(len(bt))

                b=np.vstack((bt,lr,vol,vbs,lrhl,vwap,ltt,lpx)).T
                d=tday
                c=repo.kdb_ib_col
                dbar.remove_day(d)
                dbar.update([b],[d],[c],bar_sec)
                lastpx=lpx[-1]
                prev_con=con

            it.next()
            tday=it.yyyymmdd()


