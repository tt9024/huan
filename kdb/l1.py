import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import glob

class TradingDayIterator :
    def __init__(self, yyyymmdd,adj_start=True) :
        self.dt=datetime.datetime.strptime(yyyymmdd, '%Y%m%d')
        if adj_start and self.dt.weekday() > 4 :
            self.next()

    def yyyymmdd(self) :
        return datetime.datetime.strftime(self.dt, '%Y%m%d')

    def weekday(self) :
        return self.dt.weekday()

    def next(self) :
        self.dt+=datetime.timedelta(1)
        while self.dt.weekday() > 4 :
            self.dt+=datetime.timedelta(1)
        return self

    def prev(self) :
        self.dt-=datetime.timedelta(1)
        while self.dt.weekday() > 4 :
            self.dt-=datetime.timedelta(1)
        return self

    def prev_n_trade_day(self, delta) :
        wk=delta/5
        wd=delta%5
        delta += 2*wk
        if self.dt.weekday() < wd :
            delta += 2
        self.dt-=datetime.timedelta(delta)
        return self

    def next_n_trade_day(self, delta) :
        wk=delta/5
        wd=delta%5
        delta += 2*wk
        if 4 - self.dt.weekday() < wd :
            delta += 2
        self.dt+=datetime.timedelta(delta)
        return self

    def last_month_day(self, dc=1) :
        mm=self.dt.month
        self.next_n_trade_day(dc)
        #self.next()
        mm2=self.dt.month
        self.prev_n_trade_day(dc)
        #self.prev()
        if mm != mm2 :
            return True
        return False

    def to_local_utc(self, h_ofst,m_ofst,s_ofst) :
        assert np.abs(h_ofst) < 24 and np.abs(m_ofst) < 60 and np.abs(s_ofst) < 60, 'offset wrong'
        ymd=self.yyyymmdd()
        return TradingDayIterator.local_ymd_to_utc(ymd,h_ofst,m_ofst,s_ofst)

    @staticmethod
    def local_ymd_to_utc(ymd,h_ofst=0,m_ofst=0,s_ofst=0) :
        """
        this returns the utc of 0 o'clock local time at ymd, w.r.t to the offset specified 
        at h/m/s offset On a unix, it could simply be

            dt = strptime(ymdhms)
            float(dt.strftime('%s')

        But on a windows system, python won't give correct utc.  This requires a platform
        independant way of getting utc from a local time stamp

        This requires the python's datetime being right on strptime to adjust DST. 
        However, strptime adjusts DST on 6am of a DST Sunday, instead of a 2am. 
        Bug fixing: on a Sunday of DST change, the timestamp is not accurate around
        that Sunday's 2am to 6am.  TODO: fix that issue as a low priority. 
        """
        ymdhms = '%s%02d%02d%02d'%(ymd, h_ofst, m_ofst,s_ofst)
        #dt=datetime.datetime.strptime(ymd,'%Y%m%d')
        dt=datetime.datetime.strptime(ymdhms,'%Y%m%d%H%M%S') # this should adjust DST
        utc0=(dt-datetime.datetime(1970,1,1)).total_seconds()
        dt0=datetime.datetime.fromtimestamp(utc0)
        local_offset=(dt-dt0).total_seconds()
        #utc0=utc0+local_offset+h_ofst*3600+m_ofst*60+s_ofst
        utc0=utc0+local_offset
        return utc0

    @staticmethod
    def local_dt_to_utc(dt) :
        return TradingDayIterator.local_ymd_to_utc(dt.strftime('%Y%m%d'),dt.hour,dt.minute,dt.second)

    @staticmethod
    def utc_to_local_ymd(utc) :
        return datetime.datetime.fromtimestamp(utc).strftime('%Y%m%d')


def tradinghour(dt) :
    """
    trading hour is defined time from 18:05pm to 17:00pm next day
    """
    if dt.hour==17 and dt.minute>0 :
        return False
    if dt.hour==18 and dt.minute==0 :
        return False
    return True

MonthlyFrontContract =    ['G','H','J','K','M','N','Q','U','V','X','Z','F']
BiMonthlyFrontContract =  ['G','J','J','M','M','Q','Q','V','V','Z','Z','G']
GCMonthlyFrontContract =  ['G','J','J','M','M','Q','Q','Z','Z','Z','Z','G']
HGMonthlyFrontContract =  ['H','H','K','K','N','N','U','U','Z','Z','Z','H']
#HGMonthlyFrontContract =  ['H','H','H','K','K','N','N','U','U','Z','Z','Z']
QuartlyFrontContract =    ['H','H','H','M','M','M','U','U','U','Z','Z','Z']
OddMonthlyFrontContract = ['H','H','K','K','N','N','U','U','Z','Z','Z','H']
RollDates = {'CL':  [MonthlyFrontContract,   [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]], \
             'LCO': [MonthlyFrontContract,   [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]], \
             'LFU': [MonthlyFrontContract,   [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]], \
             'LOU': [MonthlyFrontContract,   [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]], \
             'NG':  [MonthlyFrontContract,   [24, 22, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]], \
             'HO':  [MonthlyFrontContract,   [26, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]], \
             'RB':  [MonthlyFrontContract,   [26, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]], \
             'ES':  [QuartlyFrontContract,   [31, 31, 12, 31, 31, 12, 31, 31, 12, 31, 31, 12]], \
             'ZB':  [QuartlyFrontContract,   [31, 31,  1, 31, 31,  1, 31, 31,  1, 31, 31,  1]], \
             'ZN':  [QuartlyFrontContract,   [31, 31,  1, 31, 31,  1, 31, 31,  1, 31, 31,  1]], \
             'ZF':  [QuartlyFrontContract,   [31, 31,  1, 31, 31,  1, 31, 31,  1, 31, 31,  1]], \
             'US':  [QuartlyFrontContract,   [31, 31,  1, 31, 31,  1, 31, 31,  1, 31, 31,  1]], \
             'TY':  [QuartlyFrontContract,   [31, 31,  1, 31, 31,  1, 31, 31,  1, 31, 31,  1]], \
             'FV':  [QuartlyFrontContract,   [31, 31,  1, 31, 31,  1, 31, 31,  1, 31, 31,  1]], \
             'FDX': [QuartlyFrontContract,   [31, 31, 17, 31, 31, 17, 31, 31, 17, 31, 31, 17]], \
             'STXE':[QuartlyFrontContract,   [31, 31, 17, 31, 31, 17, 31, 31, 17, 31, 31, 17]], \
             'FGBL':[QuartlyFrontContract,   [31, 31,  7, 31, 31,  7, 31, 31,  7, 31, 31,  7]], \
             'FGBM':[QuartlyFrontContract,   [31, 31,  7, 31, 31,  7, 31, 31,  7, 31, 31,  7]], \
             'FGBS':[QuartlyFrontContract,   [31, 31,  7, 31, 31,  7, 31, 31,  7, 31, 31,  7]], \
             'FGBX':[QuartlyFrontContract,   [31, 31,  7, 31, 31,  7, 31, 31,  7, 31, 31,  7]], \
#            'GC': [BiMonthlyFrontContract,  [25, 31, 25, 31, 25, 31, 25, 31, 25, 31, 26, 31]],  # 'V' is not good
             'GC': [GCMonthlyFrontContract,  [25, 31, 25, 31, 25, 31, 25, 31, 31, 31, 26, 31]],  # I am not sure about this \
#             'HG': [HGMonthlyFrontContract, [31, 31, 25, 31, 25, 31, 25, 31, 25, 31, 31, 25]], 
             'HG': [HGMonthlyFrontContract,  [31, 25, 31, 25, 31, 27, 31, 29, 31, 31, 29, 31]], 
             'SI': [HGMonthlyFrontContract,  [31, 25, 31, 25, 31, 27, 31, 29, 31, 31, 29, 31]], 
             'ZC': [OddMonthlyFrontContract, [31, 20, 31, 20, 31, 20, 31, 20, 31, 31, 20, 31]],  # I am not sure about this \
             'FX': [QuartlyFrontContract,    [31, 31, 13, 31, 31, 13, 31, 31, 13, 31, 31, 13]]}  # this may change

FXFutures = ['6A','6B','6C','6E','6J','6M','6N','6Z','6R','AD','BP','CD','URO','JY','MP','NE']
RicMap = {'6A':'AD', '6B':'BP', '6C':'CD', '6E':'URO', '6J':'JY', '6M':'MP', '6N':'NE','ZC':'C'}
ICEFutures = ['LCO','LFU','LOU']
# todo: fill this map
SymbolTicks = {'CL':0.01, 'ES':0.25, 'NG':0.001, 'LCO':0.01, 'LFU':0.25, 'LOU':0.0001, 'HO':0.0001}

#######################################################################
## !!! Be very careful about changin the following ven_sym_map and
## venue_by_symbol definitions as ibbar uses it for live trading
## At minimum, do not delete any of the definitions
## Adding should not be a problem
ven_sym_map={'NYM':['CL','NG','HO','RB','GC','SI','HG'], \
             'CME':['ES','6A','6C','6E','6B','6J','6N','6R','6Z','6M'],\
             'CBT':['ZB','ZN','ZF','ZC'],\
             'EUX':['FDX','STXE','FGBX','FGBL','FGBS','FGBM'],\
             'FX' :['AUD.CAD','AUD.JPY','AUD.NZD','CAD.JPY','EUR.AUD',\
                    'EUR.CAD','EUR.CHF','EUR.GBP','EUR.JPY','EUR.NOK',\
                    'EUR.SEK','EUR.TRY','EUR.ZAR','GBP.CHF','GBP.JPY',\
                    'NOK.SEK','NZD.JPY','EUR.USD','USD.ZAR','USD.TRY',\
                    'USD.MXN','USD.CNH','XAU.USD','XAG.USD'],\
             'ICE':['LCO','LFU','LOU']};

future_venues=['NYM','CME','CBT','EUX','ICE']
fx_venues=['FX']
def venue_by_symbol(symbol) :
    for k,v in ven_sym_map.items() :
        if symbol in v :
            return k
    raise ValueError('venue not found for ' + symbol)

def get_start_end_hour(venue) :
    """
    start on previous day's 18, end on 17, 
    except ICE, starts from 20 to 18
    To add other non cme/ice venues, such as IDX and FX venues
    """
    if venue == 'ICE' :
        return -4, 18
    return -6, 17
## At minimum, do not delete any of the definitions
## Adding should not be a problem
## check with ib/kisco/ibbar.py, it uses the above two functions for live
##########################################################################

def is_fx_future(symbol) :
    return symbol in FXFutures

def FC_ICE_new(symbol, yyyymmdd) :
    ti=TradingDayIterator(yyyymmdd)
    y=ti.dt.year
    m=ti.dt.month
    d=ti.dt.day
    assert y>=2016, 'year have to be great or equal to 2016'
    # make a case of 2016.1.14
    if y==2016 and m==1 :
        if d < 14 :
            return symbol + 'G6'
        else :
            if not ti.last_month_day() :
                return symbol + 'H6'
            else :
                return symbol + 'J6'
    else :
        # roll 6 days before the delivery 
        # seems to capture most of the flow
        rolldates=6
        if not ti.last_month_day(rolldates) :
            ms=MonthlyFrontContract[m%12]
        else :
            ms=MonthlyFrontContract[(m+1)%12]
        ys=str(y)[-1]
        if m>=10 and ms in ['F','G','H','J']:
            ys=str(y+1)[-1]
        return symbol+ms+ys

def FC(symbol, yyyymmdd) :
    dt=datetime.datetime.strptime(yyyymmdd,'%Y%m%d')
    if symbol in ICEFutures :
        # ice roll is tricky
        if dt.year>=2016:
            return FC_ICE_new(symbol, yyyymmdd)
    if symbol in FXFutures :
        symbol0='FX'
    else :
        symbol0=symbol
    if RicMap.has_key(symbol) :
        symbol=RicMap[symbol]
    if not RollDates.has_key(symbol0) :
        raise ValueError('symbol0 not in RollDates for ('+ symbol0 + ')' )
    rd=RollDates[symbol0]
    ms=rd[0][dt.month-1]
    ys=dt.year%10
    if dt.day>rd[1][dt.month-1] :
        ms=rd[0][dt.month%12]
    if dt.month>=10 and ms in ['F','G','H','J']:
        ys = (ys + 1)%10
    return symbol+ms+str(ys)

def FC_next(symbol, yyyymmdd) :
    ti = TradingDayIterator(yyyymmdd)
    fc=FC(symbol, yyyymmdd)
    fc_next=fc
    while fc_next == fc :
        ti.next()
        fc_next = FC(symbol, ti.yyyymmdd())
    return fc_next, ti.yyyymmdd()

def CList(symbol, yyyymmdd, days_overlap = 5) :
    ti=TradingDayIterator(yyyymmdd)
    ti.next_n_trade_day(days_overlap)
    return [ FC(symbol, yyyymmdd), FC(symbol, ti.yyyymmdd()) ]

def quote_valid(bp,bsz,ap,asz) :
    return bp*bsz*ap*asz != 0 and ap > bp

def f0(fn, sym, bar_sec, line_start=1, line_end=1000000) :
    cur_pt = {} # per contract nxt_bt, bp,bsz,ap,asz,utc
    with open(fn, 'r') as f :
        line=f.readline()
        lc=1
        while lc < line_start :
            f.readline()
            lc+=1
        
        while len(line) > 10 :
            if line[0] != '#' : # filter out comment line
                try :
                    #pdb.set_trace()
                    l = line[:-1].split(',')
                    day=l[2]
                    fl=CList(sym,day)
                    ct=l[0]
                    if ct in fl :
                        # found one work on bar time/price 
                        # don't stuff in idle times
                        tm=l[3].split('.')
                        hms = tm[0].split(':')
                        utc=TradingDayIterator.local_ymd_to_utc(day, int(hms[0]), int(hms[1]), int(hms[2]))
                        utc_frac=float(utc)+float('.'+tm[1])
                        # setting up if it's a new contract :
                        if not cur_pt.has_key(ct) :
                            cur_pt[ct]={'nxt_bt':(utc/bar_sec+1)*bar_sec, 'qt':[0,0,0,0,0], 'last_qt':[0,0,0,0,0], 'bars':[]}
                        curpt = cur_pt[ct]

                        if utc >= curpt['nxt_bt'] :
                            qt0=curpt['qt']
                            if quote_valid(qt0[1],qt0[2],qt0[3],qt0[4]) :
                                bt=(int(qt0[0])/bar_sec+1)*bar_sec
                                curpt['bars'].append( np.r_[bt, qt0] )
                            curpt['nxt_bt'] = (utc/bar_sec+1)*bar_sec

                        qt=curpt['last_qt']
                        ap=0;asz=0;bp=0;bsz=0
                        if len(l[6]) > 0 :
                            bp=float(l[6])
                            bsz=int(l[7])
                            if bp*bsz > 0 :
                                qt[1]=bp
                                qt[2]=bsz
                                qt[0]=utc_frac
                        if len(l[8]) > 0 :
                            ap=float(l[8])
                            asz=int(l[9])
                            if ap*asz > 0 :
                                qt[3]=ap
                                qt[4]=asz
                                qt[0]=utc_frac
                        if quote_valid(qt[1],qt[2],qt[3],qt[4]) :
                            curpt['qt']=np.array(qt).copy()

                except (KeyboardInterrupt, SystemExit) :
                    print 'stop ...'
                    f.close()
                    return cur_pt
                except :
                    traceback.print_exc()
                    print 'problem getting quote ', line, l, lc 
            if lc >= line_end :
                print 'read ', lc, ' lines'
                break

            line=f.readline()
            lc+=1
            if lc % 10000 == 0 :
                print 'read ', lc, ' lines'

    return cur_pt

def get_future_bar(symbol_list, start_date, end_date, kdb_util='bin/test') :
    for symbol in symbol_list :
        bar_dir = symbol
        os.system(' mkdir -p ' + bar_dir)
        ti = TradingDayIterator(start_date)
        day=ti.yyyymmdd()
        while day <= end_date :
            fc=FC(symbol, day)
            sday=day
            while day <= end_date :
                ti.next()
                day=ti.yyyymmdd()
                fc0=FC(symbol, day)
                if fc != fc0 :
                    break
            eday=day
            fn=bar_dir+'/'+fc+'_'+sday+'_'+eday+'.csv'
            cmdline=kdb_util + ' ' + fc + ' ' + sday + ' ' + eday + ' > ' + fn
            print 'running ', cmdline
            os.system( cmdline )
            os.system( 'sleep 5' )

        os.system( 'gzip '+ bar_dir+'/*.csv' )

def get_file_size(fn) :
    try :
        return os.stat(fn).st_size
    except :
        return 0

def parse_raw_fx_quote(b) :
    dt=[]
    bp=[]
    ap=[]
    mmid=[]
    for l in b :
        hms=l[2].split('.')[0].split(':')
        d0=TradingDayIterator.lcoal_ymd_to_utc(l[0], int(hms[0]), int(hms[1]),int(hms[2]))
        d0=d0+'.'+l[2].split('.')[-1]
        dt.append(d0)
        bp.append(l[3])
        ap.append(l[4])
        m0=l[5][:5]+l[5][-3:]
        mmid.append(m0)
    return np.vstack ( (dt,bp,ap,mmid) ).T

def get_daily_fx(symbol_list, start_date, end_date, kdb_util='bin/get_fx', FX_repo='FX',skip_header=5) :
    for symbol in symbol_list :
        bar_dir = FX_repo+'/'+symbol.replace('=','').replace('/','')
        os.system(' mkdir -p ' + bar_dir)
        ti = TradingDayIterator(start_date)
        day=ti.yyyymmdd()
        while day <= end_date :
            fn=bar_dir+'/'+symbol+'_'+day+'.csv'
            # check if the file exists and the size is small
            if get_file_size(fn) < 500 and get_file_size(fn+'.gz') < 500 :
                os.system( 'rm -f ' + fn + ' > /dev/null 2>&1')
                os.system( 'rm -f ' + fn + '.gz' + ' > /dev/null 2>&1')
                cmdline=kdb_util + ' ' + symbol + ' ' + day + ' > ' + fn+'.raw'
                print 'running ', cmdline
                os.system( cmdline )

                ### reduce the columns of the bar file
                try :
                    b=np.genfromtxt(fn+'.raw', delimiter=',',skip_header=skip_header,dtype='|S16')
                    b0=parse_raw_fx_quote(b)
                    np.savetxt(fn,b0,fmt='%s,%s,%s,%s')
                    os.system('gzip ' + fn)
                    os.system('rm -f ' + fn+'.raw > /dev/null 2>&1')
                except :
                    print 'problem parsing the raw file ', fn, ', skipping '
                    
                os.system( 'sleep 8' )
            ti.next()
            day=ti.yyyymmdd()

def get_bar_spot(symbol_list, start_date, end_date, kdb_util='bin/get_fx_bar', FX_repo='FX',skip_header=5, bar_days=120) :
    for symbol in symbol_list :
        bar_dir = FX_repo+'/'+symbol.replace('=','').replace('/','').replace('.','')
        os.system(' mkdir -p ' + bar_dir)
        ti = TradingDayIterator(start_date)
        day=ti.yyyymmdd()
        while day <= end_date :
            sday=day
            dc=0
            while day <= end_date and dc<bar_days:
                ti.next()
                day=ti.yyyymmdd()
                dc += 1
                eday=day

            fn=bar_dir+'/'+symbol.replace('=','').replace('/','').replace('.','')+'_'+sday+'_'+eday+'.csv'
            # check if the file exists and the size is small
            if get_file_size(fn) < 500 and get_file_size(fn+'.gz') < 500 :
                os.system( 'rm -f ' + fn + ' > /dev/null 2>&1')
                os.system( 'rm -f ' + fn + '.gz' + ' > /dev/null 2>&1')
                cmdline=kdb_util + ' ' + symbol + ' ' + sday + ' '  + eday + ' > ' + fn
                print 'running ', cmdline
                os.system( cmdline )
                os.system('gzip ' + fn)
                os.system( 'sleep 10' )

def get_future_bar_fix(symbol_list, start_date, end_date, kdb_util='bin/get_bar') :
    for symbol in symbol_list :
        bar_dir = symbol
        os.system(' mkdir -p ' + bar_dir)
        ti = TradingDayIterator(start_date)
        day=ti.yyyymmdd()
        while day <= end_date :
            fc=FC(symbol, day)
            sday=day
            while day <= end_date :
                ti.next()
                day=ti.yyyymmdd()
                fc0=FC(symbol, day)
                if fc != fc0 :
                    break
            eday=day
            fn=bar_dir+'/'+fc+'_'+sday+'_'+eday+'.csv'

            # check if the file exists and the size is small
            if get_file_size(fn) < 500 and get_file_size(fn+'.gz') < 500 :
                os.system( 'rm -f ' + fn + ' > /dev/null 2>&1')
                os.system( 'rm -f ' + fn + '.gz' + ' > /dev/null 2>&1')
                cmdline=kdb_util + ' ' + fc + ' ' + sday + ' ' + eday + ' > ' + fn
                print 'running ', cmdline
                os.system( cmdline )
                os.system ('gzip ' + fn)
                os.system( 'sleep 10' )
        
def get_future_trade(symbol_list, start_date, end_date, kdb_util='bin/get_trade', mock_run=False, front=True, second_front=False, cal_spd_front=False,cal_spd_second=False):
    for symbol in symbol_list :
        bar_dir = symbol
        os.system(' mkdir -p ' + bar_dir)
        ti = TradingDayIterator(start_date)
        day=ti.yyyymmdd()
        while day <= end_date :
            fc=FC(symbol, day)
            # for each day, get trades for FC, FC+, FC/FC+, FC+/FC++
            fc_next, roll_day=FC_next(symbol, day)
            fc_next_next, roll_day=FC_next(symbol, roll_day)
            
            cset=[]
            if front :
                cset.append(fc)
            if second_front :
                cset.append(fc_next)
            if cal_spd_front :
                cset.append(fc+'-'+fc_next[-2:])
            if cal_spd_second :
                cset.append(fc_next+'-'+fc_next_next[-2:])
            for c in cset :
                fn=bar_dir+'/'+c+'_trd_'+day+'.csv'
                print 'checking ', c, fn
                # check if the file exists and the size is small
                if get_file_size(fn) < 500 and get_file_size(fn+'.gz') < 500 :
                    os.system('rm -f ' + fn + ' > /dev/null 2>&1')
                    os.system('rm -f ' + fn + '.gz' + ' > /dev/null 2>&1')
                    cmdline=kdb_util + ' ' + c + ' ' + day + ' > ' + fn
                    print 'running ', cmdline
                    if not mock_run :
                        os.system( cmdline )
                        os.system('gzip ' + fn)
                        os.system( 'sleep 5' )
            ti.next()
            day=ti.yyyymmdd()

def bar_by_file(fn, skip_header=5) :
    bar_raw=np.genfromtxt(fn,delimiter=',',usecols=[0,2,3,4,5,6,7,9,10,11,12], skip_header=skip_header,dtype=[('day','|S12'),('bar_start','|S14'),('last_trade','|S14'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8'),('vwap','<f8'),('volume','i8'),('bvol','i8'),('svol','i8')])
    bar=[]
    for b in bar_raw :
        dt=datetime.datetime.strptime(b['day']+'.'+b['bar_start'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc=float(TradingDayIterator.local_dt_to_utc(dt))
        dt_lt=datetime.datetime.strptime(b['day']+'.'+b['last_trade'].split('.')[0],'%Y.%m.%d.%H:%M:%S')
        utc_lt=float(TradingDayIterator.local_dt_to_utc(dt))+float(b['last_trade'].split('.')[1])/1000.0

        bar0=[utc, utc_lt, b['open'],b['high'],b['low'],b['close'],b['vwap'],b['volume'],b['bvol'],b['svol']]
        bar.append(bar0)
    bar = np.array(bar)
    open_px_col=2
    ix=np.nonzero(np.isfinite(bar[:,open_px_col]))[0]
    bar=bar[ix, :]
    ix=np.argsort(bar[:, 0])
    return bar[ix, :]

def write_daily_bar(bar,bar_sec=5) :
    import pandas as pd
    dt0=datetime.datetime.fromtimestamp(bar[0,0])
    #assert dt.hour < 18 , 'start of bar file hour > 18'
    i=0
    # seek to the first bar greater or equal to 18 on that day
    dt=dt0
    while dt.hour<18 :
        i+=1
        dt=datetime.datetime.fromtimestamp(bar[i,0])
        if dt.day != dt0.day :
            #raise ValueError('first day skipped, no bars between 18pm - 24am detected')
            print 'first day skipped, no bars between 18pm - 24am detected'
            break

    # get the initial day, last price
    day_start=dt.strftime('%Y%m%d')
    utc_s = int(TradingDayIterator.local_ymd_to_utc(day_start, 18, 0, 0))
    x=np.searchsorted(bar[1:,0], float(utc_s-3600+bar_sec))
    last_close_px=bar[x,2]
    print 'last close price set to previous close at ', datetime.datetime.fromtimestamp(bar[x,0]), ' px: ', last_close_px
    day_end=datetime.datetime.fromtimestamp(bar[-1,0]).strftime('%Y%m%d')
    # deciding on the trading days
    if dt.hour > 17 :
        ti=TradingDayIterator(day_start,adj_start=False)
        ti.next()
        trd_day_start=ti.yyyymmdd()
    else :
        trd_day_start=day_start
    trd_day_end=day_end
    print 'preparing bar from ', day_start, ' to ', day_end, ' , trading days: ', trd_day_start, trd_day_end

    ti=TradingDayIterator(day_start, adj_start=False)
    day=ti.yyyymmdd()  # day is the start_day
    barr=[]
    TRADING_HOURS=23
    while day < day_end:
        ti.next()
        day1=ti.yyyymmdd()
        utc_e = int(TradingDayIterator.local_ymd_to_utc(day1, 17,0,0))

        # get start backwards for starting on a Sunday
        utc_s = utc_e - TRADING_HOURS*3600
        day=datetime.datetime.fromtimestamp(utc_s).strftime('%Y%m%d')

        i=np.searchsorted(bar[:, 0], float(utc_s)-1e-6)
        j=np.searchsorted(bar[:, 0], float(utc_e)-1e-6)
        bar0=bar[i:j,:]  # take the bars in between the first occurance of 18:00:00 (or after) and the last occurance of 17:00:00 or before

        N = (utc_e-utc_s)/bar_sec  # but we still fill in each bar
        ix_utc=((bar0[:,0]-float(utc_s))/bar_sec+1e-9).astype(int)
        bar_utc=np.arange(utc_s+bar_sec, utc_e+bar_sec, bar_sec) # bar time will be time of close price, as if in prod

        print 'getting bar ', day+'-18:00', day1+'-17:00', ' , got ', j-i, 'bars'
        # start to construct bar
        if j<=i :
            print ' NO bars found, skipping'
        else :
            bar_arr=[]
            bar_arr.append(bar_utc.astype(float))

            # construct the log returns for each bar, fill in zeros for gap
            #lpx_open=np.log(bar0[:,2])
            lpx_open=np.log(np.r_[last_close_px,bar0[:-1,5]])
            lpx_hi=np.log(bar0[:,3])
            lpx_lo=np.log(bar0[:,4])
            lpx_close=np.log(bar0[:,5])
            lpx_vwap=np.log(bar0[:,6])
            lr=lpx_close-lpx_open
            lr_hi=lpx_hi-lpx_open
            lr_lo=lpx_lo-lpx_open
            lr_vw=lpx_vwap-lpx_open

            # remove bars having abnormal return, i.e. circuit break for ES
            # with 9999 prices
            MaxLR=0.2
            ix1=np.nonzero(np.abs(lr)>=MaxLR)[0]
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_hi)>=MaxLR)[0])
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_lo)>=MaxLR)[0])
            ix1=np.union1d(ix1,np.nonzero(np.abs(lr_vw)>=MaxLR)[0])
            if len(ix1) > 0 :
                print 'warning: removing ', len(ix1), 'ticks exceed MaxLR (lr/lo/hi/vw) ', zip(lr[ix1],lr_hi[ix1],lr_lo[ix1],lr_vw[ix1])
                lr[ix1]=0
                lr_hi[ix1]=0
                lr_lo[ix1]=0
                lr_vw[ix1]=0

            # the trade volumes for each bar, fill in zeros for gap
            vlm=bar0[:,7]
            vb=bar0[:,8]
            vs=np.abs(bar0[:,9])
            vbs=vb-vs

            for v0, vn in zip([lr,lr_hi,lr_lo,lr_vw,vlm,vbs], ['lr','lr_hi','lr_lo','lr_vw','vlm','vbs']) :
                nix=np.nonzero(np.isnan(v0))[0]
                nix=np.union1d(nix, np.nonzero(np.isinf(np.abs(v0)))[0])
                if len(nix) > 0 :
                    print 'warning: removing ', len(nix), ' nan/inf ticks for ', vn
                    v0[nix]=0
                b0=np.zeros(N).astype(float)
                b0[ix_utc]=v0
                bar_arr.append(b0.copy())
         
            # get the last trade time, this is needs to be
            ltt=np.empty(N)*np.nan
            ltt[ix_utc]=bar0[:,1]
            df=pd.DataFrame(ltt)
            df.fillna(method='ffill',inplace=True)
            if not np.isfinite(ltt[0]) :
                ptt=0 #no previous trading detectable
                if i > 0 : #make some effort here
                    ptt=bar[i-1,1]
                    if not np.isfinite(ptt) :
                        ptt=0
                df.fillna(ptt,inplace=True)
            bar_arr.append(ltt)

            # get the last price, as a debugging tool
            lpx=np.empty(N)*np.nan
            lpx[ix_utc]=bar0[:,5]
            df=pd.DataFrame(lpx)
            df.fillna(method='ffill',inplace=True)
            if not np.isfinite(lpx[0]) :
                df.fillna(last_close_px,inplace=True)
            bar_arr.append(lpx)
            barr.append(np.array(bar_arr).T.copy())
            last_close_px=lpx[-1]

        day=day1

    return np.vstack(barr), trd_day_start, trd_day_end

def get_inc_idx(ts) :
    """
    This gets a index into the ts that is strictly
    increasing. i.e. t_i > t_j, for all j<i
    Used in fitering IB csv file, where timestamp
    could get back due to repeated downloading
    of same period.  
    ts: a 1d array
    return: index into ts that is increasing
    """

    N = len(ts)
    assert N > 0

    ixa=np.arange(N)
    dts = ts[1:]-ts[:-1]
    nix = np.nonzero(dts<=0)[0]
    if len(nix) == 0 :
        return ixa

    dix = np.array([])
    ix0 = nix[0]
    for i0 in np.arange(len(nix)) :
        # filter out index starting from i0+1
        # to where it recovers
        if nix[i0]<ix0 :
            continue

        ix1 = np.nonzero(ts > ts[nix[i0]])[0]
        if len(ix1) ==0:
            # delete all from nix[i0]+1 to end
            dix = np.r_[dix, np.arange(nix[i0]+1,N,1)]
            break
        else :
            # delete all from nix[i0]+1 to ix1[0] (exclusive)
            dix = np.r_[dix, np.arange(nix[i0]+1, ix1[0],1)]
            ix0 = ix1[0]

    if len(dix) > 0 :
        ixa = np.delete(ixa, dix)
    return ixa

def bar_by_file_ib(fn,bid_ask_spd,bar_qt=None,bar_trd=None) :
    """ 
    _qt.csv and _trd.csv are expected to exist for the given fn
    """
    import pandas as pd
    if bar_qt is None :
        bar_qt=np.genfromtxt(fn+'_qt.csv',delimiter=',',usecols=[0,1,2,3,4]) #, dtype=[('utc','i8'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8')])
    if bar_trd is None :
        bar_trd=np.genfromtxt(fn+'_trd.csv',delimiter=',',usecols=[0,1,2,3,4,5,6,7]) #,dtype=[('utc','i8'),('open','<f8'),('high','<f8'),('low','<f8'),('close','<f8'),('vol','i8'),('cnt','i8'),('wap','<f8')])

    # use quote as ref
    nqt =  bar_qt.shape[0]
    assert nqt > 3,  'too few bars found at ' + fn
    
    # make sure the time stamps strictly increasing
    qix=get_inc_idx(bar_qt[:,0])
    tix=get_inc_idx(bar_trd[:,0])
    bar_qt = bar_qt[qix,:]
    bar_trd = bar_trd[tix,:]

    # also make sure starts from an hour that is less than 16 o'clock
    # this is to be consistent with Reuters data.  IB data
    # file starts from start day's previous day's 18:00, while
    # Reuters KDB data file starts from start day's 0am. 
    qts = bar_qt[:,0]
    ix=0
    dt=datetime.datetime.fromtimestamp(qts[0])
    if dt.hour > 16 :
        t0 = qts[0] + (24-dt.hour)*3600
        ix=np.searchsorted(qts, t0)
        print 'cutting out leading ', ix , ' bars'
        bar_qt=bar_qt[ix:,:]
        bar_trd=bar_trd[ix:,:]

    qts=bar_qt[:,0]
    tts=bar_trd[:,0]

    assert len(np.nonzero(qts[1:]-qts[:-1]<0)[0]) == 0, 'quote time stamp goes back'
    assert len(np.nonzero(tts[1:]-tts[:-1]<0)[0]) == 0, 'trade time stamp goes back'

    tix=np.searchsorted(tts,qts)
    # they should be the same, otherwise, patch the different ones
    ix0=np.nonzero(tts[tix]-qts!=0)[0]
    if len(ix0) != 0 : 
        print len(ix0), ' bars mismatch!'
    ts=bar_trd[tix,:]
    ts[tix[ix0],5]=0
    ts[tix[ix0],6]=0
    ts[tix[ix0],7]=bar_qt[ix0,4].copy()

    vwap=ts[:,7].copy()
    vol=ts[:,5].copy()
    vb=vol.copy()
    vs=vol.copy()
    utc_ltt=ts[:,0]
    if len(ix0) > 0 : 
        utc_ltt[ix0]=np.nan
        df=pd.DataFrame(utc_ltt)
        df.fillna(method='ffill',inplace=True)

    """ 
    # for those bar without price movements, calculate the volume by avg trade price 
    ixe=np.nonzero(bar_qt[:,1]-bar_qt[:,4]==0)[0]
    #pdb.set_trace()
    vb[ixe]=np.clip((ts[ixe,7]-(bar_qt[ixe,4]-bid_ask_spd/2))/bid_ask_spd*ts[ixe,5],0,1e+10)
    vs[ixe]=ts[ixe,5]-vb[ixe]

    ixg=np.nonzero(bar_qt[:,1]-bar_qt[:,4]<0)[0]
    vs[ixg]=0
    ixl=np.nonzero(bar_qt[:,1]-bar_qt[:,4]>0)[0]
    vb[ixl]=0
    """
    spd=bid_ask_spd*np.clip(np.sqrt((bar_qt[:,2]-bar_qt[:,3])/bid_ask_spd),1,2)
    mid=(bar_qt[:,2]+bar_qt[:,3])/2
    #mid=np.mean(bar_qt[:,1:5], axis=1)

    vb=np.clip((vwap-(mid-spd/2))/spd,0,1)*vol
    vs=vol-vb

    bar=np.vstack((bar_qt[:,0],utc_ltt,bar_qt[:,1:5].T,vwap,vol,vb,vs)).T
    return bar_qt, bar_trd, bar 

def get_contract_bar(symbol, contract, yyyy) :
    """
    date,ric,timeStart,lastTradeTickTime,open,high,low,close,avgPrice,vwap,volume,buyvol,sellvol
    2017.12.20,TYH8,00:00:00.000,00:00:00.093,123.75,123.75,123.75,123.75,123.75,123.75,239,239,0
    2017.12.21,TYH8,00:00:00.000,00:00:00.082,123.578,123.578,123.578,123.578,123.578,123.578,160,160,0
    """
    import glob
    bar_dir=symbol
    fn=bar_dir+'/'+contract+'_'+yyyy+'*.csv*'
    f=glob.glob(fn)
    assert len(f) == 1, 'problem finding '+ fn + ', got ' + str(f)
    fn=f[0]
    return bar_by_file(fn)

### repo bar columns of kdb
repo_col={'utc':0, 'lr':1, 'vol':2, 'vbs':3, 'lrhl':4, 'vwap':5, 'ltt':6, 'lpx':7}
utcc=repo_col['utc']
lrc=repo_col['lr']
volc=repo_col['vol']
vbsc=repo_col['vbs']
lrhlc=repo_col['lrhl']
vwapc=repo_col['vwap']
lttc=repo_col['ltt']
lpxc=repo_col['lpx']

def gen_bar0(symbol,year,check_only=False, ext_fields=False, ibbar=True, spread=None, bar_sec=5) :
    year =  str(year)  # expects a string
    if ibbar :
        fn=glob.glob('hist/'+symbol+'/'+symbol+'*_[12]*_qt.csv*')
    else :
        fn=glob.glob(symbol+'/'+symbol+'*_[12]*.csv*')

    ds=[]
    de=[]
    fn0=[]
    for f in fn :
        if os.stat(f).st_size < 500 :
            print '\t\t\t ***** ', f, ' is too small, ignored'
            continue
        ds0=f.split('/')[-1].split('_')[1]
        if ds0[:4]!=year :
            continue
        de0=f.split('/')[-1].split('_')[2].split('.')[0]
        ds.append(ds0)
        de.append(de0)
        fn0.append(f)
    ix=np.argsort(ds)
    dss=np.array(ds)[ix]
    des=np.array(de)[ix]
    fn = np.array(fn0)[ix]
    # check that they don't overlap
    for dss0, des0, f0, f1 in zip(dss[1:], des[:-1], fn[1:], fn[:-1]):
        if des0>dss0 :
            raise ValueError('time overlap! ' + '%s(%s)>%s(%s)'%(des0,f0,dss0,f1))

    if check_only :
        print year, ': ', len(fn0), ' files'
        return
    if not ext_fields :
        num_col=5 # utc, lr, volume, buy-sell, high_rt-low_rt
    else :
        num_col=8 # adding spd vol, last_trd_time, last_close_px
    bar_lr=np.array([]).reshape(0,num_col)
    if len(fn0) == 0 :
        return bar_lr
    for f in fn :
        if f[-3:]=='.gz' :
            print 'gunzip ', f
            os.system('gunzip '+f)
            f = f[:-3]
        print 'reading bar file ',f
        if ibbar :
            _,_,b=bar_by_file_ib(f[:-7],spread)
        else :
            b=bar_by_file(f)
        ba, sd, ed = write_daily_bar(b,bar_sec=bar_sec)
        bt=ba[:,0]
        lr=ba[:,1]
        vl=ba[:,5]
        vbs=ba[:,6]
        # add a volatility measure here
        lrhl=ba[:,2]-ba[:,3]
        if not ext_fields :
            bar_lr=np.r_[bar_lr, np.vstack((bt,lr,vl,vbs,lrhl)).T]
        if ext_fields :
            vwap=ba[:,4]
            ltt=ba[:,7]
            lpx=ba[:,8]
            bar_lr=np.r_[bar_lr, np.vstack((bt,lr,vl,vbs,lrhl,vwap,ltt,lpx)).T]

    return bar_lr

def gen_bar(symbol, year_s=1998, year_e=2018, check_only=False, ext_fields=True) :
    ba=[]
    years=np.arange(year_s, year_e+1)
    for y in years :
        try :
            barlr=gen_bar0(symbol,str(y),check_only=check_only,ext_fields=ext_fields)
            if len(barlr) > 0 :
                ba.append(barlr)
        except :
            traceback.print_exc()
            print 'problem getting ', y, ', continue...'

    if check_only :
        return
    fn=symbol+'_bar_'+str(year_s)+'_'+str(year_e)
    if ext_fields :
        fn+='_ext'
    np.savez_compressed(fn,bar=ba,years=years)

if __name__ == '__main__' :
    import sys
    symbol_list= sys.argv[1:-2]
    sday=sys.argv[-2]
    eday=sys.argv[-1]
    get_future_bar_fix(symbol_list, sday, eday)

