import dateutil
import datetime
import time
import pytz
import numpy as np

def str_to_utc(YYYYMMDDHHMMSS_str, tz_str = '+0000') :
    """
    YYYYMMDDHHMMSS: the HHMMSS can be omitted 
    tz_str: '+0800' for CN
            '+0000' for GMT (default)
            '-0500' for EST
    return utc of the time str taken as tz_str
    raise except if failed
    """
    if tz_str == 'CN' :
        tz_str = '+0800'
    elif tz_str == 'GMT' :
        tz_str = '+0000'
    dt = dateutil.parser.parse(YYYYMMDDHHMMSS_str + ' ' + tz_str)
    dt0 = dateutil.parser.parse('19700101000000 +0000')
    return (dt - dt0).total_seconds()

def dt_from_utc_country(utc, tz_str) :
    """
    utc is the posix time stamp
    tz_str could be 
           'CN'
           'US/Eastern'
           'EST5EDT'
           'GMT'
           but it cannot be '+0800'
    return datetime object for utc with tz set to tz_str
    Raise exception if failed
    
    """
    try :
        tz = pytz.timezone(pytz.country_timezones(tz_str)[0])
    except :
        tz = pytz.timezone(tz_str)
    return datetime.datetime.fromtimestamp(utc, tz)

class TradingDayIterator :
    def __init__(self, start_day) :
        """
        start_day: YYYYMMDD, as a starting day, if not a trading day
                   then start_day is previous trading day

        """
        self.dt = datetime.datetime.strptime(start_day, '%Y%m%d')
        while not self.is_trading_day() :
            self.next_dt(-1)

    def next_str(self, d) :
        self.next_dt(d)
        return self.cur_day_str()

    def cur_day_str(self) :
        return self.dt.strftime('%Y%m%d')

    def next_dt(self, d) :
        """
        d: +1 next 1 day, -1 prev 1 day, skip non-trading days
        """
        while True :
            self.dt += datetime.timedelta(d)
            if self.is_trading_day() :
                break

        return self.dt

    def is_trading_day(self) :
        return self.dt.weekday() < 5

def hhmmss_bar_time(start_hour, start_min, start_sec, end_hour, end_min, end_sec, bar_sec) :
    """
    generate hhmmss integer from start (inclusive) to end (inclusive)
    """
    h = start_hour
    m = start_min
    s = start_sec
    t1 = end_hour*10000 + end_min*100 + end_sec
    bar = []
    while True :
        t0 = h*10000 + m*100 + s
        if t0 > t1 :
            break
        bar.append(t0)
        s += bar_sec
        if s>= 60 :
            m += (s/60)
            s = s % 60
            if m >= 60 :
                h += (m/60)
                m = m % 60
    return np.array(bar).astype(int)

def hhmmss_to_hms (hhmmss) :
    hs = hhmmss/10000
    ms = (hhmmss%10000)/100
    ss = hhmmss - hs*10000 - ms*100
    return hs, ms, ss

def hhmmss_diff_sec(hhmmss0, hhmmss1) :
    hs0, ms0, ss0 = hhmmss_to_hms(int(hhmmss0))
    hs1, ms1, ss1 = hhmmss_to_hms(int(hhmmss1))
    return hs1*3600 + ms1*60 + ss1 - (hs0*3600 + ms0*60 + ss0)

def hhmmss_from_utc(ts, TZ) :
    dt0 = dt_from_utc_country(ts, TZ)
    hhmmss = dt0.hour * 10000 + dt0.minute*100 + dt0.second + dt0.microsecond/1000000.0
    yymmdd = dt0.year * 10000 + dt0.month*100 + dt0.day
    return hhmmss, yymmdd
