#pragma once

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdexcept>
#include <string>
#include <cmath>

#define SMOD(a,b) (((a) % (b) + (b)) % (b)) // C modulo could get negative

namespace utils {

class TimeUtil {
public:

    //
    // utc and string conversions
    //
    static uint64_t string_to_frac_UTC(const char* str_buf, int frac_decimals=0, const char* fmt_str="%Y%m%d-%H:%M:%S", bool string_in_gmt = false);
       // expects a UTC time string like YYYYMMDD-HH:MM:SS.ssssss
       // the .ssssss part, if found, is taken as a fraction.
       // returns the utc * frac_mul + fraction * frac_mul,
       // where frac_mul = 10**frac_decimals
       // set frac_decimals to be 0 to get a whole second (fraction dropped, NOT rounded)
       // If the string is in GMT, set the string_in_gmt to be true.  
       // Otherwise, it is taken as a local time.
       //
       // For example, 
       // 1. string_to_frac_UTC( "20201123-09:30:00.987654", 3)
       //         returns utc of "20201123-09:30:00" * 1000 + 987
       // 2. string_to_frac_UTC( "20201123-09:30:00.987654", 0)
       //         returns utc of "20201123-09:30:00"
       // 3. string_to_frac_UTC( "20201123-09:30:00", 3)
       //         returns utc of "20201123-09:30:00" * 1000

    static size_t frac_UTC_to_string(uint64_t utc_frac_mul, char* char_buf, int buf_size, int frac_decimals=0, const char* fmt_str="%Y%m%d-%H:%M:%S", bool string_in_gmt=false);
       // expects utc_frac_mul = whole_seconds * frac_mul + frac_seconds
       // where frac_mul=10**frac_decimals
       // ret YYYYMMDD-HHMMSS.sss, where 0.sss = frac_seconds/frac_mul
       // set the utc_frac_mul to 0 for the current time
       // If string_in_gmt is true, return string in GMT
       // If string_in_gmt is false, return string in local time

    static std::string frac_UTC_to_string(uint64_t utc_frac_mul=0, int frac_decimals=0, const char* fmt_str="%Y%m%d-%H:%M:%S", bool string_in_gmt = false);
    static struct tm int_to_tm_UTC(time_t ts);
    static std::string string_local_to_gmt(const char* str_buf, int frac_decimals=0, const char* fmt_str="%Y%m%d-%H:%M:%S");
    static std::string string_gmt_to_local(const char* str_buf, int frac_decimals=0, const char* fmt_str="%Y%m%d-%H:%M:%S");
    static int int_to_string_second_UTC(time_t sec, char*char_buf, int buf_size);

    //
    // trading time and trading days
    //
    static bool isTradingTime_TM(struct tm tm_data, int start_hour, int start_min, int end_hour, int end_min);
       // trading ends on friday, end_hour, end_min (inclusive, last minute is end_min - 1)
       // If start hour is negative, 
       //      trading starts on Sunday start_hour, start_min inclusive, first minute is start_min)
       // else 
       //      trading starts on Monday. 

    static bool isTradingTime(time_t ts, int start_hour = -6, int start_min = 0, int end_hour = 17, int end_min = 0);

    static std::string tradingDay(time_t utc_second = 0, int start_hour = -6, int start_min = 0,
                            int end_hour = 17, int end_min = 0, int day_offset=0, int snap = 0) ;
       // get the trading day of utc_second as defined by start/end time.
       // if day_offset is set, it returns previous or future trading days (non trading
       // days are not counted in the offset days).
       // if utc_second is not a trading time: 
       //    snap = 0, returns empty string
       //    snap = 1, returns the previous trading day
       //    snap = 2, returns the next trading day
       // returns the trading day in yyyymmdd
       //
       // Note over night sessions should be represented with negative start_hour and
       // a positive end_hour.

    static std::string curTradingDay1(); // trading day of current time with snap = 1, see tradingDay()
    static std::string curTradingDay2(); // trading day of current time with snap = 2, see tradingDay()

    static time_t startUTC(time_t cur_utc = 0, int start_hour = -6, int start_min = 0,
                  int end_hour = 17, int end_min = 0, int day_offset=0, int snap = 2);
        // get the starting utc of the trading day given by cur_utc

    //
    // timer
    //
    static uint64_t cur_micro();
    static time_t   cur_utc();
    static uint64_t micro_sleep(uint64_t micro);

    // 
    // mockings
    //
    static void set_cur_time_micro(uint64_t cur_micro);
    static void unset_cur_time_micro();

private:
#ifndef NUMPY_COMPILE
    static uint64_t CurTimeMicro;  // static instance for time mocking
#endif
};

//
// inlines
//
inline std::string TimeUtil::frac_UTC_to_string(uint64_t utc_frac_mul, int frac_decimals, const char* fmt_str, bool string_in_gmt) {
    char buf[32];
    frac_UTC_to_string(utc_frac_mul, buf, sizeof(buf), frac_decimals, fmt_str, string_in_gmt);
    return std::string(buf);
}

inline std::string TimeUtil::string_local_to_gmt(const char* str_buf, int frac_decimals, const char* fmt_str) {
    return frac_UTC_to_string( 
            string_to_frac_UTC(str_buf, frac_decimals, fmt_str, false), 
            frac_decimals, 
            fmt_str, 
            true);
}

inline std::string TimeUtil::string_gmt_to_local(const char* str_buf, int frac_decimals, const char* fmt_str) {
    return frac_UTC_to_string( 
            string_to_frac_UTC(str_buf, frac_decimals, fmt_str, true), 
            frac_decimals, 
            fmt_str,
            false);
}

inline struct tm TimeUtil::int_to_tm_UTC(time_t ts) {
    struct tm tmv;
    localtime_r(&ts, &tmv);
    return tmv;
}

inline bool TimeUtil::isTradingTime(time_t ts, int start_hour, int start_min, int end_hour, int end_min) {
    if ( (start_min == end_min) && ((end_hour - start_hour)%24 == 0)) {
        // 24 hour always on!
        return true;
    }
    return isTradingTime_TM(int_to_tm_UTC(ts), start_hour, start_min, end_hour, end_min);
};

inline int TimeUtil::int_to_string_second_UTC(time_t sec, char*char_buf, int buf_size) {
    return frac_UTC_to_string(sec, char_buf, buf_size, 0, "%Y%m%d,%H:%M:%S");
}

inline void TimeUtil::set_cur_time_micro(uint64_t cur_micro) {
#ifndef NUMPY_COMPILE
    TimeUtil::CurTimeMicro = cur_micro;
#endif
}

inline void TimeUtil::unset_cur_time_micro() {
#ifndef NUMPY_COMPILE
    TimeUtil::CurTimeMicro = 0;
#endif
}

inline uint64_t TimeUtil::cur_micro() {
#ifndef NUMPY_COMPILE
    if (__builtin_expect(TimeUtil::CurTimeMicro != 0, 0)) {
        return TimeUtil::CurTimeMicro;
    }
#endif
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((uint64_t)tv.tv_sec)*1000000ULL + tv.tv_usec;
};

inline time_t TimeUtil::cur_utc() {
    return time_t(TimeUtil::cur_micro()/1000000ULL);
}

inline std::string TimeUtil::curTradingDay1() {
    return tradingDay(0,-6,0,17,0,0,1);
}

inline std::string TimeUtil::curTradingDay2() {
    return tradingDay(0,-6,0,17,0,0,2);
}

}
