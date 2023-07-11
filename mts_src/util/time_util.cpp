#include "time_util.h"

namespace utils {
    uint64_t TimeUtil::string_to_frac_UTC(const char* str_buf, int frac_decimals, const char* fmt_str, bool string_in_gmt) {
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

       if (frac_decimals<0 || frac_decimals>9) {
           throw std::runtime_error("frac_decimals out-of-range: " + std::to_string(frac_decimals));
       }

       if (fmt_str == NULL) {
           // allow a NULL as default
           fmt_str = "%Y%m%d-%H:%M:%S";
       }
       uint64_t frac_mul = 1;
       for (int i=frac_decimals; i>0; --i, frac_mul*=10);
       struct tm time_tm;
       memset(&time_tm, 0, sizeof(struct tm));
       char* p = strptime(str_buf, fmt_str, &time_tm);
       if (!p) {
           return 0;
       }
       time_tm.tm_isdst = -1;

       uint64_t tsf;
       if (string_in_gmt) {
           //adjust offset of local
           //this trick doesn't work, due to the
           //offset on 1970/1/2 maybe different
           //due to daylight saving. 
           //tsf -= string_to_frac_UTC("19700102-00:00:00");
           //tsf += (24*3600);
           tsf = (uint64_t)timegm(&time_tm);
       } else {
           tsf = (uint64_t)mktime(&time_tm);
       }
       tsf *= frac_mul;
       if(*p=='.') {
           double f = std::stod("0"+std::string(p));
           tsf+=(uint64_t)(f*(double)frac_mul);
       }
       return tsf;
    };

    size_t TimeUtil::frac_UTC_to_string(uint64_t utc_frac_mul, char* char_buf, int buf_size, int frac_decimals, const char* fmt_str, bool string_in_gmt) {
       // expects utc_frac_mul = whole_seconds * frac_mul + frac_seconds
       // where frac_mul=10**frac_decimals
       // ret YYYYMMDD-HHMMSS.sss, where 0.sss = frac_seconds/frac_mul
       // set the utc_frac_mul to 0 for the current time
       // If string_in_gmt is true, return string in GMT
       // If string_in_gmt is false, return string in local time

       if (frac_decimals<0 || frac_decimals>9) {
           throw std::runtime_error("frac_decimals out-of-range: " + std::to_string(frac_decimals));
       }
       if (fmt_str == NULL) {
           // allow NULL as default as well.
           fmt_str = "%Y%m%d-%H:%M:%S";
       }
       uint64_t frac_mul = 1;
       for (int i=frac_decimals; i>0; --i, frac_mul*=10);

       if (utc_frac_mul==0) {
           // use current time
           utc_frac_mul = cur_micro();
           int decimal_adj = frac_decimals-6;
           while (decimal_adj > 0) {
               utc_frac_mul*=10;
               --decimal_adj;
           }
           while (decimal_adj < 0) {
               utc_frac_mul/=10;
               ++decimal_adj;
           }
       }

       time_t sec = (time_t) (utc_frac_mul/frac_mul);
       struct tm t;
       memset(&t, 0, sizeof(struct tm));
       size_t bytes = 0;

       bool ok;
       if (__builtin_expect(string_in_gmt,0))
           ok = (gmtime_r(&sec, &t) !=NULL);
       else
           ok = (localtime_r(&sec, &t) != NULL);
       if (ok) {
           bytes = strftime(char_buf, buf_size, fmt_str, &t);
           double frac = (double)(utc_frac_mul%frac_mul)/(double)frac_mul;
           std::string fmt = "%.0"+std::to_string(frac_decimals)+"lf";
           char fbuf[16];
           snprintf(fbuf, sizeof(fbuf), fmt.c_str(), frac);
           bytes += snprintf(char_buf+bytes, buf_size-bytes, "%s",fbuf+1);
       }
       return bytes;
   };

    bool TimeUtil::isTradingTime_TM(struct tm tm_data, int start_hour, int start_min, int end_hour, int end_min) {
       // trading ends on friday, end_hour, end_min (inclusive, last minute is end_min - 1)
       // If start hour is negative, 
       //      trading starts on Sunday start_hour, start_min inclusive, first minute is start_min)
       // else 
       //      trading starts on Monday. 

      if (tm_data.tm_wday == 6) {
         return false;
      }
      if (tm_data.tm_wday == 5) {
         if ((tm_data.tm_hour > SMOD(end_hour, 24)) ||
               ((tm_data.tm_hour == SMOD(end_hour, 24)) &&
                (tm_data.tm_min >= end_min))) {
             return false;
         };
      }
      else if (tm_data.tm_wday == 0) {
          if (start_hour > 0) {
              return false;
          } else {
              // sunday open
              if ((tm_data.tm_hour < SMOD(start_hour, 24)) ||
                  ((tm_data.tm_hour == SMOD(start_hour, 24)) && 
                   (tm_data.tm_min < start_min))) {
                  return false;
              };
          };
      }
      else if (tm_data.tm_wday == 1) {
          if (start_hour > 0) {
              if ((tm_data.tm_hour < start_hour) ||
                   ((tm_data.tm_hour == start_hour) && 
                    (tm_data.tm_min < start_min))) {
                  return false;
              }
          }
      }

      // its a weekday, check if it is out side of trading hours
      int tmin = tm_data.tm_hour*60 + tm_data.tm_min;
      int tend = end_hour*60 + end_min;
      int tstart = SMOD(start_hour, 24) *60 + start_min;
      return SMOD(tmin-tstart, 60*24) < SMOD(tend-tstart, 60*24);
    };

    std::string TimeUtil::tradingDay(time_t utc_second, int start_hour, int start_min,
                                     int end_hour, int end_min, int day_offset, int snap) 
    {
       // get the trading day of utc_second as defined by start/end time.
       // if day_offset is set, it returns previous or future trading days (non trading
       // days are not counted in the offset days).
       // if utc_second is not a trading time: 
       //    snap = 0, returns empty string
       //    snap = 1, returns the previous trading day
       //    snap = 2, returns the next trading day
       //
       // Note over night sessions should be represented with negative start_hour and
       // a positive end_hour.

       if (utc_second == 0) {
           // current second
           utc_second = utils::TimeUtil::cur_utc();
       }
       while (! isTradingTime(utc_second, start_hour, start_min, end_hour, end_min)) {
           if (snap == 0) 
               return "";
           utc_second += ((snap*2-3)*3600);
       }

       if (day_offset != 0) {
           int offset_sign = std::abs(day_offset)/day_offset;
           do {
               utc_second += (offset_sign * 3600 * 24);
           } while (!isTradingTime(utc_second, start_hour, start_min, end_hour, end_min));
           return tradingDay(utc_second, start_hour, start_min, end_hour, end_min, day_offset - offset_sign, snap);
       }


       // if start_hour is negative, i.e. over-night session
       // trading day goes forward during over-night period
       struct tm tmsec = int_to_tm_UTC(utc_second);
       if (tmsec.tm_hour*60+tmsec.tm_min > end_hour*60+end_min) {
           utc_second += (24*3600);
       }
       return frac_UTC_to_string(utc_second, 0, "%Y%m%d");
    }

    time_t TimeUtil::startUTC(time_t utc_second, int start_hour, int start_min,
                              int end_hour, int end_min, int day_offset, int snap)
    {
        auto trd_day = TimeUtil::tradingDay(utc_second, start_hour, start_min, 
                end_hour, end_min, day_offset, snap); // in yyyymmdd
        time_t utc = TimeUtil::string_to_frac_UTC(trd_day.c_str(),0,"%Y%m%d");
        utc += (start_hour*3600);
        utc += (start_min*60);
        return utc;
    }

    uint64_t TimeUtil::micro_sleep(uint64_t micro) {
       struct timespec req, rem;
       req.tv_sec=(time_t)(micro/1000000ULL);
       req.tv_nsec=(long)((micro%1000000ULL)*1000ULL);
       int ret = nanosleep(&req, &rem);
       if (ret != 0)
           return rem.tv_sec*1000000ULL + (uint64_t)rem.tv_nsec/1000ULL;
       return 0;
    }
#ifndef NUMPY_COMPILE
    uint64_t TimeUtil::CurTimeMicro  = 0;  // init to use 
#endif                                     // 

}
