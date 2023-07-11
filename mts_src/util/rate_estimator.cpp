#include "rate_estimator.h"
#include "plcc/PLCC.hpp"
#include "csv_util.h"
#include <stdexcept>

//  compound assignment with ‘volatile’-qualified left operand is deprecated [-Werror=volatile]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvolatile"

namespace utils {

    ////////////////////////////////////
    // implementation of RateEstimator
    ////////////////////////////////////

    RateEstimator::RateEstimator(const std::vector<std::pair<double, int>>& rate_seconds, int bucket_seconds): _bucket_seconds(bucket_seconds), _lock(false){
        if (rate_seconds.size() == 0) {
            logError("RateEstimator: creation with zero length vector");
            throw std::runtime_error("RateEstimator: create with zero length vector");
        }
        if (_bucket_seconds < 1) {
            logError("RateEstimator: bucket seconds non-positive %d", _bucket_seconds);
            throw std::runtime_error("RateEstimator: bucket_seconds non-positive!");
        }
        _event_len = 0;
        const int MIN_BUCKETS = 2; // enforce a minimum numbero of buckets
        int prev_sec = 0;
        for (const auto& rate_sec: rate_seconds) {
            auto [rate, sec] = rate_sec;
            // check on the bucket
            if ((sec/_bucket_seconds*_bucket_seconds != sec) || (sec/_bucket_seconds < MIN_BUCKETS)) {
                logError("RateEstimator: time window(%d) not multiple of bucket (%d) or too small", 
                         sec, _bucket_seconds);
                throw std::runtime_error("RateEstimator: time window not multiple or too small");
            }
            sec /= _bucket_seconds;
            // enforce an strictly increasing order in sec
            if (sec <= prev_sec) {
                logError("RateEstimator: construction sec non-increasing: %d, %d", prev_sec, (int)sec);
                throw std::runtime_error("RateEstimator: non-increasing seconds");
            }
            prev_sec = (int)sec;
            _event_len = _MAX_(_event_len, sec);
            _twnd_count_limit.push_back(std::make_tuple((int)sec, 0, (int) (rate*sec*_bucket_seconds+0.5)));
        }

        if (_event_len <= 0) {
            logError("RateEstimator: non-positive event length");
            throw std::runtime_error("RateEstimator: non-positive event length");
        }
        _event = (volatile unsigned int*) malloc (sizeof(unsigned int)*_event_len);
        if (!_event) {
            logError("RateEstimator: malloc of %d", (int) sizeof(unsigned int)*_event_len);
            throw std::runtime_error(std::string("RateEstimator: malloc of ") + std::to_string((int) sizeof(unsigned int)*_event_len));
        }
        memset((void*)_event, 0, sizeof(unsigned int)*_event_len);
        _latest_second = TimeUtil::cur_utc()/_bucket_seconds;

        //logInfo("Created as %s", toString().c_str());
    }

    RateEstimator::RateEstimator(const RateEstimator& re)
    : _bucket_seconds(re._bucket_seconds), 
      _twnd_count_limit(re._twnd_count_limit),
      _event_len(re._event_len),
      _event(nullptr),
      _latest_second(re._latest_second),
      _lock(false)
    {
        _event = (volatile unsigned int*)malloc( _event_len * sizeof(unsigned int));
        if (!_event) {
            logError("RateEstimator: malloc of %d", (int) sizeof(unsigned int)*_event_len);
            throw std::runtime_error(std::string("RateEstimator copy constructor: malloc of ") + std::to_string((int) sizeof(unsigned int)*_event_len));
        }
        memcpy((void*)_event, (const void*) re._event, sizeof(unsigned int)*_event_len);
    }

    void RateEstimator::operator=(const RateEstimator& re) {
        // just copy everything, not thread safe
        _bucket_seconds = re._bucket_seconds;
        _twnd_count_limit = re._twnd_count_limit;
        _event_len = re._event_len;
        _latest_second = re._latest_second;

        if (_event) free ((void*)_event);
        _event = (volatile unsigned int*)malloc( _event_len * sizeof(unsigned int));
        if (!_event) {
            logError("RateEstimator: malloc of %d", (int) sizeof(unsigned int)*_event_len);
            throw std::runtime_error(std::string("RateEstimator copy constructor: malloc of ") + std::to_string((int) sizeof(unsigned int)*_event_len));
        }
        memcpy((void*)_event, (const void*) re._event, sizeof(unsigned int)*_event_len);
    }

    uint64_t RateEstimator::checkOnly(uint64_t cur_micro, int update_cnt) const {
        if (__builtin_expect(update_cnt < 0, 0)) {
            logError("RateEstimator: checkOnly(): negative update_cnt %d", update_cnt);
            throw std::runtime_error("RateEstimator checkOnly() negative update_cnt");
        }
        if (__builtin_expect(cur_micro == 0, 0)) {
            cur_micro = TimeUtil::cur_micro();
        }
        int cur_utc = (int) (cur_micro/1000000ULL/_bucket_seconds);
        if (__builtin_expect( cur_utc < _latest_second, 0)) {
            if (cur_utc < _latest_second - 1) {
                logError("RateEstimator: checkOnly time goes back - latest(%d):check(%d)",
                       (int) _latest_second*_bucket_seconds, (int) cur_utc*_bucket_seconds);
            }
            cur_utc = _latest_second;
        }
        // now check all the tuple
        int wait_sec = 0;
        {
            auto lock = SpinLock(_lock);
            for (const auto& tp: _twnd_count_limit) {
                auto [wnd_sec, count, limit] = tp;
                int update_cnt0 = update_cnt;

                // if update_cnt violates limit, log a warning and 
                // adjust it to the limit
                if (__builtin_expect(update_cnt > limit,0)) {
                    logError("RateEstimator: checkOnly update_cnt (%d) more than limit(%d), set it to limit", update_cnt, limit);
                    update_cnt0 = limit;
                }
                count = getCount_Unsafe(cur_utc, count, wnd_sec) + update_cnt0;
                if (__builtin_expect(count > limit, 0)) {
                    // look for the seconds to wait
                    auto utc0 = cur_utc - wnd_sec + 1;
                    int wait_sec0 = 0;
                    for (; (utc0 <= _latest_second) && (count > limit) ; ++utc0) {
                        ++wait_sec0;
                        count -= _event[utc0%_event_len];
                    }
                    wait_sec = _MAX_(wait_sec, wait_sec0);
                }
            }
        }
        if (__builtin_expect(wait_sec == 0, 1)) {
            return 0;
        }
        return (wait_sec*_bucket_seconds - (cur_micro/1000000ULL)%_bucket_seconds)*1000000ULL;
    }

    uint64_t RateEstimator::check(uint64_t cur_micro, int update_cnt) {
        if (__builtin_expect(cur_micro==0, 0)) {
            cur_micro = TimeUtil::cur_micro();
        }
        uint64_t wait_micro = checkOnly(cur_micro, update_cnt);
        if (__builtin_expect(wait_micro==0, 1)) {
            updateOnly(cur_micro, update_cnt, false);
            return 0;
        }
        return wait_micro;
    }

    uint64_t RateEstimator::updateOnly(uint64_t cur_micro, int update_cnt, bool ret_wait) {
        if (__builtin_expect(update_cnt <= 0, 0)) {
            logError("RateEstimator: updateOnly(): non-positive update_cnt %d", update_cnt);
            throw std::runtime_error("RateEstimator updateOnly() negative update_cnt");
        }
        if (__builtin_expect(cur_micro == 0, 0)) {
            cur_micro = TimeUtil::cur_micro();
        }
        int cur_utc = (int)(cur_micro/1000000ULL/_bucket_seconds);
        if (__builtin_expect( cur_utc < _latest_second, 0)) {
            if (cur_utc < _latest_second - 1) {
                logError("RateEstimator: updateOnly time goes back - latest(%d):check(%d)",
                       (int) _latest_second*_bucket_seconds, (int) cur_utc*_bucket_seconds);
            }
            cur_utc = _latest_second;
        }
        // update all the tuple
        int wait_sec = 0;
        {
            auto lock = SpinLock(_lock);
            for (auto& tp: _twnd_count_limit) {
                auto [wnd_sec, count, limit] = tp;
                int new_count = getCount_Unsafe(cur_utc, count, wnd_sec) + update_cnt;
                std::get<1>(tp) = new_count;
                if (__builtin_expect(ret_wait && (new_count > limit),0)) {
                    // figure out the wait
                    auto utc0 = cur_utc - wnd_sec + 1;
                    if (utc0 > _latest_second) {
                        wait_sec = _MAX_(wait_sec, wnd_sec);
                        continue;
                    }
                    int wait_sec0 = 0;
                    for (; (utc0 <= _latest_second) && (new_count > limit) ; ++utc0) {
                        ++wait_sec0;
                        new_count -= _event[utc0%_event_len];
                    }
                    wait_sec = _MAX_(wait_sec, wait_sec0);
                }
            }
            // fill in 0 for skipped buckets if any
            for (; _latest_second < cur_utc; ++_latest_second) {
                _event[(_latest_second+1) % _event_len] = 0;
            }
            // add the update_cnt to current bucket
            _event[_latest_second % _event_len] += update_cnt;
        }
        if (__builtin_expect(wait_sec == 0, 1)) {
            return 0;
        }
        return (wait_sec*_bucket_seconds - (cur_micro/1000000ULL)%_bucket_seconds)*1000000ULL;
    }

    // removes the counts starting from the _latest for as much as update_cnt
    void RateEstimator::removeOnly(int update_cnt) {
        if (__builtin_expect(update_cnt <= 0, 0)) {
            logError("RateEstimator: removeOnly(): non-positive update_cnt %d", update_cnt);
            throw std::runtime_error("RateEstimator removeOnly() negative update_cnt");
        }
        int rm_cnt = update_cnt;
        {
            auto lock = SpinLock(_lock);
            for (auto utc = _latest_second; (utc>_latest_second-_event_len) && (update_cnt>0); --utc) {
                auto& cnt (_event[utc%_event_len]);
                if ((int) cnt >= update_cnt) {
                    cnt -= update_cnt;
                    update_cnt = 0;
                    break;
                }
                update_cnt -= cnt;
                cnt = 0;
            }
            rm_cnt -= update_cnt;
            // update the cur_count
            for (auto& tp: _twnd_count_limit) {
                auto [wnd_sec, count, limit] = tp;
                std::get<1>(tp) = _MAX_(count - rm_cnt, 0);
            }
        }
    }

    // need to be locked, assuming cur_utc >= _latest_second
    int RateEstimator::getCount_Unsafe(int cur_utc, int cur_count, int wnd_sec) const {
        auto utc0 = _latest_second - wnd_sec + 1;
        if (__builtin_expect(cur_utc >= _latest_second + wnd_sec, 0)) {
            return 0;
        }
        auto utc1 = cur_utc - wnd_sec + 1;
        int count = cur_count;
        for (; utc0 < utc1; ++utc0) {
            count -= _event[utc0%_event_len];
        }
        return count;
    }

    void RateEstimator::updateRateLimit(const std::vector<double> & rates) {
        if (rates.size() != _twnd_count_limit.size()) {
            logError("RateEstimator: updateRateLimit mismatch shape, rate limit not updated!");
            return;
        }
        for (size_t i=0; i<rates.size(); ++i) {
            double new_rate = rates[i];
            if (new_rate < -1e-10) {
                logError("RateEstimator: updateRateLimit with negative rate %f, rate limit not updated!", new_rate);
                return;
            }
        }

        logInfo("Resetting rates: before: %s", toString().c_str());
        {
            auto lock = SpinLock(_lock);
            for (size_t i=0; i<rates.size(); ++i) {
                auto [twnd, count, limit] = _twnd_count_limit[i];
                double new_rate = rates[i];
                int new_limit = int(twnd*_bucket_seconds*new_rate+0.5);
                std::get<2>(_twnd_count_limit[i]) = new_limit;
            }
        }
        logInfo("Resetting rates: after: %s", toString().c_str());
    }

    // this resets all the counts
    void RateEstimator::resetAll() {
        {
            auto lock = SpinLock(_lock);
            for (auto& tcl: _twnd_count_limit) {
                std::get<1>(tcl) = 0;
            }
            memset((void*)_event, 0, sizeof(unsigned int)*_event_len);
        }
    }

    std::string RateEstimator::toString(time_t cur_utc) const {
        // output a csv format of
        // utc, _bucket_seconds, twnd, rate_limit, rate, percent_limit
        char buf[512];
        size_t bytes=0;
        bytes = snprintf(buf, sizeof(buf), "%d, %d", (int) cur_utc, _bucket_seconds);
        for (const auto& tcl:_twnd_count_limit) {
            auto [twnd, count, limit] = tcl;
            {
                auto lock = SpinLock(_lock);
                count = getCount_Unsafe(cur_utc/_bucket_seconds, count, twnd);
            }
            bytes += snprintf(buf+bytes, sizeof(buf)-bytes, ", %d, %.2f, %.2f, %.2f",
                    twnd*_bucket_seconds,
                    (double)limit/(double)(twnd*_bucket_seconds),
                    (double)count/(double)(twnd*_bucket_seconds),
                    (double)count/(double)limit*100.0);
        }
        return std::string(buf);
    }

    std::vector<double> RateEstimator::checkRates(int cur_utc, int upd_cnt) const {
        std::vector<double> ret;
        for (const auto& tcl:_twnd_count_limit) {
            auto [twnd, count, limit] = tcl;
            {
                auto lock = SpinLock(_lock);
                count = getCount_Unsafe(cur_utc/_bucket_seconds, count, twnd);
            }
            ret.push_back((double)(count+upd_cnt)/(double)(twnd*_bucket_seconds));
        }
        return ret;
    }

    std::shared_ptr<RateEstimator> RateEstimator::loads(const std::string& to_string) {
        // string from toString()
        const auto& line = utils::CSVUtil::read_line(to_string);
        size_t sz = line.size();

        if (sz<2) {
            fprintf(stderr, "RateEstimator: string too small in loads: %s\n", to_string.c_str());
            throw std::runtime_error("RateEstimator: string too small in loads!");
        }

        int utc = std::stoi(line[0]);
        int buckets = std::stoi(line[1]);
        std::map<int, std::tuple<double, int, double>> rsm;
        size_t i=2;
        const int Items = 4; // 4 numbers each limiter, see toString()
        while (i + Items <= sz) {
            int sec = std::stoi(line[i]);
            double rate_limit = std::stod(line[i+1]); // rate limit
            double cur_rate = std::stod(line[i+2]);   // current rate
            rsm[sec] = std::make_tuple(rate_limit, sec, cur_rate);
            i += Items;
        }

        // seconds are sorted coming from rsm's key
        std::vector<std::pair<double, int>> rate_seconds;
        std::vector<int> counts;
        for (const auto& sec_rs: rsm) {
            auto [rate_limit, sec, cur_rate] = sec_rs.second;
            rate_seconds.push_back(std::make_pair(rate_limit, sec));
            counts.push_back(int(cur_rate*sec+0.5));
        }
        auto re (std::make_shared<RateEstimator>(rate_seconds, buckets));

        // set up the counts
        utc /= buckets;
        re->_latest_second = utc;
        int cur_cnt = 0;
        for (size_t i=0; i<counts.size(); ++i) {
            auto& [twnd, cnt, limit] = re->_twnd_count_limit[i];
            cnt = counts[i];
            // populate _event so sum of [utc-twnd+1, utc] to be approximately 
            // no less than cnt 
            // we assuming the time window is increasing, to give more
            // accurate distribution of events in time
            if (cur_cnt >= cnt) {
                continue;
            }
            double ra = (double)(cnt-cur_cnt)/(double)twnd;
            int cum_cnt = 0;
            int utc0 = utc-twnd+1;
            for (int i=0; i<twnd; ++i) {
                int cnt0 = int((i+1)*ra+0.5) - cum_cnt;
                re->_event[(utc0+i)%re->_event_len]+=cnt0;
                cum_cnt += cnt0;
            }
            cur_cnt = cnt;
        }
        logInfo("RateEstimator loaded as %s",re->toString().c_str());
        return re;
    }
}

#pragma GCC diagnostic pop /* "-Werror=volatile" */
