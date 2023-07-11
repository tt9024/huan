#pragma once
#include <vector>
#include <tuple>
#include <memory>

#include "time_util.h"
#include "thread_utils.h"

namespace utils {
    class RateEstimator {
    public:
        // rate of events (Rate) defined in trailing (TimeWindow_In_Seconds)
        // different with RateLimiter, it maintains a per-second
        // event count, and check the rates, defined as count/window,
        // against given thresholds.
        // Multiple rate estimators, ie. <10.0 per seconds, 30 seconds> and
        // <5.0 per seconds, 300 seconds>, could be defined.  All estimators
        // are checked and/or updated simultanuously.
        //
        // Same as RateLimiter, it is thread safe.
        //
        // TODO: 
        // 1 - check/update cannot be for an earlier time (1 sec earlier) than
        // the latest. This can be allowed, but no usage case is identified.
        
        // rate_seconds: vector of pair of <rate, windows_seconds>
        // bucket_seconds: seconds to aggregate, i.e. 30, aggregates all
        // event counts within 30 seconds. This helps with windows in hours.
        explicit RateEstimator(const std::vector<std::pair<double, int>>& rate_seconds, int bucket_seconds=1);
        RateEstimator(const RateEstimator& re);
        void operator=(const RateEstimator& re);

        // load object from a string created from toString()
        static std::shared_ptr<RateEstimator> loads(const std::string& to_string);
        ~RateEstimator();

        uint64_t checkOnly(uint64_t cur_micro, int update_cnt) const;
        uint64_t check(uint64_t cur_micro, int update_cnt);
        uint64_t check(); // short hand for check(cur_micro, 1)

        // updates should be in non-decreasing time order. OK with
        // decrease of 1 bucket_seconds considering the turn of the bucket
        // in multiple threading setting
        uint64_t updateOnly(uint64_t cur_micro, int update_cnt, bool ret_wait=true);
        uint64_t updateOnly(); // short hand for updateOnly(cur_micro, 1, true)

        // removes the counts starting from the _latest for as much as update_cnt
        void removeOnly(int update_cnt);

        // without update, returns current rates after counting in upd_cnt
        // the returned rates in same order as given in constructor
        std::vector<double> checkRates(int cur_utc, int upd_cnt=0) const;

        // this updates the rate limits for each one of the rate_seconds
        // in the construction
        void updateRateLimit(const std::vector<double> & rates);

        // this resets all the counts
        void resetAll();
        std::string toString() const;
        std::string toString(time_t cur_utc) const;

        // convenience functions: taking and returning utc instead of micro
        time_t checkOnlyUTC(time_t cur_utc, int update_cnt) const;
        time_t checkUTC(time_t cur_utc, int update_cnt);
        time_t updateOnlyUTC(time_t cur_utc, int update_cnt, bool ret_wait=true);

    protected:
        int _bucket_seconds;
        // <_twnd, _cum_count, _count_limit>, window size, cum count and limit
        std::vector<std::tuple<int, int, int>> _twnd_count_limit;
        int _event_len;  // length of allocated _event, i.e. maximum window
        volatile unsigned int* _event;
        volatile int _latest_second;
        mutable SpinLock::LockType _lock;

        // get the event at current time as cur_utc
        // number of events in [latest_seconds-twnd+1, latest_seconds] as cur_count
        // and window_seconds as wnd_sec
        // NOTE - both cur_utc and window_seconds should be divided by bucket
        // return event counts from [cur_utc-twnd+1, cur_utc] inclusive
        int getCount_Unsafe(int cur_utc, int cur_count, int wnd_sec) const;
    };

    inline
    RateEstimator::~RateEstimator() { free ((void*)_event); _event = NULL; };

    inline
    uint64_t RateEstimator::check() {
        return check(TimeUtil::cur_micro(), 1);
    }

    inline
    uint64_t RateEstimator::updateOnly() {
        return updateOnly(TimeUtil::cur_micro(), 1);
    }

    inline
    std::string RateEstimator::toString() const {
        return toString(utils::TimeUtil::cur_utc());
    } 

    inline
    time_t RateEstimator::checkOnlyUTC(time_t cur_utc, int update_cnt) const {
        return checkOnly(cur_utc*1000000ULL, update_cnt)/1000000ULL;
    }

    inline
    time_t RateEstimator::checkUTC(time_t cur_utc, int update_cnt) {
        return check(cur_utc*1000000ULL, update_cnt)/1000000ULL;
    }

    inline
    time_t RateEstimator::updateOnlyUTC(time_t cur_utc, int update_cnt, bool ret_wait) {
        return updateOnly(cur_utc*1000000ULL, update_cnt, ret_wait)/1000000ULL;
    }
}
