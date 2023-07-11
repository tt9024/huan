#pragma once
#include <memory>
#include "time_util.h"
#include "circular_buffer.h"
#include "thread_utils.h"
#include <vector>
#include <tuple>

namespace utils {

    /*
     * This is a continuous rate estimator, used to measure number of 
     * events in a "trailing" or continous time period.  Typically,
     * to check if more than "Count" events have happened in previous
     * "TimeWindow".  
     *
     * The implementation uses a circular buffer of size "Count" to
     * store time stamps of events.  Upon a new event, it checks
     * if the difference between overriding timestamp and current timestamp
     * is more than "TimeWindow".  If so, then the limit has been broken.
     *
     * Note: the above is a baseline logic.  The implementation is 
     * more complicated due to the need to support "removal of an event".
     * Such operation could be needed for accounting "Cancel", for example.
     * Since the event times in circular buffer are overwritten upon
     * arrival of new events, a longer circular buffer, longer than "count",
     * is created according to the specified multiple.
     *
     * Note2: It is thread safe.  See the non-blocking implementation notes below.
     */
    class RateLimiter {
    public:
        // number of events (count) in trailing (TimeWindow_In_Seconds)
        // history_count_multiple is additional buffer used to store
        // previous event times. history_count_multiple=0 means no
        // additional buffer, k means k*count additional buffer space,
        // total space for _event is therefore (k+1)*count, k>=0
        RateLimiter(const int count, time_t TimeWindow_In_Second, int history_count_multiple=1);
        RateLimiter(const RateLimiter& rl);
        void operator=(const RateLimiter& rl); // have to be compatible
        static std::shared_ptr<RateLimiter> loads(const std::string& to_string);

        ~RateLimiter();

        // this checks if events happen now would violate the limit
        // return 0 if not, otherwise, micro_seconds before ok
        // In case of good, i.e. return 0
        uint64_t checkOnly(uint64_t cur_micro, int update_cnt) const;

        // this checks if events happen now would violate the limit
        // If update_cnt will not violate, update the event count.
        // Otherwise, return wait_micro WITHOUT update the event count
        // so that next try after wait_micro would come through.
        uint64_t check(uint64_t cur_micro, int update_cnt);
        uint64_t check(uint64_t cur_micro);  // short hand for check(cur_micro,1)
        uint64_t check();  // short hand for check(cur_micro,1)

        // This updates with arrival of events at cur_micro,
        // regardless of whether it could violate the rate limit.
        // Used for observer to update the arrival of events, i.e. fills,
        // for the risk monitor later to check when new orders comes.
        // Note update_cnt >= 0
        uint64_t updateOnly(uint64_t cur_micro, int update_cnt);
        uint64_t updateOnly(uint64_t cur_micro); //short hand for updateOnly(cur_micro, 1)
        uint64_t updateOnly(); //short hand for updateOnly(cur_micro, 1)

        // This removes previous events. Used in, for example, receiving Cancel,
        // to compensate the order counts.
        // Note, this is only good for removing upto _history_cnt events,
        // event times earlier than that are lost and is set to be 0. To allow
        // for longer history_window, increase history_count_multiple,
        // with a slight cost of memory and perhaps cache performance
        void removeOnly(int update_cnt);

        std::string toString() const;
        std::string toStringDump() const;  // long string used to load object

    protected:
        const int _count;
        const long long _twnd;
        const int _hist_count;
        volatile long long* _event;
        volatile long long _idx;
        mutable SpinLock::LockType _lock;

        int prevIdx(long long idx, int update_cnt) const {
            return (int)((idx+update_cnt-1+_hist_count-_count)%(long long)_hist_count);
        }
    };

    inline
    RateLimiter::~RateLimiter() { free ((void*)_event); _event = NULL; };

    inline
    uint64_t RateLimiter::check(uint64_t cur_micro) {
        return check(cur_micro, 1);
    }

    inline
    uint64_t RateLimiter::check() {
        return check(TimeUtil::cur_micro(), 1);
    }

    inline
    uint64_t RateLimiter::updateOnly(uint64_t cur_micro) {
        return updateOnly(cur_micro,1);
    }

    inline
    uint64_t RateLimiter::updateOnly() {
        return updateOnly(TimeUtil::cur_micro(),1);
    };

    ///////////////////////////////////
    // An ensemble of rate limiter with
    // similar interfaces
    //////////////////////////////////
    class RateLimiterEns {
    public:
        // vector of <count, TimeWindow_In_Second, history_count_multiple>, see RateLimiter constructor
        explicit RateLimiterEns(const std::vector<std::tuple<int, time_t, int>>& rate_limit_defs);
        RateLimiterEns(const RateLimiterEns& rle);
        RateLimiterEns() {};
        void operator=(const RateLimiterEns& rle); // have to be compatible

        // loads from toStringDump()
        static std::shared_ptr<RateLimiterEns> loads(const std::string& to_string_dump);

        // similar interfaces with RiskLimiter
        uint64_t checkOnly(uint64_t cur_micro, int update_cnt) const;
        uint64_t checkOnly() const; // short for checkOnly(cur_micro,1)
        uint64_t check(uint64_t cur_micro, int update_cnt);
        uint64_t check(uint64_t cur_micro);  // short hand for check(cur_micro,1)
        uint64_t check();  // short hand for check(cur_micro,1)
        uint64_t updateOnly(uint64_t cur_micro, int update_cnt);
        uint64_t updateOnly(uint64_t cur_micro); //short hand for updateOnly(cur_micro, 1)
        uint64_t updateOnly(); //short hand for updateOnly(cur_micro, 1)
        void removeOnly(int update_cnt);
        std::string toString() const;
        std::string toStringDump() const;  // long string used to load object

    private:
        std::vector<std::shared_ptr<RateLimiter>> m_rl;
    };

    inline
    RateLimiterEns::RateLimiterEns(const std::vector<std::tuple<int, time_t, int>>& rld) {
        for (const auto& rl: rld) {
            auto [count, sec, mul] = rl;
            m_rl.emplace_back(std::make_shared<RateLimiter>(count, sec, mul));
        }
    }

    inline
    RateLimiterEns::RateLimiterEns(const RateLimiterEns& rle) {
        for (const auto& rl: rle.m_rl) {
            m_rl.emplace_back(std::make_shared<RateLimiter>(*rl));
        }
    }

    inline
    void RateLimiterEns::operator=(const RateLimiterEns& rle) {
        if (m_rl.size() != rle.m_rl.size()) {
            throw std::runtime_error("RateLimiterEns failed copy mismatch m_rl length");
        }
        for (size_t i=0; i<m_rl.size(); ++i) {
            m_rl[i]->operator=(*rle.m_rl[i]);
        }
    }

    inline
    std::shared_ptr<RateLimiterEns> RateLimiterEns::loads(const std::string& to_string) {
        auto rle = std::make_shared<RateLimiterEns>( std::vector<std::tuple<int, time_t, int>>() );
        auto pos = to_string.find(",,");
        int pos0 = 0;
        while (pos != std::string::npos) {
            rle->m_rl.emplace_back(RateLimiter::loads(to_string.substr(pos0, pos-pos0)));
            pos0 = pos+2; // skip the ",,"
            pos = to_string.find(",,", pos0);
        }
        rle->m_rl.emplace_back(RateLimiter::loads(to_string.substr(pos0)));
        return rle;
    }

    inline
    std::string RateLimiterEns::toString() const {
        // shouldn't be empty
        std::string ret = std::string("[")+m_rl[0]->toString() + "]";
        for (size_t i=1; i<m_rl.size(); ++i) {
            ret += ",[";
            ret += m_rl[i]->toString();
            ret += "]";
        }
        return ret;
    }

    inline
    std::string RateLimiterEns::toStringDump() const {
        // shouldn't be empty
        std::string ret = m_rl[0]->toStringDump();
        for (size_t i=1; i<m_rl.size(); ++i) {
            ret += ",,";
            ret += m_rl[i]->toStringDump();
        }
        return ret;
    }

    inline
    uint64_t RateLimiterEns::checkOnly(uint64_t cur_micro, int update_cnt) const {
        uint64_t wait_sec = 0;
        for (const auto& rl : m_rl) {
            auto wait0 = rl->checkOnly(cur_micro, update_cnt);
            wait_sec = (wait_sec>wait0? wait_sec: wait0);
        }
        return wait_sec;
    }
    
    inline
    uint64_t RateLimiterEns::checkOnly() const {
        return checkOnly(utils::TimeUtil::cur_micro(), 1);
    }

    inline
    uint64_t RateLimiterEns::check(uint64_t cur_micro, int update_cnt) {
        if (__builtin_expect(cur_micro == 0, 0)) {
            cur_micro = TimeUtil::cur_micro();
        }
        uint64_t wait_micro = checkOnly(cur_micro, update_cnt);
        if (__builtin_expect(wait_micro==0, 1)) {
            updateOnly(cur_micro, update_cnt);
            return 0;
        }
        return wait_micro;
    }

    inline
    uint64_t RateLimiterEns::check(uint64_t cur_micro) {
        return check(cur_micro, 1);
    }

    inline
    uint64_t RateLimiterEns::check() {
        return check(utils::TimeUtil::cur_micro(), 1);
    }

    inline
    uint64_t RateLimiterEns::updateOnly(uint64_t cur_micro, int update_cnt) {
        uint64_t wait_sec = 0;
        for (const auto& rl : m_rl) {
            auto wait0 = rl->updateOnly(cur_micro, update_cnt);
            wait_sec = (wait_sec>wait0? wait_sec: wait0);
        }
        return wait_sec;
    }

    inline
    uint64_t RateLimiterEns::updateOnly(uint64_t cur_micro) {
        return updateOnly(cur_micro, 1);
    }

    inline
    uint64_t RateLimiterEns::updateOnly() {
        return updateOnly(utils::TimeUtil::cur_micro(), 1);
    }

    inline
    void RateLimiterEns::removeOnly(int update_cnt) {
        for (const auto& rl : m_rl) {
            rl->removeOnly(update_cnt);
        }
    }
}
