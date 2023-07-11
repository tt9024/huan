#include <stdexcept>

#include "rate_limiter.h"
#include "csv_util.h"
#include "plcc/PLCC.hpp"

//  compound assignment with ‘volatile’-qualified left operand is deprecated [-Werror=volatile]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvolatile"

namespace utils {

    // ******************************
    // implementations RateLimiter
    // ******************************
    RateLimiter::RateLimiter(const int count, time_t TimeWindow_In_Second, int history_count_multiple) 
    :_count(count), _twnd(TimeWindow_In_Second*1000000LL),
     _hist_count(count*(history_count_multiple+1)),
     _event(_hist_count>0? ((volatile long long*)malloc(sizeof(long long)*_hist_count)) : nullptr),
     _idx(0), _lock(false)
    {
        if (_hist_count <= 0) {
            throw std::runtime_error("RateLimit history_count_multiple negative!");
        }
        if (!_event) {
            throw std::runtime_error("RateLimit malloc");
        }
        memset((char*)_event, 0, sizeof(int64_t)*_hist_count);
    };

    RateLimiter::RateLimiter(const RateLimiter& rl) 
    : _count(rl._count), _twnd(rl._twnd), _hist_count(rl._hist_count), 
      _event(nullptr), _idx(rl._idx), _lock(false)
    {
        _event = (volatile long long*) malloc(_hist_count*sizeof(long long));
        if (!_event) {
            throw std::runtime_error("RateLimiter: failed construct from copy in malloc " + std::to_string(_hist_count*sizeof(long long)));
        }
        memcpy((void*)_event, (const void*)rl._event, _hist_count*sizeof(long long));
    }

    void RateLimiter::operator=(const RateLimiter& rl) {
        if ( (_count != rl._count) ||
             (_twnd != rl._twnd) ||
             (_hist_count != rl._hist_count) ) {
            throw std::runtime_error("RateLimiter: failed to copy mismatch: this and given\n" + toString() + "\n" + rl.toString());
        }
        memcpy((void*)_event, (const void*) rl._event, _hist_count*sizeof(long long));
        _idx = rl._idx;
    }

    uint64_t RateLimiter::checkOnly(uint64_t cur_micro, int update_cnt) const {
        // this checks if an event happens now would violate the limit
        // return 0 if not, otherwise, micro_seconds before ok
        // In case of good, i.e. return 0
        if (__builtin_expect(update_cnt <= 0, 0)) {
            logError("RateLimiter checkOnly(): non-positive update_cnt %d", update_cnt);
            throw std::runtime_error("RateLimiter checkOnly() negative update_cnt");
        }
        if (__builtin_expect(update_cnt > _count, 0)) {
            // if update_cnt is more than _count, sure
            // to wait _twnd
            return _twnd;
        }
        if (__builtin_expect(cur_micro == 0, 0)) {
            cur_micro = TimeUtil::cur_micro();
        };
        long long ts = cur_micro;
        {
            auto lock = SpinLock(_lock);
            ts = _event[prevIdx(_idx,update_cnt)];
        }
        long long tdiff = (long long)cur_micro - ts;
        if (tdiff < _twnd) {
            return (uint64_t) _twnd - tdiff;
        }
        return 0;
    }

    uint64_t RateLimiter::check(uint64_t cur_micro, int update_cnt) {
        // this simulate a continous operation, for example logger,
        // that if not violating, update the event count, otherwise,
        // return wait_micro without update the event count (so
        // that next try after wait_micro would come through).
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

    uint64_t RateLimiter::updateOnly(uint64_t cur_micro, int update_cnt) {
        // This updates with arrival of an event at cur_micro,
        // regardless of whether it could violate the rate limit.
        // Used for observer to update the arrival of events, i.e. fills,
        // for the risk manager later to check when new orders comes.
        //
        // Return a micro_seconds that must be waited (0 for good case).
        // Note update_cnt >= 0
        //
        // The function is thread safe.  It works in the following loop:
        // It get the current idx and do a CAS on the idx+1, upto a 
        // max_spin count. If max_spin exceeded, it throws a runtime_error.
        // 
        if (__builtin_expect(update_cnt <= 0, 0)) {
            logError("RateLimit updateOnly(): ERROR non-positive update_cnt %d", update_cnt);
            throw std::runtime_error("RateLimit updateOnly(): negative update_cnt!");
        }
        if (__builtin_expect(cur_micro == 0, 0)) {
            cur_micro = TimeUtil::cur_micro();
        };
        long long ts = cur_micro;
        {
            auto lock = SpinLock(_lock);
            long long idx = _idx;
            if (__builtin_expect(update_cnt <= _count,1)) {
                ts = _event[prevIdx(idx,update_cnt)];
            } // else ts = cur_micro, i.e. wait _twnd
            const long long idx1 = idx + (update_cnt<_hist_count?update_cnt:_hist_count);
            for (long long i=idx; i<idx1; ++i) {
                _event[i%(long long)_hist_count] = (long long)cur_micro;
            }
            _idx += update_cnt;
        }
        // regulate wait_micro between [0, _twnd]
        long long wait_micro0 = cur_micro - ts;
        if (__builtin_expect(wait_micro0 >= _twnd, 1)) {
            return 0;
        }
        wait_micro0 = (wait_micro0<0?0:wait_micro0);
        return _twnd - wait_micro0;
    }

    void RateLimiter::removeOnly(int update_cnt) {
        // This removes previous events. Used in, for example, receiving Cancel,
        // to compensate the order counts.
        // '0' is stored in place of the removed event time, that is, if 
        // removing more than hist_count, no history is kept and all is reset.
        if (__builtin_expect(update_cnt <= 0, 0)) {
            logError("RateLimit removeOnly_Approximate(): ERROR non-positive update_cnt %d", update_cnt);
            throw std::runtime_error("RateLimit removeOnly_Approximate(): negative update_cnt!");
        }
        {
            auto lock = SpinLock(_lock);
            if (__builtin_expect( (update_cnt>=_hist_count)||(update_cnt>=_idx), 0)) {
                memset((char*)_event, 0, sizeof(int64_t)*_hist_count);
                return;
            }
            long long idx = _idx;
            for (long long i=idx-1; i>=idx-update_cnt; --i) {
                _event[i%(long long)_hist_count] = 0;
            }
            _idx-=update_cnt;
        }
    }

    std::string RateLimiter::toString() const {
        // format _count, _twnd, _hist_count
        // (_event[i] in utc, NOT microseconds)
        char buf[256];
        snprintf(buf, sizeof(buf), "%d,%lld,%d", _count, _twnd, _hist_count);
        std::string ret(buf);
        return ret;
    }

    std::string RateLimiter::toStringDump() const {
        // output a readable string that could be loaded, and not too long
        // format _count, _twnd, _hist_count, _idx, _event[0], _event[1]-_event[0], ...
        // (_event[i] in utc, NOT microseconds)
        // use delta to save space
        long long idx;
        long long* evt = (long long*) malloc(_hist_count* sizeof(long long));
        {
            auto lock = SpinLock(_lock);
            idx = _idx;
            memcpy(evt, (const void*)_event, _hist_count*sizeof(long long));
        }
        std::string ret = toString();
        ret+=',';
        ret+=std::to_string(idx);
        ret+= ',';
        ret+=std::to_string(evt[0]);
        long long prev_v = evt[0]/1000000LL;
        for (int i=1; i<_hist_count; ++i) {
            ret+=',';
            auto v = (int)(evt[i]/1000000LL);
            ret+=std::to_string(v-prev_v);
            prev_v = v;
        }
        return ret;
    };

    std::shared_ptr<RateLimiter> RateLimiter::loads(const std::string& dump_string) {
        // format _count, _twnd, _hist_count, _idx, _event[0], _event[1]-_event[0], ...
        const auto& line = utils::CSVUtil::read_line(dump_string);
        size_t sz = line.size();
        if (sz < 5) {
           logError("RateLimiter: cannot load from string: too few fields %s", dump_string.c_str());
           throw std::runtime_error("RateLimiter load too few fields");
        }
        int count = std::stoi(line[0]);
        long long twnd = std::stoll(line[1]);
        int hist_count = std::stoi(line[2]);
        long long idx = std::stoll(line[3]);
        long long evt0 = std::stoll(line[4]);

        if ((int)sz-4 != hist_count) {
           logError("RateLimiter: cannot load from string: fields mismatch hist_count: dump_string_fields(%d), hist_count(%d)", (int)sz-4, hist_count);
           throw std::runtime_error("RateLimiter load object fields mistmach hist_count!");
        }

        std::shared_ptr<RateLimiter> rl = std::make_shared<RateLimiter>( count, (time_t)(twnd/1000000LL), hist_count/count-1);
        if ((rl->_count != count) || (rl->_twnd != twnd) || (rl->_hist_count != hist_count)) {
           logError("RateLimiter: cannot load from string: object mismatch: dump_string(%s), created(%s)", dump_string.c_str(), rl->toString().c_str());
           throw std::runtime_error("RateLimiter load object mismatch!");
        }

        rl->_idx = idx;
        rl->_event[0] = evt0;
        evt0/=1000000;
        for (int i=1; i<hist_count; ++i) {
            evt0 += std::stoi(line[4+i]);
            rl->_event[i] = (long long)evt0*1000000LL;
        }
        return rl;
    }
}

#pragma GCC diagnostic pop /* "-Werror=volatile" */