#include "TWAPTrader.h"
#include "FloorTrader.h"
#include "plcc/ConfigureReader.hpp"
#include <stdexcept>

namespace pm {
    TWAPTrader::TWAPTrader(const std::string& name, FloorTrader& floor, const std::string& cfg_fn)
    : ExecutionTrader(name, floor),
      _param(cfg_fn)
    {
        logInfo("TWAPTrader %s created, config file(%s), param:%s", name.c_str(), cfg_fn.c_str(), _param.toString().c_str());
    }

    std::shared_ptr<ETInfo> TWAPTrader::onEntry(const PIType& pi) {
        // initialize the state with current time
        auto state = std::make_shared<TWAPState>();
        return std::make_shared<ETInfo>(pi, state);
    }

    bool TWAPTrader::onUpdate(const PIType& pi, std::shared_ptr<ETInfo>& etinfo) {
        auto cur_micro = utils::TimeUtil::cur_micro();
        // if target pos different, copy PI in and set done false
        if (pi->qty != etinfo->_pi->qty) {
            etinfo->_pi = pi; // just copy
            etinfo->_done = false;
            etinfo->_enter_micro = cur_micro;
            etinfo->_next_micro = cur_micro; // demand to be checked
            auto state = std::make_shared<TWAPState>();
            etinfo->setStates(state);
            return true;
        } 
        // extend target_utc if not done and expired
        if ((!etinfo->_done) && (etinfo->_pi->target_utc < (int)(cur_micro/1000000ULL)) 
            && (pi->target_utc > etinfo->_pi->target_utc)) {
            logInfo("extending target utc of %s for %d seconds", 
                    etinfo->_pi->toString().c_str(), 
                    (int)(pi->target_utc - etinfo->_pi->target_utc));
            etinfo->_pi->target_utc = pi->target_utc;
            etinfo->_next_micro = cur_micro; // demand to be checked
            return true;
        }
        return false;
    }

    bool TWAPTrader::onRun(std::shared_ptr<ETInfo>& etinfo) {
        auto state = etinfo->getStates<TWAPState>();
        /*
        next_micro = scanOO(etinfo, state);
        */
        auto next_scan_micro = scanQty(etinfo, state);
        etinfo->_next_micro = next_scan_micro;
        return true;
    }

    bool TWAPTrader::onER(const ExecutionReport& er, std::shared_ptr<ETInfo>& etinfo) {
        logInfo("TWAP received er (%s)", er.toString().c_str());
        // toggle done
        const auto& pi = etinfo->_pi;
        if (_flr.checkTargetQty(pi->qty, pi->algo, pi->symbol) == 0) {
            if (!etinfo->_done) {
                etinfo->_done = true;
                logInfo("%s is done!", etinfo->toString().c_str());
            }
            return true;
        }

        // not done yet
        if (__builtin_expect(etinfo->_done,0)) {
            logInfo("extra fill on a done PI(%s)", pi->toString().c_str());
            if (er.isFill() || er.isNew()) {
                etinfo->_done = false;
                logInfo("restarting the state");
            } else {
                logInfo("ignored");
                return true;
            }
        }
        auto state = etinfo->getStates<TWAPState>();
        scanQty(etinfo, state);
        return true;
    }

    uint64_t TWAPTrader::scanQty(const std::shared_ptr<ETInfo>& etinfo, std::shared_ptr<TWAPState>& state) {
        if (__builtin_expect(etinfo->_done,0)) {
            return -1ULL;
        }
        auto cur_micro = utils::TimeUtil::cur_micro();

        // This could be called before the _next_micro, i.e. onER()
        // we decide to ignore and stick to _next_micro here
        if (__builtin_expect(etinfo->_next_micro > cur_micro,0)) {
            return etinfo->_next_micro;
        }

        const auto& pi = etinfo->_pi;
        const uint64_t target_micro = pi->target_utc*1000ULL*1000ULL;
        if (target_micro < cur_micro-_param.qty_interval_micro) {
            // expired
            return cur_micro + _param.qty_interval_micro;
        }

        // update the last_qty_micro
        if (__builtin_expect(state->_last_qty_micro + _param.qty_interval_micro/2 > cur_micro,0)) {
            logError("unexpected run for %s", etinfo->toString().c_str());
        }
        state->_last_qty_micro = cur_micro;

        // compute per-min qty
        auto trd_qty = _flr.checkTargetQty(pi->qty, pi->algo, pi->symbol);
        if (trd_qty == 0) {
            etinfo->_done = true;
            logInfo("%s is done!", etinfo->toString().c_str());
            return cur_micro + _param.qty_interval_micro;
        }

        int cnt = 1;
        if (target_micro > cur_micro + _param.qty_interval_micro/2+1) {
            cnt = (int)((double)(target_micro - cur_micro)/(double)_param.qty_interval_micro + 0.5)+1;
        }
        long long qty = trd_qty/cnt;
        if (qty == 0) {
            // trade minimum
            qty = (trd_qty>0? 1:-1);
        }
        if (std::abs(qty) > _param.max_slice_qty) {
            logError("%s slice exceed %lld, wanted %lld, state:%s", _name.c_str(), 
                    (long long)_param.max_slice_qty, qty,
                    etinfo->toString().c_str());
            qty = (trd_qty>0? _param.max_slice_qty:-_param.max_slice_qty);
        }
        logInfo("%s slice: cnt(%d), qty(%lld), state(%s)", _name.c_str(), cnt, qty, etinfo->toString().c_str());
        auto pi0 = std::make_shared<FloorBase::PositionInstruction>(*pi);
        pi0->qty = qty;
        pi0->type = FloorBase::PositionInstruction::MARKET;
        _flr.sendOrder_InOut(pi0);
        return cur_micro + _param.qty_interval_micro;
    }

    TWAPTrader::TWAPParam::TWAPParam(const std::string& cfg_fn) {
        auto fn = std::string("config/")+cfg_fn;
        const auto cfg(utils::ConfigureReader(fn.c_str()));
        qty_interval_micro = (uint64_t) cfg.get<long long>("qty_interval_micro");
        oo_interval_micro  = (uint64_t) cfg.get<long long>("oo_interval_micro");
        max_slice_qty      = cfg.get<int>("max_slice_qty");
    }

    std::string TWAPTrader::TWAPParam::toString() const {
        char buf[1024];
        snprintf(buf, sizeof(buf), \
                "TWAPParam:[qty_interval_micro(%lld), oo_interval_micro(%lld), max_slice_qty(%d)", \
                (long long) qty_interval_micro, (long long) oo_interval_micro, max_slice_qty);
        return std::string(buf);
    }

    /*
    uint64_t scanOO(const std::shared_ptr<ETInfo>& etinfo, std::shared_ptr<TWAPState>& state) {
        // use the FM's scanOO since we sent MARKET orders
        return -1ULL;
    }
    */

}
