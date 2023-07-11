#pragma once

#include "FloorBase.h"
#include "PositionManager.h"
#include <string>
#include <memory>

/*
 * To be called by thread of FloorTrader (FT), which holds FloorCPR (channel, pm and risk)
 *
 * Upon start, FT creates all the ExecutionTrader (ET)
 * FT handles messages from user and TP:
 *  - Upon receiving a new PI from strategy or user with target qty,
 *    it finds a ET that should handle this PI and call onEntry(PI).
 *  - Upon loop for each one in piMap, call onRun(), the trader
 *    does all the stuffs it needs and update state
 *  - Upon ExecutionReport, call ET.onER()
 *
 *  Upon shutdown:
 *      cancel all open orders
 *
 * ExecutionTrader notes
 *
 * It is a passive logic object, that wraps an execution logic that allows high frequency CPU-pin'ed tight loop such as
 * ZTrader with rate estimation.  It is expected to be called on a FT thread and have access to the FT's FloorCPR in thread, 
 * i.e. no lockings. It maintains a trader specific state, such as short term rates, and could be updated in following calls.
 * - ETInfo = onEntry(PI)   --> PI is the strategy's target position request
 * - bool = onRun(ETInfo)  --> scanOO, sendOrder, update states
 * - bool = onERUpd(ETInfo) --> called when ER received, update state if needed
 * - void = onShutdown()    --> any clean up necessary (OO cancel, etc)
 *
 * It has the MD and FloorCPR.
 *
 */

namespace pm {
    class FloorTrader;  // forward declaration, header file included in cpp
    using PIType = std::shared_ptr<pm::FloorBase::PositionInstruction>; // defined in FloorBase.h
    using OOType = pm::OpenOrder; // defined in PositionData.h
    struct ETInfo {
        PIType _pi;  // holds the target position and fill stats, all updated
                     // upon a new target position
        uint64_t _enter_micro;
        uint64_t _next_micro;  // time to be checked next
        bool _done;
        std::shared_ptr<void> _states;  // derived states

        template<typename T>
        ETInfo(const PIType& pi, \
               std::shared_ptr<T> state)
        : _pi(pi), // copy everything, including stats shared_ptr
          _enter_micro(utils::TimeUtil::cur_micro()),
          _next_micro(_enter_micro), // default to be checked next
          _done(false),
          _states(std::static_pointer_cast<void>(state)) {
        };

        ETInfo(): _enter_micro(0), _next_micro((uint64_t)0x7fffffffffffffffLL), _done(true) {};

        template<typename T>
        void setStates(std::shared_ptr<T> states) {
            _states = std::static_pointer_cast<void>(states);
        }

        template<typename T>
        std::shared_ptr<T> getStates() const {
            return std::static_pointer_cast<T>(_states);
        }

        std::string toString() const {
            char buf[1024];
            snprintf(buf, sizeof(buf), "ETInfo: done(%s) next(%s) entered(%s), PI(%s)", 
                    _done?"YES":"NO",
                    utils::TimeUtil::frac_UTC_to_string(_next_micro,6).c_str(),
                    utils::TimeUtil::frac_UTC_to_string(_enter_micro,6).c_str(),
                    _pi?_pi->toString().c_str():"NONE");
            return std::string(buf);
        }
    };

    class ExecutionTrader {
        /*
         * A logic object, driven by the thread wrapper. It operates 
         * on the "state" data structure that it creates upon position requests.
         * The state is indexed by "algo+symbol+type", and is created 
         * upon a new key from a position requestion, and mark done upon
         * target position achieved. 
         *
         * Specific ExecutionTrader, such as ZTrader, overwrite the virtual
         * functions of
         * onEntry(), onUpdate(), onRun() onER() to defined the bahavior
         */

    public:
        // type defs
        ExecutionTrader(const std::string& name, FloorTrader& floor):_name(name), _flr(floor) {};
        virtual std::shared_ptr<ETInfo> onEntry(const PIType& pi) = 0; // new instruction
        virtual bool onUpdate(const PIType& pi, std::shared_ptr<ETInfo>& etinfo) = 0; // update from floor
        virtual bool onRun(std::shared_ptr<ETInfo>& etinfo) = 0;  // loop on scanOO and new order, return true if need to be checked again, otherwise false
        virtual bool onER(const ExecutionReport& er, std::shared_ptr<ETInfo>& etinfo) = 0; // update on the order it sends
        virtual ~ExecutionTrader() {};
        const std::string _name;

    protected:
        FloorTrader& _flr;
    };
}
