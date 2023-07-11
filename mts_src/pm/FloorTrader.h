#pragma once

#include "FloorCPR.h"
#include "ExecutionTrader.h" // FloorTrader forward declared this header
#include <map>
#include <vector>
#include <set>

/*
 * FloorTrader is a thread that runs a set of different
 * execution trader, (i.e. ZTrader), on a pre-defined
 * markets with given parameters (from the config file).
 * The config file looks like: (ALL expands all mkts in symbol_map)
 * 
 * Instance Name: FloorTrader1
 * File Name: FloorTrader1.cfg
 * ============
 * TWAP = {  # from PositionInstruction::TYPE
 *     config = ztrader.cfg
 *     symbol = [ES_N1, FV_N1]
 *  }
 *  TWAP2 = {
 *      config = ytrader.cfg
 *      symbol = [ES_N2, FV_N1, ALL_N0]
 *  }
 *  ===========
 *  Note the trader name, like "TWAP", "TWAP2",
 *  have to be defined in PositionInstruction::TYPE, matching TypeStr, 
 *  otherwise, it would throw
 *
 *  This defines all types + symbols to be run on this FT thread.
 *  Live uniqueness check ensure there is only one instance on each 
 *  type+symbol.  There is a "ALL" that matches all symbols in the
 *  symbol map.  The _N applies to each symbol. Multiple types of
 *  traders can trade the same symbol, but one type of trader that
 *  run on different FT threads cannot trade the same symbol. This
 *  is enforced by OnlyMe check.
 *
 *  To ensure existence of a type+symbol, strategy should
 *  get+set a same target position for each market+type it
 *  is expected to trade on, and check for timeouts, which should
 *  log error for non-existence.
 *
 *  So to summarize: vector of Trader index by type  (m_et)
 *                   algo+symbol -> ETInfo (m_eiMap, only one type active)
 *  The interaction between ET is outline as the following.
 *
 *  construction: 
 *      populate a m_symbol_set, a vector of symbol set, indexed by type.
 *      The symbol set is all the symbols allowed to trade by the type.
 *      create ET from config and populate m_et, vector indexed
 *      by type (int)
 *
 *  start_derived()
 *      create and start the onlyme instance check
 *
 *  handlePositionReq_derived()
 *      gets the "set" request, check m_eiMap, the algo+symbol map, 
 *      if etinfo  exists:
 *          if not same type - remove that etinfo
 *          otherwise, 
 *              run bool ET.onUpdate(pi, et_info)
 *      if not exists:
 *          check if this FT covers it
 *          if yes, then run ET.onEntry(pi)
 *
 *  handleExecutionReport_derived(er)
 *      gets the etinfo from er's algo+symbol, if found, 
 *      call ET.onER(er, etinfo), which should update 
 *      etinfo's state such as done
 *
 *  run_loop_derived()
 *      scan through m_eiMap and run bool onRun(etinfo)
 *      if all false, then sleep a IdleSleepMicro
 *
 *  handlePositionReq_derived()
 *      handles set target position from C++ and user cmd 'X'
 *      see also FloorCPR.h's notes
 */

namespace pm {
    class FloorTrader: public FloorCPR<FloorTrader> {
    public:
        explicit FloorTrader(const std::string& instance_name); //config file expected as config/floortrader_instance_name.cfg
        ~FloorTrader();

        FloorTrader(const FloorTrader& mgr) = delete;
        FloorTrader& operator=(const FloorTrader& mgr) = delete;

        void start_derived(); // start with onlyme
        void shutdown_derived();  // destroy ETs
        std::string toString_derived() const; // dump each ET's state upon user command to dump FT
        void addPositionSubscriptions_derived(); // set only
        void run_loop_derived(); // run etInfoMap 

        void handleExecutionReport_derived(const pm::ExecutionReport& er); // get the key and call ET.onER()
        bool handleUserReq_derived(const MsgType& msg, std::string& respstr);  // handles the user request on dump FT state

        // handles positions requests from C++ and user cmd 'X'
        // for msg type GetPositionReq - it comes from C++, with msg buffer
        // being the PositionRequest; for msg type SetPositionReq, it comes
        // from both C++ and 'X', with msg buffer being PositionInstruction.
        //
        // The PositionInstruction has a (int) type 
        bool handlePositionReq_derived(const MsgType& msg, MsgType& msg_out);
        friend class FloorBase;

    protected:
        // vector of ET and symbol_set, indexed by the integer of type, 
        // reserved size of TYPE::TOTAL_TYPES
        std::vector<std::shared_ptr<ExecutionTrader>> m_et;
        std::vector<std::set<std::string>> m_symbol_set;

        // map of algo+symbol to ETInfo 
        // only one type can active for algo+symbol at any time
        // see also handlePositionReq_derived()
        std::map<std::string, std::shared_ptr<ETInfo>> m_eiMap; 
        
    private:
        enum {
            // execution related parameters
            IdleSleepMicro       = 100*1000,  // sleep duration
            MinSleepMicro        = 10,        // would spin instead
        };
        time_t m_last_onlyme_second;
        void addPositionInstruction(const std::shared_ptr<PositionInstruction> & pi);
        std::shared_ptr<ExecutionTrader> createTrader(int type, const std::string& trader_cfg_fn);
        void checkOnlyMe();

        // the key for m_eiMap
        std::string getKey(const std::string& algo, const std::string symbol) const;
    };
};
