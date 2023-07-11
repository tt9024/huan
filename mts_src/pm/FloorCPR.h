#pragma once

#include "FloorBase.h"
#include "PositionManager.h"
#include "ExecutionReport.h"
#include "RiskMonitor.h"
#include "time_util.h"
#include <stdexcept>
#include <atomic>
#include <unordered_map>
#include <set>

/*
 * All basic floor thread that maintains channel, position and risk. 
 * It also provide utility functions to access the current position and order sending. 
 * As a rule of thumb, anything at or below clOrdId level should be at FloorCPR.
 *
 * Derived Class, such as FloorManager or FloorTrader, implements 
 * additional logic at various point of loop handling of the thread.
 *
        void start_derived();  // called upon position loaded and run_loop_derived will be called afterwards
        void shutdown_derived();
        std::string toString_derived() const;
        addPositionSubscriptions_derived(); // C++ position get/set
        void run_loop_derived(); // extra handling after handling all messages in loop

        void handleExecutionReport_derived(const pm::ExecutionReport& er); // extra handling after receiving an er

        bool handleUserReq_derived(const MsgType& msg, std::string& respstr);
        // handles all except buy/sell, position req, display.
        // return false if no reply is needed after this
        // true if reply is needed and "resptr" copied to msg buffer
        // if respstr size 0, then no msg is sent

        bool handlePositionReq_derived(const MsgType& msg, MsgType& msg_out);  // handles positions get/set requests from C++ and user cmd 'X'.
        // Note - for msg type GetPositionReq - it comes from C++, 
        // with msg buffer being the PositionRequest;
        // The position get requests are handled ONLY by the FloorManager
        // the set requests is handled by the FloorTrader
        // for msg type SetPositionReq, it comes from both C++ and 
        // 'X', with msg buffer being PositionInstruction.
 */

namespace pm {
    template<typename Derived>
    class FloorCPR: public FloorBase {
    public:
        explicit FloorCPR(const std::string& name);
        ~FloorCPR();
        void start();
        void stop();
        std::string toString() const; 

        // ***
        // send order utilities
        // ***
        
        // if clOrdId is not nullptr but empty, then generate a clOrdId and assigns to it
        // if clOrdId is not nullptr and non-empty, then uses the given clOrdId
        void sendOrder_InOut(const std::shared_ptr<PositionInstruction>& pi);

        // return err string, empty upon success
        std::string sendOrder(const bool isBuy, const char* algo, 
                              const char* symbol, int64_t qty, double px, 
                              std::string* clOrdId = nullptr);
        std::string sendOrderByString(const char* bsstr, std::string* clOrdId=nullptr);
        std::string sendCancel(const char* clOrdId);
        std::string sendCancelReplacePx(const char* clOrdId, double new_price);
        std::string sendCancelReplaceSize(const char* clOrdId, uint64_t size); // size positive, same sign with original
        std::string sendCancelReplaceByString(const char* bsstr);

        // ***
        // position utilities
        // ***
        const PositionManager& getPM() const { return m_pm; };

        // returns the trade qty after accounting of the current done_qty, open_qty and held_qty,
        // AND matching possible held instructions and open orders if possible.
        // Note 1 - this cancels the open orders and removes the held instructions if needed
        // and therefore is not reentrant consistent. See checkTargetQty()
        // Note 2 - if target_px is not given (left as nullptr), it matches ANY open orders, 
        // otherwise it is used to match. 
        int64_t updateWithTargetQty(int64_t target_qty, const std::string& algo, const std::string& symbol, double* target_px=nullptr);

        // returns the trader qty after accounting of the current done_qty, open_qty and held_qty,
        // NO matching is performed.  
        // Note - this is a read only operation, does not update any state, See updateWithTargetQty()
        int64_t checkTargetQty(int64_t target_qty, const std::string& algo, const std::string& symbol) const;

        // matches any current open order on the other side with an optional limit_px, return 
        // remaining qty.  The matching strategy has to be in the same POD.  Matched OO canceled,
        // Synthetic fills will be generated if not the same strategy.
        int64_t matchOpenOrders(const std::string& algo, const std::string& symbol, int64_t qty, double* px=nullptr);

        // getters
        const std::shared_ptr<PositionInstruction> getEIByClOrdId(const std::string& clOrdId) const;
        bool isAlgoStopped(const std::string& algo) const;

        // gets a peg price of oo, given whether to peg pass, 
        // maximum number of ticks to be in the front before
        // going aggressive and the bp_threshold.  
        // The bp_threshold from 0 to 1, default 0.5, going agg 
        // if bp more than book pressure. 
        // 1 always passive, 0 always agg
        double getPegPxBP(const std::shared_ptr<const OpenOrder>& oo, bool peg_passive, int max_peg_tick_diff=1, double bp_threshold=0.5) const;

        friend class FloorBase;
        friend class FloorManager;
        bool isFloorManager() const { return false; }; // to be overload by Floormanger

    protected:
        // states limited access by derived
        bool m_started, m_loaded, m_eod_pending;
        time_t m_loaded_time;
        bool m_stale_order;

        bool requestReplay(const std::string& loadUtc, std::string* errstr = nullptr);
        void clearAllOpenOrders();
        enum {
            PendingOrderTimeOutSec    = 5,      // unacklast_utc of pi compares with current utc
        };

        /* TODO - in case the friend class doesn't work
        const std::unordered_map<std::string, std::shared_ptr<PositionInstruction>>& getOrderMap() const;
        void deleteOO(const char* clOrdId);
        */

    private:
        PositionManager m_pm;
        volatile bool m_should_run;
        std::string m_recovery_file;

        // The purpose of m_orderMap is to 1. decide if an OO should be scanned.
        // 2. check the if an entered order has been updated within a timeout
        // It is important to have such check as some lower level rejects may not
        // be visible to the fix parser callback.
        std::unordered_map<std::string, std::shared_ptr<PositionInstruction>> m_orderMap;
        std::set<std::string> m_algo_stop_set;

        FloorCPR(const FloorCPR& cpr) = delete;
        FloorCPR& operator=(const FloorCPR& cpr) = delete;
        // ***
        // main handle function from run_one_loop()
        // ***
        void handleMessage(MsgType& msg_in);

        // ***
        // specific request handlers
        // ***

        void setInitialSubscriptions(); // set to receive er and user 
        void addPositionSubscriptions();  // set to receive position requests 
        void handleExecutionReport(const MsgType& msg);
        void handleExecutionReport(const pm::ExecutionReport& er);
        void handleUserReq(const MsgType& msg);
        void handleAlgoUserReq(const MsgType& msg);
        void handleStatusSet(const MsgType& msg);

        // ***
        // helpers to send requests
        // ***
        bool requestOpenOrder(std::string* errstr=nullptr);
        bool loadRecoveryFromFill(const std::string& loadUtc, std::string* errstr = nullptr);

        int64_t sendOrder_outContract(const std::shared_ptr<PositionInstruction>& pi, const IntraDayPosition& pos);
        bool sendOrder_InContract(int64_t trade_qty, const std::shared_ptr<PositionInstruction>& pi);
        bool scanOrderMap(time_t cur_utc) const;


        // TWAP traders
        /*
        int getLotsPerMin(const std::string& tradable_symbol) const;
        std::pair<int, double> getTWAP2Trades(const std::shared_ptr<PositionInstruction>& pi, int max_px_diff_spd=MaxPISpreadDiff) const;
        bool removePI(const std::string& algo, const std::string& symbol);
        bool updatePI(const std::shared_ptr<PositionInstruction>& pi);
        double getPegPx(const std::shared_ptr<const OpenOrder>& oo, bool peg_passive) const;
        double getPegPxBP(const std::shared_ptr<const OpenOrder>& oo, bool peg_passive) const;
        bool shouldScan(const std::shared_ptr<const OpenOrder>& oo, bool& peg_passive) const;
        std::vector<std::pair<time_t,long long>> getTWAPSlice(int64_t trade_qty, const std::string& tradable_symbol, time_t target_utc, int lot_seconds = 60) const;
        std::string handleUserReqMD(const char* cmd);
        enum {
            // execution related parameters
            ScanIntervalMicro    = 1000,    // open order scan interval
            RunIntervalMicro     = 10000,   // main loop sleep micro
            PassiveIntervalMicro = 75000,   // passive interval micro
            AggAllIntervalMicro  = 5000000, // aggressive all interval micro
            MaxPegSpreadDiff     = 3,       // maximum spreads before go aggressive in peg
            MaxPISpreadDiff      = 5,       // maximum spreads before go aggressive in PositionInstruction
            PegAggIntervalMilli  = 5000,      // OO Peg, wait between consecutive aggressive
            LotsPerMinFraction   = 8,       // fraction of per-minute volume as target rate
            MAXPassiveSize       = 2,       // maximum size showing at passive side
            MaxSwipeLevels       = 5,       // maximum number of levels when aggressive
            PassiveSizeFraction  = 50,      // fraction of bbo size as a minimum passive size
            // floor related parameters
            PendingOrderTimeOutSec      = 10,      // last_utc of pi compares with current utc
        };
        bool scanOpenOrders();
        // this is the main function where orders are create towards the target position
        // order size is determined by trading logic such as twap, sent by sendOrder_InOut(size)
        void addPositionInstruction(const std::shared_ptr<PositionInstruction>& pos_inst);
        bool handlePI(std::shared_ptr<PositionInstruction>& pi);
        const std::shared_ptr<PositionInstruction> getPI(const std::string& algo, const std::string& symbol) const;
        */
    };


    /*
     * templated implementations
     */

    template<typename Derived> 
    FloorCPR<Derived>::FloorCPR(const std::string& name)
    : FloorBase(name, true),
      m_started(false), m_loaded(false), 
      m_eod_pending(false),
      m_loaded_time(0),
      m_pm(m_name),
      m_should_run(false)
    {};

    template<typename Derived> 
    FloorCPR<Derived>::~FloorCPR() {};

    template<typename Derived> 
    void FloorCPR<Derived>::start() {
        // process of startup
        // 1. create posisiotn manager and load eod
        // 2. create channel and start to process er
        // 3. send request for replay
        // 4. wait for done
        // 5. process recovery
        // 6. request open order
        // 7. Mark SoD load done
        // 8. wait for 2 seconds before accepting position requests
        // 9. enter the loop

        if (m_started) {
            logInfo("Floor already started!");
            return;
        }
        m_started = true;
        m_loaded = false;
        m_should_run = true;
        m_eod_pending = false;
        m_stale_order = false;
        m_loaded_time = 0;
        std::string errstr;
        pm::risk::Monitor::get().set_instance_name(m_name);
        while (m_should_run && (!requestReplay(m_pm.getLoadUtc(), &errstr))) {
            logError("problem requesting replay: %s, retrying in next 5 seconds", errstr.c_str());
            utils::TimeUtil::micro_sleep(5*1000000);
            continue;
        };
        while (m_should_run && (!requestOpenOrder(&errstr))) {
            logError("problem requesting open orders download: %s, retrying in next 5 seconds", errstr.c_str());
            utils::TimeUtil::micro_sleep(5*1000000);
            continue;
        }

        setInitialSubscriptions();
        int stale_check_utc = 0;
        while (m_should_run) {
            // this calls the handleMessage(), 
            // return true if processed one, false if idle
            bool has_message = this->run_one_loop(*this);
            if (__builtin_expect(!has_message,1)) {
                // idle, do checks and call derived
                if (__builtin_expect(!m_loaded,0)) {
                    // wait until the replay is done
                    utils::TimeUtil::micro_sleep(1000*1000);
                    continue;
                }
                auto cur_utc = utils::TimeUtil::cur_utc();
                if (__builtin_expect(cur_utc > stale_check_utc,0)) {
                    // this checks the unacknowledged orders
                    stale_check_utc = cur_utc;
                    m_stale_order = !scanOrderMap(cur_utc);
                    if (m_stale_order) {
                        logError("stale order detected, floor pause for trading!");
                        // this pauses trading for all globally
                        pm::risk::Monitor::get().status().notify_pause("","",true);
                    }
                }
                // all good, call derived
                static_cast<Derived*>(this)->run_loop_derived();
            }
        }

        // done for the loop, exiting
        logInfo("FloorCPR loop: Stop received, exit.");
        clearAllOpenOrders();
        static_cast<Derived*>(this)->shutdown_derived();
        m_started = false;
        m_loaded = false;
    }

    template<typename Derived> 
    void FloorCPR<Derived>::handleMessage(MsgType& msg_in) {
        // the messages could be from user, the tp or the algo
        // refer to the FloorBase for msg types
        // Each type has a FloorCPR handling and a derived handling. 
        // The msgs from tp, such as execution reports are handled at FloorCPR
        // Some user control msg (such as pause) and algo (such as stop)
        // are handled at FloorCPR
        switch (msg_in.type) {
        case ExecutionReport:
        {
            // such message doesn't need response
            handleExecutionReport(msg_in);
            break;
        }
        case FloorBase::ExecutionReplayDone: 
        {
            if (!m_eod_pending) {
                // start up recovery
                logInfo("FloorCPR got ExecutionReplayDone, replaying fills to PositinManager!");

                // the replayed er updates risk only if configured
                const bool risk_update_recovery = !risk::Monitor::get().config().m_skip_replay;
                m_pm.loadRecovery(m_recovery_file, false, risk_update_recovery);

                m_loaded_time = time(nullptr);
                clearAllOpenOrders();
                m_loaded = true;
                addPositionSubscriptions();
                static_cast<Derived*>(this)->start_derived();

                logInfo("%s accepting position requests, starting with position: \n%s",
                        m_name.c_str(), 
                        m_pm.toString(nullptr, nullptr, true).c_str());
            } else {
                std::string difflog;
                if (!m_pm.reconcile(m_recovery_file, difflog, false)) {
                    logError("%s failed to reconcile! \nrecovery_file: %s\ndiff:%s", 
                            m_name.c_str(), m_recovery_file.c_str(), difflog.c_str());
                } else {
                    m_pm.persist();
                    logInfo("before resetting daily pnl:\n%s", m_pm.toString(nullptr, nullptr, true).c_str());
                    m_pm.resetPnl();
                    logInfo("after resetting daily pnl:\n%s", m_pm.toString(nullptr, nullptr, true).c_str());
                    logInfo("%s EoD Done!", m_name.c_str());
                }
                m_eod_pending = false;
            }
            break;
        }
        case FloorBase::UserReq :
        {
            // this mainly handled by FloorManager
            // who is responsible in reponding the request
            // except some usere control req, such as pause
            // are handled and responded by CPR
            handleUserReq(msg_in);
            break;
        }
        case FloorBase::AlgoUserCommand:
        {
            // the request from the algo, they will be
            // handled by algo thread and responded from there
            // this gets a copy of requests and process at floor
            // no need to respond
            handleAlgoUserReq(msg_in);
            break;
        }
        case FloorBase::GetPositionReq :
        case FloorBase::SetPositionReq :
        {
            if (static_cast<Derived*>(this)->handlePositionReq_derived(msg_in, m_msgout)) {
                m_channel->update(m_msgout);
            }
            break;
        }
        case FloorBase::TradingStatusNotice:
        {
            // Set pause status
            //     Z algo, symbol ,ON|OFF
            // algo or symbol could be empty, matches all
            // ':' delimited string
            // no need to response
            handleStatusSet(msg_in);
            break;
        }

        default:
            logError("%s received a unknown message: %s", 
                    m_name.c_str(), msg_in.toString().c_str());
            break;
        }
    }

    // helpers
    template<typename Derived> 
    void FloorCPR<Derived>::setInitialSubscriptions() {
        std::set<int> type_set;
        type_set.insert((int)FloorBase::ExecutionReport);
        type_set.insert((int)FloorBase::UserReq);  // only limited control commands
        type_set.insert((int)FloorBase::AlgoUserCommand);
        type_set.insert((int)FloorBase::ExecutionReplayDone);
        type_set.insert((int)FloorBase::TradingStatusNotice);
        subscribeMsgType(type_set);
    }

    template<typename Derived> 
    void FloorCPR<Derived>::addPositionSubscriptions() {
        // CPR doesn't subscribe to any position instructions
        static_cast<Derived*>(this)->addPositionSubscriptions_derived();
    }

    template<typename Derived> 
    void FloorCPR<Derived>::handleExecutionReport(const MsgType& msg) {
        const auto* er((pm::ExecutionReport*)(msg.buf));
        handleExecutionReport(*er);
    }

    template<typename Derived> 
    void FloorCPR<Derived>::handleExecutionReport(const pm::ExecutionReport& er) {
        const bool update_risk = true;
        const bool do_pause_notify = !static_cast<Derived*>(this)->isFloorManager();
        // don't run notify for FloorManager, to avoid it waiting for ack for
        // other FloorCPR instances who are also waiting for ack of notify
        bool is_newfill = m_pm.update(er, false, update_risk, do_pause_notify);
        const std::string& clOrdId (er.m_clOrdId);
        auto iter = m_orderMap.find(clOrdId);
        if (iter != m_orderMap.end()) {
            auto&pi(iter->second);
            // update the last utc of this pi, used to alert for order
            // sent that was not heard back from for a while
            pi->last_utc = 0;  // disable the check, as order was updated

            // update fill stats if it's a new fill
            if (is_newfill) {
                pi->addFill(er);
            }
            // remove m_orderMap if clOrdId is no longer open in pm
            if (!m_pm.getOO(clOrdId)) {
                logDebug("removed pi %s (clOrdId=%s)from m_orderMap",
                        pi->toString().c_str(),
                        clOrdId.c_str());
                // print fill stats 
                logInfo("%s", iter->second->dumpFillStat().c_str());
                m_orderMap.erase(iter);
            } else {
                if (__builtin_expect(er.isReject(),0)) {
                    // could be reject of new order or reject of replace
                    // cannot be reject of cancel since cancel's clOrdId doesn't enter m_orderMap
                    // note if a CancelReplace is rejected, the
                    // original order is still open in both m_order and pm's open order list
                    // the original clOrId's pi in m_orderMap points to the same shared pointer
                    // its last_utc set to 0 as above.
                    logInfo("handleExecutionReport(): removed rejected order: %s (clOrdId=%s)from m_orderMap",
                            pi->toString().c_str(),
                            clOrdId.c_str());
                    m_orderMap.erase(iter);
                }
            }
        }
        static_cast<Derived*>(this)->handleExecutionReport_derived(er);
    }

    template<typename Derived> 
    void FloorCPR<Derived>::handleUserReq(const MsgType& msg) {
        const char* cmd = msg.buf;
        logDebug("%s got user command: %s", m_name.c_str(), msg.buf);
        m_msgout.type = FloorBase::UserResp;
        m_msgout.ref = msg.ref;

        // "Ack" is the default good response
        std::string respstr("Ack");
        switch (cmd[0]) {
            case 'K' : 
            {
                stop();
                break;
            }
            case 'A':
            {
                //A position adjustment
                //algo, symbol, tgt_qty, tgt_vap

                const auto& tk(utils::CSVUtil::read_line(cmd+1));
                if (tk.size() != 4) {
                    std::string errstr = "Problem parsing a position adjustment: " + std::string(cmd) + "\nExecpted: algo, symbol, tgt_qty, tgt_vap";
                    logError("%s", errstr.c_str());
                    respstr = errstr;
                    break;
                }

                const std::string& algo (tk[0]);
                const std::string& symbol(tk[1]);
                int64_t tgt_qty = std::stoll(tk[2]), qty;
                double tgt_vap, px;
                if (!md::getPriceByStr(symbol, tk[3].c_str(), tgt_vap)) {
                    respstr = std::string("Cannot parse the price string ") + tk[3];
                    logError("%s", respstr.c_str());
                    break;
                }
                const auto& idp_arr (m_pm.listPosition(&algo, &symbol));
                if (idp_arr.size() != 1) {
                    logInfo("Adjusting a non-existing position - simply adding");
                    qty = tgt_qty;
                    px = tgt_vap;
                } else {
                    qty = idp_arr[0]->tgt_fill(tgt_qty, tgt_vap, &px);
                }
                if (!qty) {
                    logInfo("Position already meets target, nothing to do");
                    break;
                }
                try {
                    // generating a fill and add to the erpersist and update
                    const auto& er (ExecutionReport::genSyntheticFills(symbol, algo, qty, px, "PA"));
                    // append it to the er report
                    utils::CSVUtil::write_line_to_file(er.toCSVLine(), ExecutionReport::ERPersistFile(), true);
                    handleExecutionReport(er);
                } catch (std::exception& e) {
                    respstr = std::string("Failed to ajust position: ") + e.what();
                    logError("%s : %s", cmd, respstr.c_str());
                    break;
                }
                break;
            }
            case 'Y':
            {
                // get or set operator id
                // Y [operator_id]
                const auto& tk (utils::CSVUtil::read_line(cmd+1));
                if (tk.size() == 0) {
                    // getting current operator id
                    respstr = risk::Monitor::get().status().get_operator_id();
                    break;
                }
                if (__builtin_expect(tk.size() != 1,0)) {
                    respstr = "set operator id not understood. \nY [operator_id]";
                    logError("set operator id not understtood: %s\n%s", cmd, respstr.c_str());
                    break;
                }
                risk::Monitor::get().status().set_operator_id(*this, tk[0]);
                break;
            }
            case 'Z':
            {
                // user query for trading status, format
                //     Z algo, mkt
                //  algo/mkt can be empty, match all, or a ':' delimited list
                // Note the 'Set' is sent via msg type TradingStatusNotice
                // see RiskMonitor.cpp:notify_pause() and flr.h for detail
                // The reason is that to make 'Set' command to be sent without
                // requiring ack from the receiver(s). This is needed for
                // Feedhandlers to send such requests. 
                const auto& tk (utils::CSVUtil::read_line(cmd+1));
                if (__builtin_expect(tk.size() != 2,0)) {
                    respstr = "Pause command not understood. \nZ algo,symbol\nalgo or symbol could be empty, matches all";
                    logError("Error command %s\n%s", cmd, respstr.c_str());
                    break;
                }
                respstr = risk::Monitor::get().status().queryPause(tk[0], tk[1]);
                break;
            }
            default :
            {
                if (!static_cast<Derived*>(this)->handleUserReq_derived(msg, respstr)) {
                    respstr = "";
                }
                break;
            }
        }
        if (respstr.length() > 0) {
            m_msgout.copyString(respstr);
            m_channel->update(m_msgout);
        }
    }

    template<typename Derived> 
    void FloorCPR<Derived>::handleAlgoUserReq(const MsgType& msg) {
        // get the strategy start/stop command for floor
        const char* cmd = msg.buf;
        logInfo("%s got algo user command: %s", m_name.c_str(), msg.buf);

        /*
        m_msgout.type = FloorBase::UserResp;
        m_msgout.ref = msg.ref;
        std::string respstr("Ack");
        */

        const auto tk = utils::CSVUtil::read_line(cmd, ' ');
        const auto& sn(tk[0]);
        if (strcmp(sn.c_str(), "L")==0) {
            return;
        }

        if (tk[1] == "E") {
            logInfo("Floor blocks trading from %s", tk[0].c_str());
            m_algo_stop_set.insert(tk[0]);
        } else {
            if (tk[1] == "S") {
                logInfo("Floor removes %s from block list", tk[0].c_str());
                m_algo_stop_set.erase(tk[0]);
            }
        }
    }

    template<typename Derived> 
    void FloorCPR<Derived>::handleStatusSet(const MsgType& msg) {
        static std::string s_prev_cmd;
        const char* cmd = msg.buf;
        if ((s_prev_cmd.size()>0) && (strncmp(s_prev_cmd.c_str(), cmd, s_prev_cmd.size())==0)) {
            return;
        }
        s_prev_cmd = std::string(cmd);

        // Set pause status
        // Z algo,symbol,ON|OFF, algo or symbol could be empty, matches all
        const auto& tk (utils::CSVUtil::read_line(cmd+1));
        if (__builtin_expect(tk.size() != 3,0)) {
            std::string respstr = "Pause command not understood. \nZ algo,symbol,ON|OFF\nalgo or symbol could be empty, matches all";
            logError("TradingStatusSet error command %s\n%s", cmd, respstr.c_str());
            return;
        }

        const std::string& set_status(tk[2]);
        bool ret = false;
        if (strncmp(set_status.c_str(), "ON", set_status.size())==0) {
            ret = risk::Monitor::get().status().setPause(tk[0], tk[1], true);
        } else if (strncmp(set_status.c_str(), "OFF", set_status.size())==0) {
            ret = risk::Monitor::get().status().setPause(tk[0], tk[1], false);
        } else {
            logError("TradingStatusSet error command %s\nunknown pause status %s, either ON or OFF.", cmd, set_status.c_str());
            return;
        }
        // persist if FM
        if (ret && static_cast<Derived*>(this)->isFloorManager()) {
            risk::Monitor::get().status().persist_pause(
                    *static_cast<const Derived*>(this), cmd+1);
        }
    }

    template<typename Derived> 
    void FloorCPR<Derived>::sendOrder_InOut(const std::shared_ptr<PositionInstruction>& pi) {
        // if need to trade
        logInfo("sendOrder_InOut(): got instruction: %s", pi->toString().c_str());
        int64_t trade_qty = pi->qty;
        if (__builtin_expect(trade_qty==0,0)) {
            return;
        }

        // check if we have out contract to be traded out first
        const auto & outin_vec(m_pm.listOutInPosition(pi->algo, pi->symbol));
        for (const auto& pos : outin_vec) {
            const auto& pi_tradable (utils::SymbolMapReader::get().getTradableSymbol(pi->symbol));
            if (pi_tradable != pos->get_symbol()) {
                // out contract, this will trade out the out contract as much as possible
                // and adjust the trade_qty
                trade_qty = sendOrder_outContract(pi, *pos);
                break;
            }
        }

        // in contract
        sendOrder_InContract(trade_qty, pi);
    }

    template<typename Derived> 
    std::string FloorCPR<Derived>::sendOrderByString(const char* bsstr, std::string* clOrdId) {
        // expect bssr to be in format of 
        // B|S algo, symbol, qty, px_str [,clOrdId]
        // if clOrdId is not in bsstr, it is generated
        // and assigned to *clOrdId if not null
        if (__builtin_expect(m_stale_order, 0)) {
            logError("stale order detected, not sending order for %s", bsstr?bsstr:"");
            return "stale order detected, not sending order";
        }
        try {
            if ((*bsstr == 'B') || (*bsstr == 'S')) {
                // an order string: B|S algo,symbol,qty,px
                // parse and check the order before sending
                int side = ((*bsstr == 'B')?1:-1);

                const auto& tk(utils::CSVUtil::read_line(bsstr+1));
                std::string ordId;
                if (tk.size() == 4) {
                    ordId = ExecutionReport::genClOrdId();
                    if (clOrdId) {
                        *clOrdId = ordId;
                    }
                } else if (tk.size() == 5) {
                    ordId = tk[4];
                } else {
                    return std::string("Order string has to have 4 or 5 comma delimited fields: algo, symbol, qty, px_str [,clOrdId]. Not accepting the string: ") + std::string(bsstr);
                }

                // get the pi if there were one. We need to
                // do a book keeping here, as the order maybe
                // matched internally and therefore not being sent
                // so we take a copy here and reinsert if needed
                std::shared_ptr<PositionInstruction> pi;
                auto iter = m_orderMap.find(ordId);
                if (iter!=m_orderMap.end()) {
                    pi = iter->second;
                    m_orderMap.erase(iter);
                }

                const std::string& algo(tk[0]);
                const std::string& symbol(utils::SymbolMapReader::get().getTradableSymbol(tk[1]));
                if (!symbol.size()) {
                    return std::string("problem sending order: symbol not found " + tk[1]);
                }

                int qty = std::stod(tk[2]);
                double px;
                if (! md::getPriceByStr(symbol, tk[3].c_str(), px) ) {
                    return std::string("problem parsing price string: ") + std::string(bsstr);
                }
                // format good, do stop/risk check here
                if (__builtin_expect(m_algo_stop_set.count(algo),0)) {
                    logError("Trading %s blocked by user", algo.c_str());
                    return algo + std::string(" blocked by user, not trading.");
                }
                // match self trade if possible
                qty = matchOpenOrders(algo, symbol, qty*side, &px) * side;
                if (__builtin_expect(qty,1)) {
                    if (__builtin_expect(!risk::Monitor::get().checkNewOrder(algo, symbol, qty * side, m_pm),0)) {
                        // errors logged in check
                        return std::string("Risk failed for request! ") + std::string(bsstr);
                    }
                    // recreat the price string in caes needed
                    char ordstr[256];
                    size_t bytes = snprintf(ordstr, sizeof(ordstr), "%c %s, %s, %d, %s, %s", *bsstr, algo.c_str(), symbol.c_str(), qty, PriceCString(px),ordId.c_str());
                    FloorBase::MsgType req(FloorBase::SendOrderReq, ordstr, bytes + 1);
                    FloorBase::MsgType resp;
                    if (!m_channel->requestAndCheckAck(req, resp, 1, FloorBase::SendOrderAck)) {
                        return std::string("problem sending order: ") + std::string(req.buf);
                    }
                    // save the map
                    if (pi) {
                        // update the last utc
                        pi->last_utc = utils::TimeUtil::cur_utc();
                        m_orderMap[ordId] = pi;
                    }
                }
            } else {
                logError("Order string not starting from B or S: %s", bsstr);
                return "Order string not starting from B or S";
            }
        } catch (const std::exception& e) {
            logError("Exception when send order string %s: %s", bsstr, e.what());
            return std::string("Exception when send order string ") + std::string(bsstr) + " : " + std::string( e.what());
        }
        return "";
    }

    template<typename Derived> 
    std::string FloorCPR<Derived>::sendCancel(const char* clOrdId) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "C %s", clOrdId);
        return sendCancelReplaceByString(std::string(cmd));
    }

    template<typename Derived> 
    std::string FloorCPR<Derived>::sendCancelReplacePx(const char* clOrdId, double new_price) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "R %s,,%f", clOrdId, new_price);
        return sendCancelReplaceByString(std::string(cmd));
    }

    template<typename Derived> 
    std::string FloorCPR<Derived>::sendCancelReplaceSize(const char* clOrdId, uint64_t size) {
        // size positive, same sign with original
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "R %s,%lld,", clOrdId, (long long)size);
        return sendCancelReplaceByString(std::string(cmd));
    }

    template<typename Derived> 
    std::string FloorCPR<Derived>::sendCancelReplaceByString(const char* bsstr) {
        // bsstr is one of the following form:
        // C clOrdId 
        // R clOrdId, qty, px
        // where qty must be positive, same sign with orinial order
        // qty and px could be an empty string, which does not change
        //
        // Note however the 'qty' in the final string sending to tpmain 
        // is sign significant, see sendOrder() in CEngine.cpp
        //
        // Note if clOrdId is not provided in cancel, all open
        // orders are cancelled.
        try {
            if ((*bsstr == 'C') || (*bsstr == 'R')) {
                const auto& line = utils::CSVUtil::read_line(std::string(bsstr+1));
                // handle cancel all if no clOrdId is provided
                if ((*bsstr == 'C') && (line.size() == 0)) {
                    logInfo("sendCancelReplaceByString() clearing all open orders!");
                    clearAllOpenOrders();
                    return "";
                }

                if (((*bsstr == 'C') && (line.size()!= 1)) ||
                    ((*bsstr == 'R') && (line.size()!= 3))) {
                    logError("Error parsing the cancel/eplace order string %s!", bsstr);
                    return std::string("Error parsing cancel/replace ") + std::string(bsstr);
                }
                const auto& oo (m_pm.getOO(line[0]));
                if (!oo) {
                    logError("Cannot find clOrdId %s", line[0].c_str());
                    return std::string("clOrdId not found: ") + line[0];
                }
                const std::string& algo (oo->m_idp->get_algo());
                const std::string& symbol(oo->m_idp->get_symbol());
                int64_t qty = oo->m_open_qty; // sign significant
                int64_t ord_qty = oo->m_ord_qty; // sign significant
                double px = oo->m_open_px;

                char ordStr[256];
                size_t bytes = 0;
                if (*bsstr == 'C') {
                    bytes = snprintf(ordStr, sizeof(ordStr), "C %s, %s", line[0].c_str(), algo.c_str());
                } else {
                    int64_t qty_delta = 0;
                    if ((line[1] == "") && (line[2] == "")) {
                        logError("Both qty and px are emtpy, nothing to be replaced");
                        return "nothing to be replaced";
                    }
                    if (line[1] != "") {
                        // replace qty
                        int64_t qty_new = std::stoll(line[1]);
                        if (qty_new < 0) {
                            logError("Replace qty (%lld) must be positive, same side as the original order", (long long)qty_new);
                            return "replace qty must be positive";
                        }
                        // now make qty to be signed
                        qty_new *= (oo->m_open_qty>0?1:-1);

                        // replace uses order qty
                        qty_delta = qty_new - qty;
                        ord_qty += qty_delta;
                        qty = qty_new;
                    }
                    if (line[2] != "") {
                        // replace px, parse line[2] as px_str
                        if (!md::getPriceByStr(symbol, line[2].c_str(), px)) {
                            logError("Failed to parse the replacement price string %s", line[2].c_str());
                            return std::string("Failed to parse the replacement price string ") + line[2];
                        }
                    }

                    // if we have anything to be replaced
                    if (__builtin_expect((qty == oo->m_open_qty) && (std::abs(px - oo->m_open_px)<1e-10),0)) {
                        logError("Nothing to be replaced from %s with command %s!", oo->toString().c_str(), bsstr);
                        return "nothing to be replaced!";
                    }

                    if (!risk::Monitor::get().checkReplace(oo->m_idp->get_algo(), oo->m_idp->get_symbol(), qty_delta, m_pm)) {
                        //errors logged in the function
                        return std::string("Risk failed for request! ") + std::string(bsstr);
                    }

                    const std::string replaceClOrdId = ExecutionReport::genReplaceClOrdId(line[0]);
                    bytes = snprintf(ordStr, sizeof(ordStr), "R %s, %lld, %s, %s, %s, %s", line[0].c_str(), (long long)ord_qty, PriceCString(px), algo.c_str(), symbol.c_str(), replaceClOrdId.c_str());

                    // update m_orderMap if necessary
                    if (qty != 0) {
                        auto iter = m_orderMap.find(line[0]);
                        if (iter != m_orderMap.end()) {
                            auto& pi(iter->second);
                            pi->last_utc=utils::TimeUtil::cur_utc();
                            m_orderMap[replaceClOrdId] = pi;
                        }
                    }
                }
                FloorBase::MsgType req(FloorBase::SendOrderReq, ordStr, bytes+1);
                FloorBase::MsgType resp;
                if (!m_channel->requestAndCheckAck(req, resp, 1, FloorBase::SendOrderAck)) {
                    return std::string("problem sending order: ") + std::string(req.buf);
                }
            } else {
                logError("CancelReplace not starting from C or R: %s", bsstr);
                return "CancelReplace not starting from C or R";
            }
        } catch (const std::exception& e) {
            logError("Exception when send cancel or replace string %s: %s", bsstr, e.what());
            return std::string("Exception when send cancel or replace string ") + std::string(bsstr) + " : " + std::string(e.what());
        }
        return "";
    }

    template<typename Derived> 
    std::string FloorCPR<Derived>::sendOrder(const bool isBuy, 
            const char* algo, const char* symbol,
            int64_t qty, double px, std::string* clOrdId) {

        char buf[256];
        snprintf(buf, sizeof(buf), "%c %s,%s,%lld,%s%s",
                isBuy?'B':'S', algo, symbol,
                (long long)qty, PriceCString(px), 
                ((clOrdId&&(clOrdId->size()>0))?(std::string(",")+*clOrdId):std::string("")).c_str());
        return sendOrderByString(buf, clOrdId);
    }

    template<typename Derived> 
    int64_t FloorCPR<Derived>::sendOrder_outContract(const std::shared_ptr<PositionInstruction>& pi, const IntraDayPosition& pos) {
        logInfo("sendOrde_OutContract(): trying to reduce IDP(%s) by PI(%s)", pos.toString().c_str(), pi->toString().c_str());
        auto trade_qty = pi->qty;
        int64_t qty = pos.getPosition(), oqty = pos.getOpenQty();
        qty += oqty;
        if (qty * trade_qty < 0) {
            // we could reduce out-contract position
            if (std::abs(qty) > std::abs(trade_qty)) {
                qty = -trade_qty;
            }
            auto pi_out = std::make_shared<PositionInstruction>(*pi);
            pi_out->type = PositionInstruction::PASSIVE;
            std::strcpy(pi_out->symbol, pos.get_symbol().c_str());
            logInfo("sending out contract PI(%s) qty(%d)", pi_out->toString().c_str(), (int)-qty);
            if (sendOrder_InContract(-qty, pi_out)) {
                trade_qty += qty;
            }
        }
        return trade_qty;
    }

    template<typename Derived> 
    bool FloorCPR<Derived>::sendOrder_InContract(int64_t trade_qty, const std::shared_ptr<PositionInstruction>& pi) {
        if (__builtin_expect(trade_qty==0, 0)) {
            return true;
        }

        // figure out the price
        const bool isBuy = (trade_qty > 0);
        double px = pi->px;
        if (pi->type == PositionInstruction::MARKET) {
            if (!md::getPriceByStr(pi->symbol, std::string(isBuy?"a+t10":"b-t10").c_str(), px)) {
                logError("Error getting market order price for %s, order of %s not sent", pi->symbol, pi->toString().c_str());
                return false;
            }
        } else if ((pi->type == PositionInstruction::PASSIVE) ||
                   (pi->type == PositionInstruction::TWAP)) {

            logInfo("sending order for pi (%s)",pi->toString().c_str());
            if (!md::getPriceByStr(pi->symbol, std::string(isBuy?"b+t0":"a+t0").c_str(), px)) {
                logError("Error getting market order price for %s, order of %s not sent", pi->symbol, pi->toString().c_str());
                return false;
            }
        }

        trade_qty = (isBuy? trade_qty : -trade_qty); // trade_qty is now positive
        std::string clOrdId = ExecutionReport::genClOrdId((int)pi->type, 0);

        // save the map
        logDebug("Adding to orderMap at sendOrder_InContract: %s: %s", clOrdId.c_str(), pi->toString().c_str());
        m_orderMap[clOrdId] = pi;

        const auto& errstr = sendOrder(isBuy, pi->algo, pi->symbol, trade_qty, px, &clOrdId);
        if ((errstr.size() > 0) || (clOrdId.size() == 0)) {
            logError("Error sending order: %s", errstr.c_str());
            return false;
        }
        return true;
    }

    template<typename Derived> 
    bool FloorCPR<Derived>::scanOrderMap(time_t cur_utc) const {
        // order timeout set to 
        for (const auto& order : m_orderMap) {
            const auto& pi(order.second);
            if (__builtin_expect(pi->last_utc == 0,1)) continue;
            if ((int)((int) cur_utc - (int)pi->last_utc) > PendingOrderTimeOutSec) {
                logError("detected stale order: %s - %s", order.first.c_str(), pi->toString().c_str());
                return false;
            }
        }
        return true;
    }

    template<typename Derived> 
    void FloorCPR<Derived>::clearAllOpenOrders() {
        const auto& vec(m_pm.listOO());
        logInfo("Canceling all %d open order(s)...", (int) vec.size());
        for (const auto& oo: vec) {
            uint64_t qty = oo->m_open_qty;
            if (!qty) {
                continue;
            }
            logInfo("Canceling open order %s", oo->toString().c_str());
            char tstr[256];
            snprintf(tstr, sizeof(tstr), "C %s", oo->m_clOrdId);
            std::string errstr = sendCancelReplaceByString(tstr);
            if (errstr != "") {
                logError("failed to send cancel %s error = %s", 
                        tstr, errstr.c_str());
            }
            m_pm.deleteOO(oo->m_clOrdId);
        }
    }

    template<typename Derived> 
    bool FloorCPR<Derived>::requestReplay(const std::string& loadTime, std::string* errstr) {
        // Note loadTime is a string in *local* time
        // make start/stop time string in *readable* local time
        const std::string endTime = utils::TimeUtil::frac_UTC_to_string(0, 0);

        // compose request 
        m_recovery_file = loadTime+'_'+endTime+"_replay.csv";

#ifdef USE_TT_RECOVERY
        const std::string reqstr = loadTime + "," + m_pm.getRecoveryPath()+"/"+m_recovery_file;
        MsgType msgReq(FloorBase::ExecutionReplayReq, reqstr.c_str(), reqstr.size()+1);
        MsgType msgResp;
        if(!m_channel->requestAndCheckAck(msgReq, msgResp, 60,  FloorBase::ExecutionReplayAck) ) {
            if (errstr)
                *errstr = std::string(msgResp.buf);
            return false;
        }
        return true;
#else
        return loadRecoveryFromFill(loadTime, errstr);
#endif
    }

    template<typename Derived> 
    bool FloorCPR<Derived>::loadRecoveryFromFill(const std::string& loadTime, std::string* errstr) {
        const std::string endTime = utils::TimeUtil::frac_UTC_to_string(0, 0);
        m_recovery_file = loadTime+'_'+endTime+"_replay.csv";

        if (! pm::ExecutionReport::loadFromPersistence(loadTime, endTime, m_pm.getRecoveryPath()+"/"+m_recovery_file)) {
            if (errstr)
                *errstr = "problem dumping fills to file " + m_pm.getRecoveryPath()+"/"+m_recovery_file + " from " + loadTime + " to " + endTime;
            return false;
        }
        // Recover file is ready, update event
        MsgType msg_in;
        msg_in.type = FloorBase::ExecutionReplayDone;
        handleMessage(msg_in);
        return true;
    }

    template<typename Derived> 
    bool FloorCPR<Derived>::requestOpenOrder(std::string* errstr) {
        // this should be replayed
        const std::string reqstr = "ALL";
        MsgType msgReq(FloorBase::ExecutionOpenOrderReq, reqstr.c_str(), reqstr.size()+1);
        MsgType msgResp;
        if(!m_channel->requestAndCheckAck(msgReq, msgResp, 3,  FloorBase::ExecutionOpenOrderAck)) {
            if (errstr) {
                *errstr = std::string(msgResp.buf);
            }
            return false;
        }
        return true;
    }

    template<typename Derived> 
    std::string FloorCPR<Derived>::toString() const {
        char buf[1024];
        size_t cnt = snprintf(buf, sizeof(buf),
                "Floor %s [running: %s, stopping: %s, loaded: %s, eod_pending: %s, recovery_file: %s, time_loaded: %s]\n"
                "Position Dump\n",
                m_name.c_str(), 
                m_started?"Y":"N", 
                (!m_should_run)?"Y":"N", 
                m_loaded?"Y":"N", 
                m_eod_pending?"Y":"N",
                m_recovery_file.c_str(), 
                utils::TimeUtil::frac_UTC_to_string(m_loaded_time,0).c_str());

        // dump the m_orderMap
        cnt += snprintf(buf+cnt, sizeof(buf)-cnt, "order map size: %d\n", (int)m_orderMap.size());
        for (const auto& kv:m_orderMap) {
            cnt += snprintf(buf+cnt, sizeof(buf)-cnt, "\t%s:%s\n", kv.first.c_str(),kv.second->toString().c_str());
        }
        return std::string(buf) + m_pm.toString() + static_cast<const Derived*>(this)->toString_derived();
    }

    template<typename Derived> 
    void FloorCPR<Derived>::stop() {
        logInfo("FloorManager %s stop received", m_name.c_str());
        m_should_run = false;
    }

    /*
     * utilities for getting positions and matching open orders
     */
    template<typename Derived> 
    int64_t FloorCPR<Derived>::updateWithTargetQty(int64_t target_qty, const std::string& algo, const std::string& symbol, double* target_px_ptr) {
        auto trade_qty = checkTargetQty(target_qty, algo, symbol);
        if (trade_qty) {
            // try to match own open orders if exist
            double px_ = target_px_ptr? (*target_px_ptr) : ((trade_qty > 0)? -1e+10:1e+10);

            // this could cancel offseting open orders that satisfies matching rules:
            // - algo can be matched (same pod)
            // - px can be satisfied (open px equal or better than requested px)
            // In case of matching - 
            // - open orders are canceled
            // - if OO is from the same algo, nothing happens
            // - if OO is from a different algo (same pod), a pair of "synthetic" fills generated on the open px
            // - return remaining quanty after matching
            trade_qty = matchOpenOrders(algo, symbol, trade_qty, &px_);
        }
        return trade_qty;
    }

    template<typename Derived> 
    int64_t FloorCPR<Derived>::checkTargetQty(int64_t target_qty, const std::string& algo, const std::string& symbol) const {
        int64_t done_qty=0, open_qty=0, trade_qty=0, out_done_qty=0;
        done_qty = m_pm.getPosition(algo, symbol, nullptr, nullptr, &open_qty, &out_done_qty);
        trade_qty = target_qty - (done_qty + open_qty);
        return trade_qty;
    }

    template<typename Derived> 
    int64_t FloorCPR<Derived>::matchOpenOrders(const std::string& algo, const std::string& symbol, int64_t qty, double* px) {
        // match internal open orders and return the remaining qty after match
        // qty is sign significant, px is the limit price to be matched at
        // return qty remaining, also sign significant
        //
        // For each matched open order, it sends cancel and then
        // generates two synthetic fills on the matched amount.
        //
        // set px to be NULL to match any open order of algo/symbol.
        const auto& vec (m_pm.matchOO(symbol, qty, px, &algo));

        logDebug("got vector len (%d)", (int)vec.size());

        for (const auto& matched_pair : vec) {
            const auto& oo(matched_pair.first);
            const auto& qty0(matched_pair.second); // qty0 has opposite sign
            int64_t cur_qty = oo->m_open_qty;
            double cur_px = oo->m_open_px;

            logInfo("Matching OpenOrder %s. Total wanted: %lld, matched size: %lld", oo->toString().c_str(), (long long)qty, (long long)qty0);

            const auto& algo_ma(oo->m_idp->get_algo());
            const auto& pod(IntraDayPosition::get_pod_from_algo(algo));

            // don't bother for different pod
            if (pod != IntraDayPosition::get_pod_from_algo(algo_ma)) {
                logInfo("NOT matching OpenOrder %s. different POD", oo->toString().c_str());
                continue;
            }

            // cancel/replace it to cur_qty - qty0
            char tstr[256];
            snprintf(tstr, sizeof(tstr), "R %s,%lld,", oo->m_clOrdId, (long long)std::abs(cur_qty-qty0));
            std::string errstr = sendCancelReplaceByString(tstr);
            if (errstr != "") {
                logError("Failed to send replace order %s error: %s", 
                        tstr, errstr.c_str());
                continue;
            }

            // ONLY generate fake fills for different algo within same POD
            // no need for fake fills for same algo matching 
            bool gen_synthetic_fills = (algo != algo_ma);
            if (gen_synthetic_fills) {
                // generates two fills and publish them
                const auto& er0 (ExecutionReport::genSyntheticFills(symbol, algo, -qty0, cur_px, "IM0"));
                const auto& er1 (ExecutionReport::genSyntheticFills(oo->m_idp->get_symbol(), oo->m_idp->get_algo(), qty0, cur_px, "IM1"));
                // append it to the er report
                utils::CSVUtil::write_line_to_file(er0.toCSVLine(), ExecutionReport::ERPersistFile(), true);
                utils::CSVUtil::write_line_to_file(er1.toCSVLine(), ExecutionReport::ERPersistFile(), true);
                
                // update position
                handleExecutionReport(er0);
                handleExecutionReport(er1);
            }

            // adjust the qty
            qty += qty0;
        }
        return qty;
    }

    template<typename Derived>
    double FloorCPR<Derived>::getPegPxBP(const std::shared_ptr<const OpenOrder>& oo, bool peg_passive, int max_peg_tick_diff, double bp_threshold) const {
        // go aggressive if expired or price moved away
        // otherwise, check bp and trade accordingly
        double px = oo->m_open_px;
        int64_t qty = oo->m_open_qty;

        const std::string& symbol(oo->m_idp->get_symbol());
        double bidpx, askpx;
        int bidsz, asksz;
        if (__builtin_expect(!md::getBBO(symbol,  bidpx, bidsz, askpx, asksz),0)) {
            logError("Cannot get bbo for %s, Open Order not scanned %s", symbol.c_str(), oo->toString().c_str());
            return px;
        }
        if (__builtin_expect((bidsz*asksz==0) || (askpx<bidpx+1e-10),0)) {
            // debug
            //logError("BBO invalid, cannot make decision, BBO: %d %s:%s %d",
            //        bidsz, PriceCString(bidpx), PriceCString(askpx), asksz);
            return px;
        }

        const int side = (qty>0)?1:-1;
        const double spd = askpx-bidpx;
        const auto ti (utils::SymbolMapReader::get().getTradableInfo(symbol));
        if (!ti) {
            logError("Cannot find tradable info for symbol %s", symbol.c_str());
        }
        const double tick_size { ti? ti->_tick_size:spd };
        const double px_agg = qty>0?askpx:bidpx;
        const double trade_agg_px = px_agg + (10*tick_size*side);
        if (!peg_passive) {
            logInfo("PegPassive False, aggressive to %s. OO: %s",
                    PriceCString(trade_agg_px), oo->toString().c_str());
            return trade_agg_px;
        }

        if ((px_agg-px)*side > spd+max_peg_tick_diff*tick_size) {
            logInfo("MaxPegSpreadDiff reached, aggressive to %s. BBO: %d %s:%s %d, OO: %s", 
                    PriceCString(trade_agg_px),
                    bidsz, PriceCString(bidpx), PriceCString(askpx), asksz,
                    oo->toString().c_str());
            return trade_agg_px;
        }

        double bp = (double)bidsz/(double)(asksz+bidsz);
        bp=side>0?bp:(1-bp);
        if (bp>bp_threshold) {
            // allowing flipping
            if ((px_agg - px)*side > (spd+2*tick_size-1e-10)) {
                logInfo("BP reached, aggressive to %s. BBO: %d %s:%s %d, OO: %s", 
                        PriceCString(trade_agg_px),
                        bidsz, PriceCString(bidpx), PriceCString(askpx), asksz,
                        oo->toString().c_str());
                return trade_agg_px;
            }
        }
        // stay
        return px;
    }
};
