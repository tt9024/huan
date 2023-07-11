#include "FloorManager.h"
#include "RiskMonitor.h"
#include "time_util.h"
#include "md_bar.h"
#include "instance_limiter.h"
#include <stdexcept>
#include <atomic>

namespace pm {
    FloorManager& FloorManager::get() {
        static FloorManager mgr("FloorManager");
        return mgr;
    };

    FloorManager::FloorManager(const std::string& name)
    : FloorCPR<FloorManager>(name) {
    };

    FloorManager::~FloorManager() {
    };

    void FloorManager::start_derived()
    {
        utils::OnlyMe::get().add_name(m_name);

        const utils::ConfigureReader config(utils::PLCC::getConfigPath());

        /*
        enable_position_server_ = config.get<bool>("EnablePositionServer");
        logInfo("PositionServer is %s.", enable_position_server_ ? "enabled" : "disabled");
        if (enable_position_server_) {

            using namespace com::massar::core::interface;

            const auto& host = config.get<std::string>("PositionServerHost");
            const auto& port = config.get<std::int32_t>("PositionServerPort");
            const auto& auth_token = config.get<std::string>("PositionServerAuthToken");

            // Static way - not ideal.
            PositionServer::init(host, port, auth_token);
            PositionServer& instance = PositionServer::instance();

            // Log to console + log
            std::cout << "PositionServer started ok: " << instance.getHost() <<
                      ":" << instance.getPort() << "\n";

            logInfo("PositionServer started: %s:%d",
                    instance.getHost().c_str(), instance.getPort());

            initializePositionServerInterface();
        }*/
    };

    void FloorManager::shutdown_derived() {
    };

    std::string FloorManager::toString_derived() const {
        return "" ; 
    };

    void FloorManager::addPositionSubscriptions_derived() {
        std::set<int> type_set;
        // no set only get
        // TODO - allow set, to check for the existence and uniqueness
        // of algo+symbol+type
        //type_set.insert((int)FloorBase::SetPositionReq);
        type_set.insert((int)FloorBase::GetPositionReq);
        subscribeMsgType(type_set);
    };

    void FloorManager::run_loop_derived() {
        static int check_utc = 0;
        static const int check_interval = 5;

        utils::TimeUtil::micro_sleep(IdleSleepMicro);
        auto cur_utc = utils::TimeUtil::cur_utc();
        if (__builtin_expect(cur_utc>check_utc, 0)) {
            check_utc = cur_utc + check_interval;
            if (!utils::OnlyMe::get().check()) {
                logError("other %s detected, exiting!", m_name.c_str());
                stop();
            }
        }

        scanOpenOrders();

        // Publish updates to PS
        /*
        if (isPositionServerEnabled())
            updatePositionServer();
        */
    }

    std::optional<double> FloorManager::getReferencePx(const std::string& symbol) const
    {
        static const std::string L1_BOOK_STR = "L1";

        // Get the reference price
        try {
            md::BookDepotDepot<1> book;
            if (md::LatestBook(symbol, L1_BOOK_STR, book))
                return {book.getMid()};

        } catch (const std::exception& e) {
            // nothing - we don't care about this for PS
        }

        return {};
    }

    void FloorManager::initializePositionServerInterface() const
    {
        // Initialize risk settings in Position Server
        using namespace com::massar::core::interface;

        auto& position_server = PositionServer::instance();
        const auto& risk_monitor = pm::risk::Monitor::get();
        const auto& risk_config = risk_monitor.get().config();

        // Add All Markets to config
        const auto& market_vec = risk_config.listMarket();
        for (const auto& market : market_vec) {
            auto max_engine_position = 0ul;
            auto engine_fat_finger = 0ul;
            auto eng_spread_limit = 0;
            auto max_price_ticks = 0;

            auto mitp = risk_config.m_eng_max_pos.find(market);
            if (mitp != risk_config.m_eng_max_pos.end()) {
                max_engine_position = mitp->second;
            }

            auto fitp = risk_config.m_eng_fat_finger.find(market);
            if (fitp != risk_config.m_eng_fat_finger.end()) {
                engine_fat_finger = fitp->second;
            }

            position_server.addEngineRiskConfig(market, max_engine_position,
                                                engine_fat_finger, eng_spread_limit, max_price_ticks);
        }

        // Add All Algos to config
        const auto& strategy_vec = risk_config.listAlgo();
        for (const auto& strategy : strategy_vec) {

            auto max_strat_pnl_drawdown = 0.0f;
            auto strategy_scale = 1.0f;
            auto paper_trading = false;

            auto mitp = risk_config.m_strat_pnl_drawdown.find(strategy);
            if (mitp != risk_config.m_strat_pnl_drawdown.end()) {
                max_strat_pnl_drawdown = mitp->second;
            }

            auto pitp = risk_config.m_strat_paper_trading.find(strategy);
            if (pitp != risk_config.m_strat_paper_trading.end()) {
                paper_trading = pitp->second;
            }

            auto sitp = risk_config.m_scale.find(strategy);
            if (sitp != risk_config.m_scale.end()) {
                strategy_scale = sitp->second;
            }

            position_server.addStrategyRiskConfig(strategy, paper_trading,
                                                  max_strat_pnl_drawdown, strategy_scale);
        }

        // Send risk config to PS
        auto response = position_server.sendRiskConfig();
        if (!response) {
            logError("Error sending risk config to position server!");
        } else {
            logInfo("Successfully sent risk config to position server!");
        }
    }

    void FloorManager::syncLocalPositionServerStates(const std::vector<com::massar::core::interface::AlgoStateMessage>& local_algo_states,
            const auto now) const
    {
        using namespace com::massar::core::interface;

        auto& algo_status = risk::Monitor::get().status();
        auto& position_server = PositionServer::instance();

        const auto notified = position_server.notifyState(local_algo_states);
        if (!notified) {
            logError("Unable to notify state from position server!");
            return;
        }

        // Check panic mode
        if (position_server.isPanicActivated()) {
            const bool set_state_set = algo_status.setPauseAll(true);
            if (!set_state_set) {
                logError("Unable to set pause all from panic mode actication!");
            } else {
                logInfo("Pause all from panic mode actication!");
            }

            // Clear out states from remote and local
            //position_server.notifyPanic();
            return;
        }

        // Enact on our updated remote states
        for (const auto& algo : local_algo_states) {

            // TODO: FIXME - HACK
            if (algo.algo == "TSC-7000-1")
                continue;

            // Check if we have PS up and the remote state is reported.
            const auto remote_working_state = position_server.getRemoteUpdatedCachedState(algo.algo, algo.symbol);
            if (remote_working_state == AlgoMessageState::NONE)
                continue;

            // We have a remote state command from PS (UI)
            if (remote_working_state != algo.state) {

                bool set_state_set{};
                bool new_paused_state{};

                switch (remote_working_state) {
                    case AlgoMessageState::PAUSED:

                        logInfo("Setting state on algo: %s/%s, %s (remote) =>%s (local)",
                                algo.algo.c_str(), algo.market.c_str(),
                                AlgoMessageState_Name(remote_working_state).c_str(),
                                AlgoMessageState_Name(algo.state).c_str());

                        new_paused_state = true;
                        set_state_set = algo_status.setPause(algo.algo, algo.market, new_paused_state);

                        break;
                    case AlgoMessageState::PAUSED_MARKET:
                        logInfo("Settings state on market: %s, %s (remote) =>%s (local)",
                                algo.market.c_str(),
                                AlgoMessageState_Name(remote_working_state).c_str(),
                                AlgoMessageState_Name(algo.state).c_str());

                        new_paused_state = true;
                        set_state_set = algo_status.setPause(algo.algo, "ALL", new_paused_state);

                        break;
                    case AlgoMessageState::PAUSED_STRATEGY:

                        logInfo("Settings state on strategy: %s, %s (remote) =>%s (local)",
                                algo.algo.c_str(),
                                AlgoMessageState_Name(remote_working_state).c_str(),
                                AlgoMessageState_Name(algo.state).c_str());

                        new_paused_state = true;
                        set_state_set = algo_status.setPause("ALL", algo.market, new_paused_state);

                        break;
                    case AlgoMessageState::TERMINATED:
                        logInfo("Settings terminated  on strategy: %s, %s (remote) =>%s (local)",
                                algo.algo.c_str(),
                                AlgoMessageState_Name(remote_working_state).c_str(),
                                AlgoMessageState_Name(algo.state).c_str());

                        new_paused_state = true;
                        set_state_set = algo_status.setPause("ALL", "ALL", new_paused_state);
                        break;
                    default:
                        new_paused_state = false;
                        // No unset for pause from PS (2023/03/08)
                        break;
                }

                // Verify the state was actually set
                if (!set_state_set ||
                    algo_status.getPause(algo.algo, algo.market) != new_paused_state) {

                    logError("Unable to update working state on algo: %s/%s, %s (remote) =>%s (local)",
                            algo.algo.c_str(), algo.market.c_str(),
                            AlgoMessageState_Name(remote_working_state).c_str(),
                            AlgoMessageState_Name(algo.state).c_str());

                } else {
                    // We've correctly sync'd remote with local states
                    const auto removed = position_server.clearRemoteUpdatedCachedState(algo.algo, algo.symbol);
                    if (!removed) {
                        logError("Unable to clear working state on algo/market: %s/%s, %s (remote) =>%s (local)",
                                 algo.algo.c_str(), algo.market.c_str(),
                                 AlgoMessageState_Name(remote_working_state).c_str(),
                                 AlgoMessageState_Name(algo.state).c_str());
                    } else {
                        logInfo("Updated working state of algo/market: %s/%s, %s (remote) =>%s (local)",
                                 algo.algo.c_str(), algo.market.c_str(),
                                 AlgoMessageState_Name(remote_working_state).c_str(),
                                 AlgoMessageState_Name(algo.state).c_str());
                    }

                    // Send back updated state
                    AlgoStateMessage updated_state = algo;
                    updated_state.state = new_paused_state ? AlgoMessageState::PAUSED : AlgoMessageState::WORKING;
                    const auto remote_updated = position_server.notifyState(std::vector{updated_state});
                    if (!remote_updated) {
                        logError("Unable to update remote state on algo/market: %s/%s, %s (remote) =>%s (local)",
                                 algo.algo.c_str(), updated_state.market.c_str(),
                                 AlgoMessageState_Name(remote_working_state).c_str(),
                                 AlgoMessageState_Name(updated_state.state).c_str());
                    } else {
                        logInfo("Updated remote state of algo/market: %s/%s, %s (remote) =>%s (local)",
                                algo.algo.c_str(), updated_state.market.c_str(),
                                AlgoMessageState_Name(remote_working_state).c_str(),
                                AlgoMessageState_Name(updated_state.state).c_str());
                    }
                }
            } else {
                logInfo("Working states on algo/market are identical: %s/%s, %s (remote) =>%s (local)",
                         algo.algo.c_str(), algo.market.c_str(),
                         AlgoMessageState_Name(remote_working_state).c_str(),
                         AlgoMessageState_Name(algo.state).c_str());
            }
        }
    }

    void FloorManager::updatePositionServer(bool snapshot)
    {
        using namespace com::massar::core::interface;

        PositionServerDetails position_details{};
        PositionServer& position_server = PositionServer::instance();

        // Initial check for initialization
        if (!position_server.isInitialized()) {
            auto risk_ok = position_server.sendRiskConfig();
            if (!risk_ok) {
                logError("Unable to update Position Server Risk!");
                // Do not attempt to send positions if risk is not sent first.
                return;
            }
        }

        auto& algo_status = risk::Monitor::get().status();
        const auto& symbol_map = utils::SymbolMapReader::get();
        const auto& positions = getPM().getPositionServerDetails();

        std::vector<AlgoStateMessage> local_algo_states{};

        const auto now = std::chrono::high_resolution_clock::now();

        // Iterate our positions and update PS UI
        for (const auto& position : positions) {

            bool ps_ok{};
            double reference_px{};

            const auto &market = symbol_map.getTradableMkt(position.symbol);

            // TODO: FIXME - HACK
            bool current_paused_state{};
            if (position.algo != "TSC-7000-1") {
                current_paused_state = algo_status.getPause(position.algo, market);
            } else {
                // ONLY TEMP
                current_paused_state = false;
            }

            local_algo_states.emplace_back(position.algo, position.symbol, market,
                                           current_paused_state ? AlgoMessageState::PAUSED : AlgoMessageState::WORKING);

            // Get the ref price; nb: this can throw
            auto reference_price_opt = getReferencePx(position.symbol);
            if (reference_price_opt.has_value())
                reference_px = *reference_price_opt;

            if (snapshot) {
                // nb: This is a little silly now; Make a struct and pass that instead
                ps_ok = position_server.addPosition(0, PositionAction::ADD,
                                                    (position.quantity > 0 ? PositionSide::BUY : PositionSide::SELL),
                                                    position.algo, position.symbol, std::abs(position.quantity),
                                                    position.value, reference_px, 0,
                                                    position.mark_to_market, position.profit_and_loss,
                                                    position.total_mark_to_market, position.total_profit_and_loss,
                                                    position.has_aggregate, current_paused_state);
            } else {
                // Nb: this is guarded against sending DUPs
                ps_ok = position_server.updatePosition(0, PositionAction::UPDATE,
                                                       (position.quantity > 0 ? PositionSide::BUY : PositionSide::SELL),
                                                       position.algo, position.symbol, std::abs(position.quantity),
                                                       position.value, reference_px, 0,
                                                       position.mark_to_market, position.profit_and_loss,
                                                       position.total_mark_to_market, position.total_profit_and_loss,
                                                       position.has_aggregate, current_paused_state);
            }

            if (!ps_ok) {
                logError("Unable to add/save the position to position server: (snapshot=%s) %s/%s",
                        (snapshot ? "true" : "false"),
                        position.algo.c_str(), position.symbol.c_str());
            }
        }

        // We might not have updates from MTS so send a ping to notify UI we're alive
        const auto duration =
                std::chrono::duration_cast<std::chrono::seconds>(now - position_server.lastUpdateTime()).count();
        if (duration >= 5) {
            if (!position_server.sendHeartbeat()) {
                logError("Unable to heartbeat the position server!");
            }
        }

        // Finally sync the remote states with local
        const auto state_duration =
                std::chrono::duration_cast<std::chrono::seconds>(now - position_server.lastStateUpdateTime()).count();
        if (state_duration >= 1) {
            syncLocalPositionServerStates(local_algo_states, now);
        }
    }

    void FloorManager::handleExecutionReport_derived(const pm::ExecutionReport& er) {
        // extra handling after receiving an er
    };

    bool FloorManager::handleUserReq_derived(const MsgType& msg, std::string& respstr) {
        const char* cmd = msg.buf;
        logDebug("%s got user command: %s", m_name.c_str(), msg.buf);
        const std::string helpstr(
                    "Command Line Interface\n"
                    "P algo_name, symbol\n\tlist positions (and open orders) of the specified algo and symbol\n\thave to specify both algo and symbol, leave empty to include all entries\n"
                    "B|S algo_name, symbol, qty, price\n\tenter buy or sell with limit order with given qty/px (qty is positive)\n\tprice string [b|s][+|-][t|s]count, where \n\t\t[b|s]: reference to b (best bid price) or a (best ask price)\n\t\t[+|-]: specifies a delta of plus or minus\n\t\t[t|s]: specifies unit of delta, t (tick) or s (current spread)\n\t\tcount: number of units in delta\n"
                    "C [ClOrdId]\n\tcancel an order from tag 11, the client order id\n\tIf no ClOrdId is provided, cancel all open orders\n"
                    "R ClOrdId, qty, px\n\tqty positive, same sign with original order\n"
                    "X algo_name, symbol, qty [,px_str|twap_str]\n\tset target position by trading if necessary, with an optional\n\tlimit price string (see B|S order). If no price is specified,\n\ttrade aggressively using limit price of the other side\n\ttwap_str can be specified in place of a px_str, in format of Tn[s|m|h]\n\twhere n is a number, 's','m' or 'h' specifies unit of time.\n"
                    "A algo_name, symbol, target_position, target_vap\n\tset the position and vap to the given targets with synthetic fills\n"
                    "E \n\tinitiate the reconcile process, if good, persist currrent position to EoD file\n"
                    "D \n\tdump the state of Floor Manager\n"
                    "F \n\tdump the state of Floor Trader\n"
                    "K \n\tstop the floor message processing and done\n"
                    "M [symbol]\n\tget the snap and bars (with bar_sec for bar_cnt), empty symbol matches all symbols\n"
                    "Z algo, market_list [,ON|OFF]\n\tget or set trading status paused for algo and list of markets, a ':' delimitered string, to be ON or OFF if given, otherwise gets paused status\n\tempty algo or symbol matches all\n\tmarket_list doesn't specify contract, i.e. WTI:Brent\n\t"
                    "Y [opeator_id] \n\tget or set operator id"
                    // Below commands have message type
                    // "FloorBase::AlgoUserCommand"
                    // They are handled by AlgoThread
                    "@L \n\tlist loaded strategies\n"
                    "@strat_name S\n\tstart strat_name\n"
                    "@strat_name E\n\tstop strat_name, specify '*' for all strategies\n"
                    "@strat_name D\n\tdump pmarameters and state of strat_name\n"
                    "@state_name R config_file\n\tstop, reload with config_file and start\n"
                    "H\n\tlist of commands supported\n");

        // control messages
        // handled by CPR: 'K', 'A', 'Z', 'L'
        // User cmd of 'X' is converted to "PositionReq' by flr, so handled at
        // handlePositionReq_derived
        //
        switch (cmd[0]) {
            case 'H' : 
            {
                respstr = helpstr;
                break;
            };
            case 'P':
            {
                // get position or open order
                auto tk = utils::CSVUtil::read_line(cmd+1);
                if (tk.size()!=2) {
                    respstr = std::string("Failed to parse Algo or Symbol name: ")+ std::string(cmd)+ "\n"+ helpstr;
                } else {
                    respstr = getPM().toString(&tk[0], &tk[1], true);
                }
                break;
            }
            case 'D' :
            {
                respstr = toString();
                break;
            }
            case 'E' :
            {
                if (m_eod_pending) {
                    respstr = "Already in EoD\n";
                } else {
                    PositionManager pmr("reconcile");
                    m_eod_pending = true;
                    std::string errstr;
                    if (!requestReplay(pmr.getLoadUtc(), &errstr)) {
                        m_eod_pending = false;
                        respstr = "problem requesting replay: " + errstr;
                    }
                }
                break;
            }
            case 'B':
            case 'S':
            {
                // send a req and return (qty is positive)
                //"B|S algo_name, symbol, qty, price_str 
                logInfo("%s got user command: %s", m_name.c_str(), msg.buf);
                const char* bsstr = cmd;
                auto errstr = sendOrderByString(bsstr);
                if (errstr.size()>0) {
                    respstr = errstr;
                }
                break;
            }
            case 'C':
            case 'R':
            {
                // send a cancel/replace (qty is positive)
                // C|R ClOrdId [, qty, px] - qty positive, same sign
                logInfo("%s got user command: %s", m_name.c_str(), msg.buf);
                const char* cmdstr = cmd;
                auto errstr = sendCancelReplaceByString(cmdstr);
                if (errstr.size()>0) {
                    respstr = errstr;
                }
                break;
            }
            case 'M' :
            {
                //M symbol
                respstr = handleUserReqMD(cmd);
                break;
            }
            // handled by FloorTrader, no need to response here
            case 'F' :
            {
                return false;
            }
            default :
                respstr = "not supported (yet?)";
        }
        return true;
    }

    bool FloorManager::handlePositionReq_derived(const MsgType& msg, MsgType& msg_out) {
        // handles both GetPositionReq and SetPositionReq
        if (msg.type == FloorBase::GetPositionReq) {
            // expect a FloorBase::PositionRequest
            // the algo_name is allowed to be "ALL", but symbol has to be specified
            // returns another PositionRequest struct populated with
            // two int64_t as (aggregated) qty_done and qty_open in m_msgout.buf

            m_msgout.ref = msg.ref;
            m_msgout.type = FloorBase::GetPositionResp;
            m_msgout.copyData(msg.buf, msg.data_size);
            FloorBase::PositionRequest* prp = (FloorBase::PositionRequest*)m_msgout.buf;
            prp->qty_done = getPM().getPosition(prp->algo, prp->symbol, nullptr, nullptr, &(prp->qty_open));
            return true;
        }
        return false;
    }

    std::string FloorManager::handleUserReqMD(const char* cmd) {
        //M symbol
        std::string respstr;
        const auto& tk (utils::CSVUtil::read_line(cmd+1));
        std::vector<std::string> sym;
        if ((tk.size() == 0 ) || (tk[0] == "")) {
            sym = utils::SymbolMapReader::get().getPrimarySubscriptions(1);
        } else {
            sym.push_back(tk[0]);
        }
        // output the results
        char buf[1024*64];
        size_t bcnt = 0;
        bcnt = snprintf(buf, sizeof(buf), "%-16s  %-13s  %s\n"
               "---------------------------------------------------\n", 
               "symbol", "quote (mid)", "updated (sec ago)");
        std::map<std::string, int> stale_symbols;
        const int stale_secs = 60;
        for (const auto& s: sym) {
            md::BookDepotLevel<1> book;
            bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "%-16s  ", s.c_str());
            bool ret = false;
            try {
                ret = md::LatestBook(s, "L1", book);
            } catch (const std::exception& e) {
            }
            if (ret) {
                double mid = book.getMid();
                int secs_ago = (int)((utils::TimeUtil::cur_micro() - book.update_ts_micro)/1000000ULL);
                bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "%-13s  %d",
                        PriceCString(mid), secs_ago);
                if (secs_ago > stale_secs) {
                    bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, " (stale?) ");
                    stale_symbols[s] = secs_ago;
                }
            } else {
                bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "%-13s  %s", "N/A", "N/A");
                stale_symbols[s] = -1;
            }
            bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "\n");
        }
        bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "===\nSummary: ");
        if (stale_symbols.size() == 0) {
            bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "All Good!");
        } else {
            int sz = (int)stale_symbols.size();
            bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "Found %d %s (updated %d seconds ago):\n", sz, sz>1?"warnings":"warning", stale_secs);
            for (const auto& ss : stale_symbols) {
                bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, "\t%-16s (%s)\n",
                        ss.first.c_str(), 
                        (ss.second > 0? std::to_string(ss.second).c_str():"N/A"));
            }
        }
        buf[bcnt-1] = 0;
        respstr = std::string(buf);
        return respstr;
    }

    // open order scan, only scan PASSIVE or MARKET orders
    bool FloorManager::shouldScan(const std::shared_ptr<const OpenOrder>& oo, bool& peg_passive) const {
        peg_passive = false;
        // do not scan oo if qty is zero
        int64_t qty = oo->m_open_qty;
        if (!qty) {
            return false;
        }

        // do not scan manual order
        if (risk::Monitor::get().config().isManualStrategy(oo->m_idp->get_algo())) {
            return false;
        }

        // only scan PI type of
        // PASSIVE or MARKET
        // TWAP/VWAP/TRADER_WO orders are not scanned 
        // here by FM
        const auto& clOrdId (oo->m_clOrdId);
        auto iter (m_orderMap.find(clOrdId));
        if (iter != m_orderMap.end()) {
            const auto& pi = iter->second;
            const int64_t cur_micro = utils::TimeUtil::cur_micro();
            if (cur_micro - (int64_t)oo->m_open_micro > 100000LL) {
                if (pi->type == PositionInstruction::PASSIVE) {
                    peg_passive = true;
                    return true;
                }
                if (pi->type == PositionInstruction::MARKET) {
                    peg_passive = false;
                    return true;
                }
            }
        }
        return false;
    }

    bool FloorManager::scanOpenOrders() {
        static uint64_t last_scan_micro = 0;
        auto cur_micro = utils::TimeUtil::cur_micro();
        if ((int64_t) (cur_micro - last_scan_micro) < (int64_t) ScanIntervalMicro) {
            return false;
        }
        last_scan_micro = cur_micro;

        // trade out open orders with less 
        bool ret = false;
        const auto& vec(m_pm.listOO());
        for (const auto& oo: vec) {
            //logDebug("scanning oo: %s", oo->toString().c_str());
            bool peg_passive = false;
            if (!shouldScan(oo, peg_passive)) {
                continue;
            }
            double px_now = getPegPxBP(oo, peg_passive, MaxPegTickDiff, 0.5);
            double px_diff = oo->m_open_px - px_now; 
            if (std::abs(px_diff) > 1e-10) {
                ret = true;
                logDebug("Price moved away for %s, new px: %s, px-diff: %s, trading in", oo->toString().c_str(), PriceCString(px_now), PriceCString(px_diff));

                // additional check on the replacement
                logDebug("Price moved away detail: oo_price: %.8f(%s), px_now: %.8f(%s), px_diff: %.8f(%s)",
                        oo->m_open_px, PriceCString(oo->m_open_px), px_now, PriceCString(px_now), px_diff, PriceCString(px_diff));

                char tstr[256];
                snprintf(tstr, sizeof(tstr), "R %s,,%s", oo->m_clOrdId, PriceCString(px_now));
                std::string errstr = sendCancelReplaceByString(tstr);
                if (errstr != "") {
                    logError("failed to send replace order %s error = %s", 
                            tstr, errstr.c_str());
                    continue;
                }
                // In case the cancel is rejected, the fill 
                // should be in the way to be applied without oo
                // and orderMap entry, which is fine
                m_pm.deleteOO(oo->m_clOrdId);
                if (!peg_passive) {
                    // avoid burst and wait for rtt
                    last_scan_micro = last_scan_micro - ScanIntervalMicro + PegAggIntervalMilli*1000;
                };
                break;
            }
        }
        return ret;
    }
}
