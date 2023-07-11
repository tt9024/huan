#pragma once

#include <unordered_map>
#include <cstdlib>
#include <vector>
#include <tuple>

#include "flr.h" //includes FloorBase.h and then floor.h
#include "rate_limiter.h"
#include "rate_estimator.h"
#include "thread_utils.h"

#define DEFAULT_RISK_JSON "/home/mts/run/config/risk/risk.json"

namespace pm {
namespace risk {

// forward declaration of Monitor - the main class
class Monitor;

struct Config {
    // global variables
    std::string m_config_file, m_manual_strat;
    bool m_skip_replay; // whether to skip report during  PM's startup replay

    // System
    // per-strategy scale, i.e. {TSC-7000-380: 1.5}
    std::unordered_map<std::string, double> m_scale;

    // per-market price banding, key ie WTI, val ie 0.2(price)
    // TODO - use price_ticks for now, should be tighter
    std::unordered_map<std::string, double> m_price_band;

    // output a visualization string for each actively trading
    // strategy/symbol
    std::string m_volume_spread_file;
    std::string m_ppp_file;  // full path to ppp file
    std::string m_status_file;

    // Engine
    // aggregated at market level (all N), key: ie WTI
    std::unordered_map<std::string, long long> m_eng_max_pos;

    // count limit of <orders, seconds>
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> m_eng_count;

    // vector of 5m bar rate limit of vector<<rate, seconds>>
    std::unordered_map<std::string, std::vector<std::vector<std::pair<double,int>>>> m_eng_rate_limit;
    std::unordered_map<std::string, std::vector<double>> m_eng_spread_limit;
    std::unordered_map<std::string, long long> m_eng_fat_finger;

    // TODO - price checking
    std::unordered_map<std::string, double> m_price_ticks;

    // Strategy
    // aggregated at per-strtegy per-mkt, pos/pnl/count/rate scaled
    std::unordered_map<std::string, double> m_strat_pnl_drawdown;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> m_strat_mkt_pnl_drawdown;
    std::unordered_map<std::string, bool>   m_strat_paper_trading;

    std::unordered_map<std::string, std::unordered_map<std::string, long long>> m_strat_max_pos;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<int, int>>>> m_strat_count;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<double, int>>>> m_strat_rate_limit; // strat - mkt - vector< <turnover_rate, seconds> >
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<int, int>>>> m_strat_flip;

    // Member Functions
    explicit Config(const std::string& risk_file);
    std::string toStringConfig() const;
    // return the mkt of symbol if algo+symbol found in map
    std::string checkAlgoSymbol(const std::string& algo, const std::string& tradable_symbol, bool verbose=true) const;

    // convenience const functions
    bool isPaperTrading(const std::string& algo) const;
    std::string ManualStrategyName() const { return m_manual_strat; };
    bool isManualStrategy(const std::string& algo) const { return algo == m_manual_strat ; };

    const std::vector<std::string>& listAlgo() const;
    const std::vector<std::string>& listMarket() const;
    const std::vector<std::string>& listSymbol() const;

    bool algoExists(const std::string& algo) const;
    bool marketExists(const std::string& mtk) const;
    bool symbolExists(const std::string& symbol) const;

    // a constructor mainly for testing with given scale
    // see risk_test.cpp
    Config(const std::string& risk_file,
        std::unordered_map<std::string, double> input_scale);
private:
    void load (const std::string& risk_file);
    void read_strat_scale(const utils::ConfigureReader& reader);
    void read_system(const utils::ConfigureReader& reader);

    using VolumeSpreadType = std::unordered_map<std::string, 
          std::vector<
                std::pair< // first: volume, second: spread
                   std::vector<
                       std::pair<double, double>>, // the three avg/std for 5m/15m/60m
                   std::pair<double, double>>>>;

    const VolumeSpreadType getVolumeSpread() const;
    void read_engine(const utils::ConfigureReader& reader);
    void read_engine_mkt(const std::string& mkt, const utils::ConfigureReader& reader, const VolumeSpreadType& volume_spread);
    void read_strategy(const utils::ConfigureReader& reader);
    void read_strategy_mkt(const std::string& strat, const std::string& mkt, const utils::ConfigureReader& reader);
};

class Status {
public:

    // per-strategy-mkt trading status of
    // trading status (pause or not)
    // market status (pause or not)
    //
    // TODO - operator ID, and other possible status to be implemented
    //
    Status(std::shared_ptr<Config> cfg);

    // pause related functions on market (not tradable), empty string matches all
    bool setPause(const std::string& algo, const std::string& market, bool if_pause);
    bool setPauseSymbol(const std::string& market, bool if_pause);  // this pauses market for all algos
    bool setPauseAlgo(const std::string& algo, bool if_pause);  // this pauses algo for all symbols
    bool setPauseAll(bool if_pause);  // this pauses all
    bool getPause(const std::string& algo, const std::string& market) const;
    std::string queryPause(const std::string& algo, const std::string& market) const;

    // this sends message on the floor to notify a new pause status being initiated by
    // this RM.  All RMs on the floor receives it, including FM, who also persist this
    // message to the "pause file" as persistence.
    //
    // This uses the same formt of user's pause request as:
    // Z algo, market ,ON|OFF
    // set pause for algo, market to be ON or OFF, empty algo or market matches all
    //
    // NOTE - this should trigger local set_pause() via the message being received, 
    // but it doesn't hurt to explicit set just in case channel is not responsive
    void notify_pause(const std::string& algo, const std::string& market, bool if_pause = true);

    // used by FM to write to the pause file
    // just persist the user command into the load format
    template<typename FMType>
    void persist_pause(const FMType& fm, const char* cmd);

    template<typename FMType>
    void set_operator_id(const FMType& fm, const std::string& operator_id);

    //bool getOperatorID(const std::string& strategy);

    // periodic tasks such as sync pause status and operator id
    void statusLoop() { load_pause(); }; // nothing to be done periodically

    // read status from a persist file
    void load_pause();

    // a human readable string in format of
    //     strat, mkt_1:mkt_2:...mkt_n, ON
    // Otherwise, in case all is alive
    //     ALL, ALL, OFF
    const std::string toStringStatus() const;
    const std::string get_operator_id() const { return m_tag50; };

    // data members
    std::string m_name;
    const std::string m_persist_file;

private:
    enum TradingStatus {
        TradingStatus_INVALID = 0,
        TradingStatus_OK = 1,
        TradingStatus_PAUSED = 2,
        TradingStatus_KILLED = 3, // stopped and clear position
        TradingStatus_PAPER = 4,  // stopped and clear position
        TradingStatus_TOTAL = 5  // stopped and clear position
    };
    struct StatusInfo {
        uint8_t status;  // TradingStatus
        uint8_t reserved1;
        uint16_t reserved2;
        uint32_t reserved3;
        StatusInfo(): status((uint8_t)TradingStatus_INVALID), reserved1(0), reserved2(0), reserved3(0) {};
        bool is_paused() const { return status != (uint8_t)TradingStatus_OK; };
        void set_pause(bool if_pause) {
            status = (uint8_t)(if_pause? TradingStatus_PAUSED:TradingStatus_OK);
        };
    };

    pm::FloorClientUser m_fcu; // used for send out pause command upon failed checks
    std::unordered_map<std::string, std::unordered_map<std::string, StatusInfo>> m_status;
    std::shared_ptr<Config> m_cfg;
    std::string m_tag50;
};

class State {
public:
    explicit State(std::shared_ptr<Config> cfg);
    /*
    // TODO - to be implemented if there were a need
    // to load previous risk state. 
    // Both RateLimitEns and RateEstimator are ready to 
    // be persisted and retrieved.
    //
    State(const std::string& persist_file);
    static std::shared_ptr<State> load(const std::string& persist_file);
    */

    // checks the strategy and eng level limiters, if all good, then update all
    // otherwise, log error and return false
    bool checkOrder (const std::string& algo,
                     const std::string& mkt,
                     int64_t ord_qty,
                     time_t cur_utc);

    // do updateOnly for all rate limiters, since the fill already happened,
    // and check the results, log error if non-zero returned
    bool reportFill (const std::string& algo,
                     const std::string& mkt,
                     int64_t fill_qty,
                     time_t cur_utc);

    // do updateOnly for all limiters, and check the results, log error if non-zero returned
    bool reportNew (const std::string& algo,
                     const std::string& mkt,
                     time_t cur_utc);

    // remove order count from both order and report side
    void reportCancel (const std::string& algo,
                     const std::string& mkt);


    // check the max position, the pnl and flip
    // allow for risk reducing trades to go throw
    // if_from_report is true if it is called on a fill
    // from the execution report, in which case the
    // position state (for flip) will be updated regardless
    // of risk check outcome.
    //
    // return if the max_pos and pnl drawdown is good
    // Following quantities are queried from PM:
    //     algo_qty: the aggreated position of algo+mkt
    //     eng_qty:  the aggreated position of mkt of all algos in engine
    //     mkt_pnl:  the mark-to-market pnl of algo+mkt (all _N aggreated)
    //     algo_pnl: the mark-to-market pnl of algo (all mkts aggregated)
    template<typename PositionManager>
    bool checkPosition(const std::string& algo,
                       const std::string& mkt,
                       int64_t qty,
                       const PositionManager& pm,
                       bool is_from_report,
                       time_t cur_utc);

    // run periodical tasks, such as 
    // update rate estimation, output logs for gui, etc
    void stateLoop() { } ;

    // output a log string to be visualized by GUI
    std::string toString() const { return "RiskMonitor State: " + m_cfg->toStringConfig(); };
    int getBarIdx(time_t cur_utc) const;

    std::string m_name;
private:
    // for each symbol, define two pairs of limiters,
    // first updated from strategy, second updated from execution report
    using LimitMap = std::unordered_map<
        std::string, // per-mkt
            std::pair<std::shared_ptr<utils::RateLimiterEns>,   // order count limits
                      std::shared_ptr<utils::RateEstimator>     // order rate/turn over limits
                     > [2] // first: ord_upd, second: er_upd
        >;

    std::shared_ptr<Config> m_cfg;
    LimitMap m_eng; // engine level limiters
    std::unordered_map<std::string, LimitMap> m_strat; // strate level limiters
    std::unordered_map<std::string, LimitMap> m_flip;  // strate level position flips limiters
    std::unordered_map<std::string, std::unordered_map<std::string, long long>> m_prev_direction;

    void init();
    bool checkTradeRate(time_t cur_utc, const std::string& mtk, int ord_cnt, std::shared_ptr<utils::RateEstimator>& re);
    bool checkFlip(const std::string& algo, const std::string& mkt, long long new_pos_qty, bool is_fill_report, time_t cur_utc);
};

class Monitor {
public:
    // singleton main instance, using default risk config file
    static Monitor& get();

    // mainly used for test purpose
    static std::shared_ptr<Monitor> get(const std::string& cfg_fn);


    // checks used by trader before sending order to FeedHandler
    template<typename PositionManager>
    bool checkNewOrder(const std::string& algo,
                       const std::string& tradable_symbol,
                       int64_t ord_qty,
                       const PositionManager& pm,
                       time_t cur_utc=0);

    template<typename PositionManager>
    bool checkReplace (const std::string& algo,
                       const std::string& tradable_symbol,
                       int64_t qty_delta,
                       const PositionManager& pm,
                       time_t cur_utc=0);

    // checks used by a FeedHandler where Position Manager cannot be
    // obtained
    bool checkNewOrder(const std::string& algo,
                       const std::string& tradable_symbol,
                       int64_t ord_qty,
                       time_t cur_utc=0);

    bool checkReplace (const std::string& algo,
                       const std::string& tradable_symbol,
                       int64_t qty_delta);

    // functions for execution report update
    template<typename ExecutionReport, typename PositionManager>
    bool updateER(const ExecutionReport& er, 
                  const PositionManager& pm,
                  bool do_notify_pause=false);

    template<typename ExecutionReport>
    bool updateER(const ExecutionReport& er, bool do_notify_pause=false);


    // Used to allow the status manipulations, 
    // such as set/get pause, persit and notify pauses, etc
    Status& status() { return m_status ; };

    // Used to get config such as paper trading,
    // manual strategy name, etc
    const Config& config() { return *m_cfg; };

    // check if the limit price against the 
    // m_price_ticks[mkt] and m_eng_spread_limit[mkt][bar_idx]
    bool checkLimitPrice(const std::string& symbol, double price, time_t cur_utc) const;

    // run periodical tasks, such as 
    // sync with status, output ppp for gui, etc
    void runLoop() { };

    // whether to skip ER during PM's startup replay
    bool skip_replay() const { return m_cfg->m_skip_replay; };

    // const convenience function
    long long maxPosition(const std::string& algo, const std::string& tradable_symbol) const;
    void set_instance_name(const std::string& name);

    std::string m_name;
private:
    std::shared_ptr<Config> m_cfg;
    State m_state;
    Status m_status;

    explicit Monitor(const std::string& risk_cfg = "");
    Monitor(const Monitor& rm) = delete;
    void operator = (const Monitor& rm) = delete;

    bool checkNewOrder_mkt(const std::string& algo,
                       const std::string& mkt,
                       int64_t ord_qty,
                       time_t cur_utc);
};

/*
 * Implemenations for RiskMonitor
 */

/*********
 * Monitor
 ********/
inline
Monitor& Monitor::get() {
    static Monitor rm;
    return rm;
}

// this is not thread safe, mainly used for test purpose
// 
inline
std::shared_ptr<Monitor> Monitor::get(const std::string& cfg_name) {
    static std::atomic<bool> s_lock (false);
    static std::map<std::string, std::shared_ptr<Monitor>> s_monitor_map;

    utils::SpinLock spin_lock(s_lock);
    auto iter = s_monitor_map.find(cfg_name);
    if (__builtin_expect(iter != s_monitor_map.end(),1)) {
        return iter->second;
    }
    std::shared_ptr<Monitor> mon (new Monitor(cfg_name));
    s_monitor_map.emplace(std::make_pair(cfg_name, mon));
    return mon;
}

// checks used by trader before sending order to FeedHandler
template<typename PositionManager>
bool Monitor::checkNewOrder(const std::string& algo,
                            const std::string& tradable_symbol,
                            int64_t ord_qty,
                            const PositionManager& pm,
                            time_t cur_utc) {
    if (__builtin_expect(m_cfg->isManualStrategy(algo), 0)) {
        return true;
    }
    const auto& mkt (m_cfg->checkAlgoSymbol(algo, tradable_symbol));
    if (__builtin_expect(mkt == "",0)) {
        // algo or symbol not found
        return false;
    }
    if (__builtin_expect(!checkNewOrder_mkt(algo, mkt, ord_qty, cur_utc),0)) {
        return false;
    }
    if (__builtin_expect(!m_state.checkPosition(algo, mkt, ord_qty, pm, false, cur_utc), 0)) {
        return false;
    }
    return true;
}

template<typename PositionManager>
bool Monitor::checkReplace (const std::string& algo,
                            const std::string& tradable_symbol,
                            int64_t qty_delta,
                            const PositionManager& pm, 
                            time_t cur_utc) {

    if (__builtin_expect(m_cfg->isManualStrategy(algo), 0)) {
        return true;
    }
    const auto& mkt (m_cfg->checkAlgoSymbol(algo, tradable_symbol));
    if (__builtin_expect(mkt == "",0)) {
        // algo or symbol not found
        return false;
    }
    if (__builtin_expect(m_status.getPause(algo, mkt), 0)) {
        // in pause
        return false;
    }
    // skip the order count limit checks
    if (__builtin_expect(!m_state.checkPosition(algo, mkt, qty_delta, pm, false, cur_utc), 0)) {
        return false;
    }
    return true;
}

// checks used by a FeedHandler where Position Manager cannot be
// obtained
inline
bool Monitor::checkNewOrder(const std::string& algo,
                            const std::string& tradable_symbol,
                            int64_t ord_qty,
                            time_t cur_utc) {
    if (__builtin_expect(m_cfg->isManualStrategy(algo), 0)) {
        return true;
    }
    const auto& mkt (m_cfg->checkAlgoSymbol(algo, tradable_symbol));
    if (__builtin_expect(mkt == "",0)) {
        // algo or symbol not found
        return false;
    }
    return checkNewOrder_mkt(algo, mkt, ord_qty, cur_utc);
}

inline
bool Monitor::checkReplace (const std::string& algo,
                            const std::string& tradable_symbol,
                            int64_t qty_delta) {
    if (__builtin_expect(m_cfg->isManualStrategy(algo), 0)) {
        return true;
    }
    const auto& mkt (m_cfg->checkAlgoSymbol(algo, tradable_symbol));
    if (__builtin_expect(mkt == "",0)) {
        // algo or symbol not found
        return false;
    }
    if (__builtin_expect(m_status.getPause(algo, mkt), 0)) {
        // in pause
        return false;
    }
    return true;
}

// functions for execution report update
template<typename ExecutionReport>
bool Monitor::updateER(const ExecutionReport& er, bool do_notify_pause) {
    if (__builtin_expect((er.m_symbol[0] == 0) || (er.m_algo[0] == 0), 0)) {
        return true;
    }
    const auto&tradable_symbol (er.m_symbol);
    const std::string algo (er.m_algo);
    if (__builtin_expect(m_cfg->isManualStrategy(algo), 0)) {
        return true;
    }
    const auto& mkt (m_cfg->checkAlgoSymbol(algo, tradable_symbol, false));
    if (__builtin_expect(mkt == "",0)) {
        // algo or symbol not found
        return true;
    }
    const auto cur_utc = er.m_recv_micro/1000000ULL;
    const auto qty = er.m_qty;

    if (er.isNew()) {
        if (__builtin_expect(m_state.reportNew(algo, mkt, cur_utc), 1)) {
            return true;
        }
        logError("RiskMonitor(%s) detected count violation on NEW: \n%s",
                m_name.c_str(), er.toString().c_str());
        // fall down for pause notification
    } else if (er.isCancel()) {
        //update the m_rateLimiter_order
        m_state.reportCancel(algo, mkt);
        return true;
    } else if (er.isFill()) {
        if (__builtin_expect(m_state.reportFill(algo, mkt, qty, cur_utc),1)) {
            return true;
        }
        logError("RiskMonitor(%s) detected count/rate vilation on Fill: \n%s",
                m_name.c_str(), er.toString().c_str());
    } else { 
        // rejectsions, etc, not risk related
        return true;
    }
    if (do_notify_pause) {
        logError("RiskMonitor(%s) notifying trading pause on %s:%s!", 
            m_name.c_str(), algo.c_str(), mkt.c_str());
        m_status.notify_pause(algo, mkt, true);
    } else {
        m_status.setPause(algo, mkt, true);
    }
    return false;
}

template<typename ExecutionReport, typename PositionManager>
bool Monitor::updateER(const ExecutionReport& er, const PositionManager& pm, bool do_notify_pause) {
    bool good = updateER(er, do_notify_pause);
    if (er.isFill()) {
        // only check if this fill not yet applied
        if (__builtin_expect(pm.haveThisFill(er),0)) {
            logError("RiskMonitor(%s) Fill already applied by PositionManager when checking its risk! Note, it is assumed that PM checkes risk before update with the execution report... this maybe a serious error! skipped for now... \n%s", m_name.c_str(), er.toString().c_str());
            return good;
        }

        const auto&tradable_symbol (er.m_symbol);
        const std::string algo (er.m_algo);
        if (__builtin_expect(m_cfg->isManualStrategy(algo), 0)) {
            return true;
        }
        const auto& mkt (m_cfg->checkAlgoSymbol(algo, tradable_symbol, false));
        if (__builtin_expect(mkt == "",0)) {
            // algo or symbol not found
            return good;
        }
        const auto qty = er.m_qty;
        if (__builtin_expect(m_state.checkPosition(algo, mkt, qty, pm, true, (time_t) (er.m_recv_micro/1000000ULL)),1)) {
            return good;
        }

        // we don't need to notify if previous check not good
        if (good) {
            logError("RiskMonitor(%s) notifying trading pause on %s:%s",
                    m_name.c_str(), algo.c_str(), mkt.c_str());
            if (do_notify_pause) {
                m_status.notify_pause(algo, mkt, true);
            } else {
                m_status.setPause(algo, mkt, true);
            }
        }
        return false;
    }
    return good;
}

inline
Monitor::Monitor(const std::string& risk_cfg):
    m_name("default"),
    m_cfg(std::make_shared<Config>(risk_cfg==""? DEFAULT_RISK_JSON:risk_cfg)),  // loads the config

    m_state(m_cfg), // creates rate limits from config
    m_status(m_cfg) // loads the status from persists
{
    logInfo("RiskMonitor loaded from %s", m_cfg->m_config_file.c_str());
}

inline
void Monitor::set_instance_name(const std::string& name) {
    m_name = name;
    m_state.m_name = name;
    m_status.m_name = name;
}

inline
bool Monitor::checkNewOrder_mkt(const std::string& algo,
                                const std::string& mkt,
                                int64_t ord_qty,
                                time_t cur_utc) {

    if (__builtin_expect(m_cfg->isManualStrategy(algo), 0)) {
        return true;
    }

    if (__builtin_expect(m_status.getPause(algo, mkt), 0)) {
        logError("RiskMonitor(%s) %s:%s is Paused!", m_name.c_str(), algo.c_str(), mkt.c_str());
        // in pause
        return false;
    }
    cur_utc = cur_utc?cur_utc:utils::TimeUtil::cur_utc();
    if (__builtin_expect(!m_state.checkOrder(algo, mkt, ord_qty, cur_utc),0)) {
        // failed order count limiters
        return false;
    }

    return true;
}

inline
long long Monitor::maxPosition(const std::string& algo, const std::string& tradable) const {
    const auto& mkt (m_cfg->checkAlgoSymbol(algo, tradable, false));
    if (mkt=="") return 0;
    return m_cfg->m_strat_max_pos.find(algo)->second.find(mkt)->second;
}

/**********
 * Config
 *********/

inline
bool Config::isPaperTrading(const std::string& algo) const {
    if (__builtin_expect(isManualStrategy(algo),0)) {
        return false;
    }
    const auto iter=m_strat_paper_trading.find(algo); 
    if (__builtin_expect(iter ==m_strat_paper_trading.end(), 0)) {
        logError("RiskMonitor algo(%s) not found", algo.c_str());
        throw std::runtime_error("RiskMonitor algo " + algo + " not found");
    }
    return iter->second;
}

/**********
 * Status
 *********/

template<typename FMType>
void Status::persist_pause(const FMType& fm, const char* cmd) {
    // used by FM to write to the pause file, format:
    //   yyyymmdd, inst_name, Pause, algo, mkt, [ON|OFF]
    if (!fm.isFloorManager()) {
        logError("RiskMonitor.Status(%s): persist_pause() caller not a floor manager, not persisted!\n%s",
                m_name.c_str(), fm.toString().c_str());
        return;
    }
    logInfo("RiskMonitor.Status: persisting trading pause(%s:%s) to file (%s)",
            m_name.c_str(), cmd, m_persist_file.c_str());

    const auto tstr = utils::TimeUtil::frac_UTC_to_string();
    FILE *fp = fopen(m_persist_file.c_str(), "at");
    fprintf(fp, "%s, %s, Pause, %s\n", tstr.c_str(), m_name.c_str(), cmd);
    fclose(fp);
}

template<typename FMType>
void Status::set_operator_id(const FMType& fm, const std::string& operator_id) {
    if (m_tag50 == operator_id) {
        return;
    }
    m_tag50 = operator_id;
    if (!fm.isFloorManager()) {
        // persist
        //     yyyymmdd, inst_name, Operator, oper_id
        const auto tstr = utils::TimeUtil::frac_UTC_to_string();
        FILE *fp = fopen(m_persist_file.c_str(), "at");
        fprintf(fp, "%s, %s, Operator, %s\n", tstr.c_str(), m_name.c_str(), m_tag50.c_str());
        fclose(fp);
        return;
    };
    // else
    logError("RiskMonitor(%s) setting operator id(%s) from a non-FloorManager",
            m_name.c_str(), operator_id.c_str());
}

/********
 * State
 *******/

template<typename PositionManager>
bool State::checkPosition(const std::string& algo,
                          const std::string& mkt,
                          int64_t qty,
                          const PositionManager& pm,
                          bool is_from_report,
                          time_t cur_utc) {
    // check the max position, the pnl and flip
    // allow for risk reducing trades to go throw
    // if_from_report is true if it is called on a fill
    // from the execution report, in which case the
    // position state (for flip) will be updated regardless
    // of risk check outcome.
    //
    // return if the max_pos and pnl drawdown is good
    // Following quantities are queried from PM:
    //     algo_qty: the aggreated position of algo+mkt
    //     eng_qty:  the aggreated position of mkt of all algos in engine
    //     mkt_pnl:  the mark-to-market pnl of algo+mkt (all _N aggreated)
    //     algo_pnl: the mark-to-market pnl of algo (all mkts aggregated)

    // getting these quantities
    int64_t algo_qty = 0, eng_qty = 0;
    double algo_pnl = 0, mkt_pnl = 0;
    algo_qty = pm.getPosition_Market(&algo, &mkt, &mkt_pnl);
    pm.getPosition_Market(&algo, nullptr, &algo_pnl);
    eng_qty =  pm.getPosition_Market(nullptr, &mkt);
    bool strat_risk_reduce = (algo_qty+qty) * qty < 0;
    bool eng_risk_reduce   = (eng_qty+qty) * qty < 0;

    // check for algo maximum position
    if (__builtin_expect(std::abs((long long)(algo_qty+qty)) > m_cfg->m_strat_max_pos[algo][mkt],0)) {
        // risk reducing ?
        if (!strat_risk_reduce) {
            logError("RiskMonitor(%s) %s:%s qty %lld violates strategy maxpos: |%lld| > %lld", 
                    m_name.c_str(), algo.c_str(), mkt.c_str(), (long long) qty, 
                    (long long) (algo_qty+qty), (long long) m_cfg->m_strat_max_pos[algo][mkt]);
            return false;
        }
        logInfo("RiskMonitor(%s) %s:%s allowing risk reducing qty of %lld, although it violates strategy maxpos:  |%lld| > %lld", 
                m_name.c_str(), algo.c_str(), mkt.c_str(), (long long) qty, 
                (long long) (algo_qty+qty), (long long) m_cfg->m_strat_max_pos[algo][mkt]);
    }

    // check for eng maximum position
    if (__builtin_expect(std::abs((long long)(eng_qty+qty)) > m_cfg->m_eng_max_pos[mkt],0)) {
        // risk reducing ?
        if (!eng_risk_reduce) {
            logError("RiskMonitor(%s) %s:%s qty %lld violates engine maxpos: |%lld| > %lld", 
                    m_name.c_str(), algo.c_str(), mkt.c_str(), (long long) qty, 
                    (long long) (eng_qty+qty), (long long) m_cfg->m_eng_max_pos[mkt]);
            return false;
        }
        logInfo("RiskMonitor(%s) %s:%s allowing risk reducing qty of %lld, although it violates engine maxpos: |%lld| > %lld", 
                m_name.c_str(), algo.c_str(), mkt.c_str(), (long long) qty, 
                (long long) (eng_qty+qty), (long long) m_cfg->m_eng_max_pos[mkt]);
    }

    // check for algo pnl drawdown
    // TODO - consider moving the PNL check as a function
    if (__builtin_expect(algo_pnl < m_cfg->m_strat_pnl_drawdown[algo],0)) {
        // risk reducing ?
        if (!strat_risk_reduce) {
            logError("RiskMonitor(%s) %s:%s strategy(%s) overall mtm_pnl %lld hits drawdown %lld", 
                    m_name.c_str(), algo.c_str(), mkt.c_str(), algo.c_str(), (long long) algo_pnl,
                    (long long) m_cfg->m_strat_pnl_drawdown[algo]);
            return false;
        }
        logInfo("RiskMonitor(%s) %s:%s allowing risk reducing qty of %lld, although strategy's mtmpnl %lld hits drawdown %lld", 
                m_name.c_str(), algo.c_str(), mkt.c_str(), (long long) qty, (long long) algo_pnl,
                (long long) m_cfg->m_strat_pnl_drawdown[algo]);
    }

    // check for algo-mkt pnl drawdown
    if (__builtin_expect(mkt_pnl < m_cfg->m_strat_mkt_pnl_drawdown[algo][mkt],0)) {
        // risk reducing ?
        if (!strat_risk_reduce) {
            logError("RiskMonitor(%s) %s:%s strategy's market(%s) mtm_pnl %lld hits drawdown %lld", 
                    m_name.c_str(), algo.c_str(), mkt.c_str(), mkt.c_str(), (long long) mkt_pnl,
                    (long long) m_cfg->m_strat_mkt_pnl_drawdown[algo][mkt]);
            return false;
        }
        logInfo("RiskMonitor(%s) %s:%s allowing risk reducing qty of %lld, although strategy's market(%s) mtm_pnl %lld hits drawdown %lld", 
                m_name.c_str(), algo.c_str(), mkt.c_str(), (long long) qty, mkt.c_str(), (long long) mkt_pnl,
                (long long) m_cfg->m_strat_mkt_pnl_drawdown[algo][mkt]);
    }

    // check for too much position direction flippings
    if (__builtin_expect(!checkFlip(algo, mkt, algo_qty+qty, is_from_report, cur_utc),0)) {
        return false;
    }

    return true;
};

}  // namespace of risk
}  // namespace of pm
