#include "RiskMonitor.h"

#include <cmath>
#include <set>
#include <memory>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace pm {
namespace risk {

/**********
 * Config
 **********/
// const and static utilities
const std::vector<std::string>& Config::listAlgo() const {
    static std::vector<std::string> algo_vector;
    if (__builtin_expect(algo_vector.size()==0, 0)) {
        for (const auto& kv: m_strat_max_pos) {
            algo_vector.push_back(kv.first);
        }
    }
    return algo_vector;
}

const std::vector<std::string>& Config::listMarket() const {
    static std::vector<std::string> mkt_vec;
    if (__builtin_expect(mkt_vec.size() == 0, 0)) {
        std::set<std::string> mkt_set;
        for (const auto& symbol: listSymbol()) {
            mkt_set.insert(symbol.substr(0, symbol.find('_')));
        }
        for (const auto& mkt: mkt_set) {
            mkt_vec.push_back(mkt);
        };
    }
    return mkt_vec;
}

const std::vector<std::string>& Config::listSymbol() const {
    static std::vector<std::string> symbol_vector;
    if (__builtin_expect(symbol_vector.size()==0,0)) {
        symbol_vector = utils::SymbolMapReader::get().getPrimarySubscriptions();
    }
    return symbol_vector;
}

bool Config::algoExists(const std::string& algo) const{
    static std::set<std::string> algo_set(listAlgo().begin(), listAlgo().end());
    return (algo_set.find(algo) != algo_set.end());
}

bool Config::marketExists(const std::string& mkt) const {
    static std::set<std::string> mkt_set(listMarket().begin(), listMarket().end());
    return (mkt_set.find(mkt) != mkt_set.end());
}

bool Config::symbolExists(const std::string& symbol) const {
    static std::set<std::string> symbol_set(listSymbol().begin(), listSymbol().end());
    return (symbol_set.find(symbol) != symbol_set.end());
}

// return the mkt of symbol if algo+symbol found in map
std::string Config::checkAlgoSymbol(const std::string& algo, const std::string& tradable_symbol, bool verbose) const {
    if (__builtin_expect(!algoExists(algo),0)) {
        if (verbose) {
            logError("RiskMonitor failed to find algo(%s)", algo.c_str());
        }
        return "";
    }

    const auto& mkt(utils::SymbolMapReader::get().getTradableMkt(tradable_symbol));
    if (__builtin_expect(!marketExists(mkt),0)) {
        if (verbose) {
            logError("RiskMonitor failed to find mkt(%s) from symbol(%s)", 
                mkt.c_str(), tradable_symbol.c_str());
        }
        return "";
    }
    return mkt;
}

std::string Config::toStringConfig() const {
    // very long string
    std::stringstream ret;
    ret << "RiskMonitor Config (" << m_config_file << ")" << std::endl;
    // scale
    ret << "Scale:" << std::endl;
    for (const auto& strat: m_scale) {
        ret << "\t" << strat.first << ": " << strat.second << std::endl;
    }
    // engine max_pos
    ret << "Engine Max Position:" << std::endl;
    for (const auto& mp : m_eng_max_pos) {
        ret << "\t" << mp.first << ": " << mp.second << std::endl;
    }
    // strat max_pos/pnl
    ret << "Strategy Pnl Draw Down/Max Position:" << std::endl;
    for (const auto& mp: m_strat_max_pos) {
        const auto& strat = mp.first;
        ret << "\t" << strat << "(" << (isPaperTrading(strat)?"Paper":"Live") << ", Max Total Pnl Drawdown " << m_strat_pnl_drawdown.find(strat)->second << "):" << std::endl;
        for (const auto& mp0 : mp.second) {
            const auto& mkt(mp0.first);
            ret << "\t\t" << mkt << ": max_pos(" << mp0.second << "), max_drawdown(" << 
                m_strat_mkt_pnl_drawdown.find(strat)->second.find(mkt)->second << ")" << std::endl;
        }
    }
    return ret.str();
}

////////////
// load functions to be called during construction
////////////

Config::Config(const std::string& risk_file): m_config_file(risk_file) {
    load(risk_file);
}

// a constructor mainly for testing purpose
Config::Config(const std::string& risk_file,
        std::unordered_map<std::string, double> input_scale)
: m_config_file(risk_file),  m_scale(input_scale) {
    load(risk_file);
}

void Config::load (const std::string& risk_file) {
    // TODO - m_scale, spread, hourly, price_band not loaded
    // 1. read in the scale from weight and map
    // 2. read in participation rate - real rate = hourly + 
    // 3. ppp file

    auto reader = utils::ConfigureReader::getJson(risk_file.c_str());
    read_system(*reader); // read into m_volume_spread_file, m_prie_band, m_ppp_file
    if (m_scale.size() == 0) {
        read_strat_scale(*reader);  // into m_scale
    }
    read_engine(*reader);
    read_strategy(*reader);
}

void Config::read_strat_scale(const utils::ConfigureReader& reader) {
    const auto& scale_file = reader.get<std::string>("Scale.file");
    const auto& strat_reader = reader.getReader("Scale.map");
    const std::string tsc_str("TSC-7000-");
    const std::string tsd_str("TSD-7000-");
    const auto& file_token (utils::CSVUtil::read_file(scale_file, ':'));
    for (const auto& line: file_token) {
        // line in strat: scale
        const auto& sname(line[0]);
        const auto& sc(std::stod(line[1]));

        // allow the zero weight strategies
        // this would make setting scale in STRATEGY_WEIGHTS.yaml to
        // disable trading 
        //if (sc == 0) continue;

        // get the strat code from map
        try {
            const auto& code (strat_reader.get<std::string>(sname.c_str()));
            m_scale[tsc_str+code] = sc;
            m_scale[tsd_str+code] = sc;
        } catch (const std::exception& e) {
            logInfo("RiskMonitor: %s from file %s not defined in risk config, ignored",
                    sname.c_str(), scale_file.c_str());
        }
    }
    if (m_scale.size() == 0) {
        logError("RiskMonitor: problem reading the scale file %s", scale_file.c_str());
        throw std::runtime_error("RiskMonitor: problem reading the scale file " + scale_file);
    }
}

void Config::read_system(const utils::ConfigureReader& reader) {
    const auto& path (reader.get<std::string>("System.path"));
    const auto& fn_vs (reader.get<std::string>("System.files.volume_spread_csv"));
    m_volume_spread_file = path + "/" + reader.get<std::string>("System.files.volume_spread_csv");
    m_ppp_file =           path + "/" + reader.get<std::string>("System.files.price_position_pnl");
    m_status_file =        path + "/" + reader.get<std::string>("System.files.trading_status");
    m_manual_strat = reader.get<std::string>("System.manual_strategy");
    m_skip_replay = reader.get<bool>("System.skip_replay");
}

const Config::VolumeSpreadType Config::getVolumeSpread() const {
    VolumeSpreadType volume_spread;

    // read in volume_spread, a map of:
    // {mkt: pair<vector<pair<vol, vol_std>>, pair<spd, spd_std>>[276]}
    // from a csv file with format, volume with 3 lookback - 5m/15m/60m:
    // symbol, [volume, volume_std], spread, spread_std, * 5m_bars from 18 to 17
    const auto& file_token (utils::CSVUtil::read_file(m_volume_spread_file));
    const int bars = 276;
    const int nlb = 3; // number of lookbacks in volume/std
    for (const auto& line: file_token) {
        if (line.size() != bars * (nlb+1)*2 + 1) {
            logError("RiskMonitor failed to read volume_spread_csv file (%s) - line size not %d!", m_volume_spread_file.c_str(), bars*4+1);
            throw std::runtime_error("RiskMonitor failed to read volume_spread_csv file " + m_volume_spread_file);
        }
        const auto& mkt (line[0]);
        //volume_spread[mkt].reserve(bars);
        for (int i=0; i<bars; ++i) {
            int k = 1+i*(nlb+1)*2;
            std::vector<std::pair<double, double>> v;
            int lb = 0;
            for (; lb<nlb; ++lb) {
                v.push_back(std::make_pair( 
                        std::stod(line[k+lb*2]), 
                        std::stod(line[k+lb*2+1])));
            }

            const auto& pr ( std::make_pair(v, std::make_pair(std::stod(line[k+lb*2]), std::stod(line[k+lb*2+1]))) );
            /*
            volume_spread[mkt][i] = std::make_pair(
                    v,
                    std::make_pair(
                        std::stod(line[k+lb*2]), 
                        std::stod(line[k+lb*2+1])));
            */
            volume_spread[mkt].emplace_back(pr);
        }
    }

    // make sure all markets exist in the volume_spread
    for (const auto& mkt: listMarket()) {
        if (volume_spread.find(mkt) == volume_spread.end()) {
            logInfo("RiskMonitor warning loading config: market(%s) not found in volume_spread_5m file (%s)", mkt.c_str(), m_volume_spread_file.c_str());
            // still allows the load, these markets are not checked by volume and spread
            //throw std::runtime_error("RiskMonitor error loading config: market " + mkt + " not found in " + m_volume_spread_file);
        }
    }

    return volume_spread;
}

void Config::read_engine(const utils::ConfigureReader& reader) {

    // before everything else, read in the time varying volume and spread from 
    // given csv file once for all markets
    const auto& volume_spread(getVolumeSpread());

    // read the default first, and read any specific mkt to 
    // overwrite the previous default
    const auto& all_mkt (listMarket());
    // put in all for defaults before reading specifics
    const auto& default_reader (reader.getReader("Engine.default"));
    for (const auto& mkt: all_mkt) {
        read_engine_mkt(mkt, default_reader, volume_spread);
    }
    const auto& eng_reader (reader.getReader("Engine"));
    const auto& mkt_specific (eng_reader.listKeys());
    for (const auto& mkt: mkt_specific) {
        if (mkt == "default") {
            continue;
        }
        if (! marketExists(mkt) ) {
            logError("RiskMonitor error reading config for Engine.%s: mkt doesn't exist from symbol map", mkt.c_str());
            throw std::runtime_error("RiskMonitor error reader config for Engine." + mkt + " market doesn't exist from symbol map");
        }
        read_engine_mkt(mkt, eng_reader.getReader(mkt), volume_spread);
    }
}

void Config::read_engine_mkt(const std::string& mkt, const utils::ConfigureReader& reader, const VolumeSpreadType& volume_spread) {
    // read the engine limit at
    // m_eng_max_pos:      mkt: long long
    // m_eng_count,        mkt:         std::vector< (orders) int,  (seconds) int >
    // m_eng_rate_limit,   mkt: vector< std::vector< (rate) double, (seconds) int > >
    // m_eng_spread_limit  mkt: vector< double >
    // m_eng_fat_figner mkt: long long
    // m_price_ticks: mkt: int
    const auto& all_keys (reader.listKeys());

    // max_pos
    if (std::find(all_keys.begin(), all_keys.end(), "max_position")!=all_keys.end()) {
        long long max_pos = reader.get<long long>("max_position");
        m_eng_max_pos[mkt] = max_pos;
    }

    // order_count
    std::string key = "order_count";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        std::vector<std::pair<int, int>> ord_sec_vec;
        const auto& mr(reader.getReader(key));
        const size_t sz = mr.arraySize();
        key = "[";
        for (size_t i=0; i<sz; ++i) {
            const auto& mr_i (mr.getReader(key + std::to_string(i) + "]"));
            ord_sec_vec.push_back( std::make_pair (
                        mr_i.get<int>("orders"), 
                        mr_i.get<int>("seconds")));
        }
        m_eng_count[mkt] = ord_sec_vec;
    }

    // m_eng_fat_finger
    key = "fat_finger";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        m_eng_fat_finger[mkt] = reader.get<long long>(key.c_str());
    }

    // price_ticks
    key = "max_price_ticks";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        m_price_ticks[mkt] = reader.get<int>(key.c_str()) * 
            utils::SymbolMapReader::TickSize(mkt+"_N1");
    }

    // Given the volume_spread, populate the m_eng_rate_limit and m_eng_spread_limit
    // read in from "ratio", "std", "minutes"
    key = "participation_ratio";
    int nbars = 276;
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        std::vector<std::tuple<double, double, int>> rm_vec;
        const auto& rm(reader.getReader(key));
        const size_t sz = rm.arraySize();
        key = "[";
        for (size_t i=0; i<sz; ++i) {
            const auto& rm_i (rm.getReader(key + std::to_string(i) + "]"));
            rm_vec.push_back(std::make_tuple(
                        rm_i.get<double>("ratio"),
                        rm_i.get<double>("std"),
                        rm_i.get<int>("minutes")));
        }
        // enforce the 5m/15m/60m check
        int nlb = 3;
        if ((int)sz != nlb || 
                std::get<2>(rm_vec[0]) != 5 ||
                std::get<2>(rm_vec[1]) != 15 ||
                std::get<2>(rm_vec[2]) != 60 ) {
            logError("RiskMonitor error reading participation ratio: not 5/15/60m");
            throw std::runtime_error("RiskMonitor error participation ratio");
        }
        
        // populate for each bar
        std::vector<std::vector<std::pair<double, int>>> rl_bar;
        const auto& iter(volume_spread.find(mkt));
        if (iter != volume_spread.end()) {
            for (int bar=0; bar<nbars; ++bar) {
                const auto& vol (iter->second[bar].first); // pair<avg, std>[3]
                std::vector<std::pair<double, int>> rl;
                for (int lb=0; lb<nlb; ++lb) {
                    const auto& [ratio_, std_m_, min_] = rm_vec[lb];
                    const auto& [avg_, std_] = vol[lb];
                    rl.push_back( std::make_pair(ratio_* (avg_ + std_m_*std_), min_*60) );
                }
                rl_bar.push_back(rl);
            }
            m_eng_rate_limit[mkt] = rl_bar;
        }
        // else {
        //     logInfo("RiskMonitor %s volume not checked", mkt.c_str());
        // }
    }

    // spread_multiple
    key = "max_spread_std_multiple";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        auto spd_mul = reader.get<double>(key.c_str());
        std::vector<double> sl_bar;
        const auto& iter (volume_spread.find(mkt));
        if (iter != volume_spread.end()) {
            for (int bar=0; bar<nbars; ++bar) {
                const auto& [avg_, std_] = iter->second[bar].second;
                sl_bar.push_back( avg_ + spd_mul * (_MAX_(std_,avg_)) );
            }
            m_eng_spread_limit[mkt] = sl_bar;
        }
        // else {
        //     logInfo("RiskMonitor %s spread not checked", mkt.c_str());
        // }
    };
}

void Config::read_strategy(const utils::ConfigureReader& reader) {
    const auto& sr = reader.getReader("Strategy");
    const auto& all_strat = sr.listKeys();
    for (const auto& strat_i: all_strat) {
        if (m_scale.find(strat_i) == m_scale.end()) {
            logError("RiskMonitor config error: strategy(%s) not found in the scale", strat_i.c_str());
            throw std::runtime_error("RiskMonitor config error, scale not found for " + strat_i);
        }
        const auto& smr0 = sr.getReader(strat_i);

        // get paper and drawdown
        m_strat_paper_trading[strat_i] = (smr0.get<int>("paper_trading") != 0);
        // get strategy maximum pnl drawdown, scaled
        m_strat_pnl_drawdown[strat_i] = -(double)((long long)(smr0.get<double>("max_strat_pnl_drawdown") * m_scale[strat_i] +0.5));

        // get per-market specifications
        const auto& smr = smr0.getReader("markets");
        const auto& strat_mkt = smr.listKeys();
        // make sure strat_mkt exists
        for (const auto& mkt_i : strat_mkt) {
            if (mkt_i=="default") continue;
            if (!marketExists(mkt_i)) {
                logError("RiskMonitor config error: strategy(%s)'s market(%s) doesn't exist!", strat_i.c_str(), mkt_i.c_str());
                throw std::runtime_error("RiskMonitor config error: " + strat_i + "'s market " + mkt_i + " doesn't exist!");
            }
        }
        // trying to find default and read first
        if (std::find(strat_mkt.begin(), strat_mkt.end(), "default") != strat_mkt.end()) {
            // fill in all the strategy's defined market with default
            const auto& smr_d (smr.getReader("default"));
            //for (const auto& mkt_i : strat_mkt) {
            const auto& allmkt (listMarket());
            for (const auto& mkt_i : allmkt) {
                if (mkt_i == "default") {
                    continue;
                }
                read_strategy_mkt(strat_i, mkt_i, smr_d);
            }
        }
        // read the market specific for strat_i
        for (const auto& mkt_i : strat_mkt) {
            if (mkt_i == "default") {
                continue;
            }
            const auto& smr_i (smr.getReader(mkt_i));
            read_strategy_mkt(strat_i, mkt_i, smr_i);
        }
    }
}

void Config::read_strategy_mkt(const std::string& strat, const std::string& mkt, const utils::ConfigureReader& reader) {
    const auto& all_keys(reader.listKeys());

    // first the maxpos
    std::string key = "max_position";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        m_strat_max_pos[strat][mkt]=(long long)(reader.get<long long>(key.c_str()) * m_scale[strat] + 0.5);
    }

    // mkt pnl drawdown, should scale
    key = "max_mkt_pnl_drawdown";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        m_strat_mkt_pnl_drawdown[strat][mkt] = -(double)((long long)(reader.get<double>(key.c_str()) * m_scale[strat] +0.5));
    }


    // order count
    key = "order_count";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        const auto& rdr (reader.getReader(key));
        size_t sz = rdr.arraySize();
        std::vector<std::pair<int, int>> cnts;
        key = "[";
        for (size_t i=0; i<sz; ++i) {
            const auto& rdr0 (rdr.getReader(key + std::to_string(i) + "]"));
            cnts.push_back(std::make_pair( 
                        rdr0.get<int>("orders"), 
                        rdr0.get<int>("seconds")));
        }
        m_strat_count[strat][mkt]=cnts;
    }

    // trade rate
    key = "trade_rate";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        const auto& rdr (reader.getReader(key));
        size_t sz = rdr.arraySize();
        std::vector<std::pair<double, int>> rates;
        key = "[";
        for (size_t i=0; i<sz; ++i) {
            const auto& rdr0 (rdr.getReader(key + std::to_string(i) + "]"));
            int seconds = rdr0.get<int>("minutes") * 60;
            /// max_pos_turnover: 2 times max_position (max_long to max_short) in given time window as rate
            // we are not calculating the rate here, to allow it to be specified 
            // in default but not in markets, with different maxpos of course
            rates.push_back(std::make_pair(
                        rdr0.get<double>("max_pos_turnover"),
                        seconds));
        }
        m_strat_rate_limit[strat][mkt]=rates;
    }

    // flip count
    key = "flip_count";
    if (std::find(all_keys.begin(), all_keys.end(), key) != all_keys.end()) {
        const auto& rdr (reader.getReader(key));
        size_t sz = rdr.arraySize();
        std::vector<std::pair<int, int>> flips;
        key = "[";
        for (size_t i=0; i<sz; ++i) {
            const auto& rdr0 (rdr.getReader(key + std::to_string(i) + "]"));
            flips.push_back(std::make_pair( 
                        rdr0.get<int>("flips"),
                        rdr0.get<int>("hours")*3600));
        }
        m_strat_flip[strat][mkt]=flips;
    }
}

/*********
 * Status
 *********/
Status::Status(std::shared_ptr<Config> cfg)
: m_name("default"),
  m_persist_file(cfg->m_status_file),
  m_fcu(m_name.c_str()),
  m_cfg(cfg)
{
    // read the latest trading status 
    // and populate m_status
    load_pause();
    logInfo("RiskMonitor: Status %s loaded: %s", m_name.c_str(), toStringStatus().c_str());
}

bool Status::setPause(const std::string& algo, const std::string& market, bool if_pause) {
    if ( (algo == "") || (algo == "ALL")) {
        return setPauseSymbol(market, if_pause);
    }

    if ((market == "") || (market == "ALL")) {
        return setPauseAlgo(algo, if_pause);
    }

    // market could be a ':' delimitered list
    const auto& mkt_list (utils::CSVUtil::read_line(market.c_str(),':'));
    for (const auto& mkt:mkt_list) {
        if (__builtin_expect(m_cfg->algoExists(algo) && m_cfg->marketExists(mkt), 1)) {
            m_status[algo][mkt].set_pause(if_pause);
            //debug
            //printf("setting pause %s, %s, %s\n", algo.c_str(), market.c_str(), if_pause?"Paused":"Live");

            //logInfo("RiskMonitor %s:%s setting trading status: %s\n", algo.c_str(), market.c_str(), if_pause?"Paused":"Live");
            continue;
        }
        logError("RiskMonitor(%s) Failed to set Pause: algo(%s) or mkt(%s) not found", m_name.c_str(), algo.c_str(), mkt.c_str());
        fprintf(stderr,"RiskMonitor(%s) Failed to set Pause: algo(%s) or mkt(%s) not found", m_name.c_str(), algo.c_str(), mkt.c_str());
        return false;
    }
    return true;
}

bool Status::setPauseSymbol(const std::string& market, bool if_pause) {
    const auto& algo_vec (m_cfg->listAlgo());
    bool ret = true;
    for (const auto& algo: algo_vec) {
        ret &= setPause(algo, market, if_pause);
    }
    return ret;
}

bool Status::setPauseAlgo(const std::string& algo, bool if_pause) {
    const auto& symbol_vec (m_cfg->listMarket());
    bool ret = true;
    for (const auto& market: symbol_vec) {
        ret &= setPause(algo, market, if_pause);
    }
    return ret;
}

bool Status::setPauseAll(bool if_pause) {
    return setPause("","",if_pause);
}

bool Status::getPause(const std::string& algo, const std::string& market) const {
    const auto iter = m_status.find(algo);
    if (iter != m_status.end()) {
        const auto iter2 = iter->second.find(market);
        if (iter2 != iter->second.end()) {
            return iter2->second.is_paused();
        } else {
            logError("RiskMonitor(%s) getPause() failed - market %s not found", 
                    m_name.c_str(), market.c_str());
        }
    } else {
        logError("RiskMonitor(%s) getPause() failed - algo %s not found", m_name.c_str(), algo.c_str());
    }
    return true;
}

std::string Status::queryPause(const std::string& algo, const std::string& market) const {
    std::string ret = "Trading status from(" + m_name + ")\n";
    std::vector<std::string> algo_list;
    if (algo == "" || algo == "ALL") {
        algo_list = m_cfg->listAlgo();
    } else {
        algo_list.push_back(algo);
    }
    std::vector<std::string> mkt_list;
    if (market == "" || market == "ALL") {
        mkt_list = m_cfg->listMarket();
    } else {
        mkt_list = utils::CSVUtil::read_line(market.c_str(),':');
    }
    for (const auto& algo:algo_list) {
        ret += (algo + ": ");
        for (const auto& mkt: mkt_list) {
            ret += (mkt+ "(" + (getPause(algo,mkt)?"STOP":"LIVE") + ") ");
        }
        ret += "\n";
    }
    return ret;
}

void Status::notify_pause(const std::string& algo, const std::string& market, bool if_pause) {
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
    //
    // NOTE2 - there could be race condition if all RM on floor, upon updating 
    // a same ER, call notify and waiting for ack. Therefore, FM should not call
    // notify, but run persist instead. 
    auto cmd = std::string("Z ") + algo + ", " + market + (if_pause? ", ON":", OFF");
    auto resp = m_fcu.sendReq(cmd);
    logInfo("RiskMonitor(%s): Trading Status notify (%s) got response (%s)", m_name.c_str(), cmd.c_str(), resp.c_str());
    setPause(algo, market, if_pause);
}

const std::string Status::toStringStatus() const {
    std::string ret;
    int pause_cnt = 0;
    for (const auto& stat: m_status) {
        const auto& strat(stat.first);
        std::string ret0 = strat + ", ";
        int this_cnt = 0;
        for (const auto& kv: stat.second) {
            const auto& mkt(kv.first);
            const auto& if_pause(kv.second.is_paused());
            if (if_pause) {
                ret0 += ((this_cnt>0?":":"") + mkt);
                this_cnt++;
            }
        }
        if (this_cnt) {
            pause_cnt += this_cnt;
            if (this_cnt == (int)stat.second.size()) {
                ret += (strat + ", ALL, ON\n");
            } else {
                ret += (ret0 + ", ON\n");
            }
        }
    }
    if (pause_cnt == 0) {
        ret += "ALL, ALL, OFF";
    }
    return ret;
}

void Status::load_pause() {
    // load pause file and
    // format of either
    //     yyyymmdd, inst_name, Pause, algo, market_list, ON|OFF
    // or 
    //     yyyymmdd, inst_name, Operator, oper_id
    const auto& file_token (utils::CSVUtil::read_file(m_persist_file));
    int line_number = 0;

    m_tag50 = "Invalid";
    // initialize with all NOT paused
    setPause("","",false);
    for (const auto& line: file_token) {
        if ((line.size() == 6) && (line[2] == "Pause")) {
            const auto& algo(line[3]);
            const auto& market(line[4]);
            setPause(algo, market, line[5]=="ON");
        } else if ((line.size() == 4) && (line[2] != "OperatorID")){
            m_tag50 = line[3];
        } else {
            logError("RiskMonitor(%s): Pause File Reading Error: unregonized line found from %s, line %d",
                    m_name.c_str(), m_persist_file.c_str(), line_number);
            throw std::runtime_error("Pause File reading error" + m_persist_file);
        }
        ++line_number;
    }
    if (m_tag50 == "Invalid") {
        throw std::runtime_error("no tag50 set from status file "+ m_persist_file);
    }
    //logInfo("RiskMonitor(%s): Read Pause file, current pause status: %s", m_name.c_str(), toStringStatus().c_str());
}

/*
 * Use a status file written by FM
static const int QLen = 1024*1024*64; // 64M
using QType = utils::MwQueue<QLen, utils::ShmCircularBuffer>;
std::shared_ptr<QType> m_q;
std::shared_ptr<QType::Reader> m_q_reader;
std::shared_ptr<QType::Writer> m_q_writer;
*/

/* no need to maintain this, just use fcu
using ChannelType = std::unique_ptr<utils::Floor::Channel>;
ChannelType m_flr_client;
*/


/********
 * State
 ********/

// State could create from config, or retrieved from previous persist
State::State(std::shared_ptr<Config> cfg) : m_name("default"), m_cfg(cfg) {
    init();
}

bool State::checkOrder(const std::string& algo,
                       const std::string& mkt,
                       int64_t ord_qty,
                       time_t cur_utc) {
    // checks the strategy and eng level count+rate limiters, if all good, then update all
    // otherwise, log error and return false

    // strat level order side, check only
    uint64_t cur_micro = cur_utc*1000*1000ULL;
    int cnt = (int)std::abs(ord_qty);

    // fat finger?
    if (__builtin_expect(m_cfg->m_eng_fat_finger[mkt] < cnt,0)) {
        logError("RiskMonitor(%s) %s:%s order size too large (fat finger detected): %d > %d",
                m_name.c_str(), algo.c_str(), mkt.c_str(), cnt, (int)m_cfg->m_eng_fat_finger[mkt]);
        return false;
    }

    auto& [rle_s, re_s] = m_strat[algo][mkt][0];  // < rle, re >
    if (__builtin_expect(rle_s->checkOnly(cur_micro, 1)!=0, 0)) {
        logError("RiskMonitor(%s) %s:%s order of %d would break strategy order count: %s",
                m_name.c_str(), algo.c_str(), mkt.c_str(), ord_qty, rle_s->toString().c_str());
        return false;
    }
    if (__builtin_expect(re_s->checkOnly(cur_micro, cnt)!=0, 0)) {
        logError("RiskMonitor(%s) %s:%s order of %d would break strategy trade rate: %s",
                m_name.c_str(), algo.c_str(), mkt.c_str(), ord_qty, re_s->toString().c_str());
        return false;
    }

    // engine level order side, check only
    auto& [rle_e, re_e] = m_eng[mkt][0];  // < rle, re >
    if (__builtin_expect(rle_e->checkOnly(cur_micro, 1)!=0, 0)) {
        logError("RiskMonitor(%s) %s:%s order of %d would break engine order count: %s",
                m_name.c_str(), algo.c_str(), mkt.c_str(), ord_qty, rle_e->toString().c_str());
        return false;
    }

    if (__builtin_expect(!checkTradeRate(cur_utc, mkt, cnt, re_e), 0)) {
        return false;
    }

    // all good! do update
    rle_s->updateOnly(cur_micro, 1);
    re_s->updateOnly(cur_micro, cnt, false);
    rle_e->updateOnly(cur_micro, 1);
    re_e->updateOnly(cur_micro, cnt, false);
    return true;
}

bool State::reportFill (const std::string& algo,
                        const std::string& mkt,
                        int64_t fill_qty,
                        time_t cur_utc) {
    // do updateOnly for all rate limits, since the fill already happened,
    // and check the results, log error if non-zero returned
    bool ret = true;

    // strat level report side, update and check return
    uint64_t cur_micro = cur_utc * 1000ULL*1000ULL;
    int cnt = (int)std::abs(fill_qty);
    auto& re_s (m_strat[algo][mkt][1].second);  // < rle, re >
    if (__builtin_expect(re_s->updateOnly(cur_micro, cnt)!=0, 0)) {
        logError("RiskMonitor(%s) %s:%s reported fill of %d violates strategy trade rate: %s", 
                m_name.c_str(), algo.c_str(), mkt.c_str(), fill_qty, re_s->toString().c_str());
        ret = false;
    }

    // engine level report side, update and check return
    auto& re_e(m_eng[mkt][1].second);  // < rle, re >
    re_e->updateOnly(cur_micro, cnt, false);
    if (__builtin_expect(!checkTradeRate(cur_utc, mkt, 0, re_e), 0)) {
        ret = false;
    }

    return ret;
}

bool State::reportNew (const std::string& algo,
                       const std::string& mkt,
                       time_t cur_utc) {
    // do updateOnly of 1 for all order count limiters, 
    // and check the results, log error if non-zero returned
    bool ret = true;

    // strat level report side, update and check return
    uint64_t cur_micro = cur_utc * 1000ULL*1000ULL;
    auto& rle_s(m_strat[algo][mkt][1].first);  // < rle, re >
    if (__builtin_expect(rle_s->updateOnly(cur_micro, 1)!=0, 0)) {
        logError("RiskMonitor(%s) %s:%s reported NEW violates strategy order count: %s",
                m_name.c_str(), algo.c_str(), mkt.c_str(), rle_s->toString().c_str());
        ret = false;
    }

    // engine level report side, update and check return
    auto& rle_e (m_eng[mkt][1].first);  // < rle, re >
    if (__builtin_expect(rle_e->updateOnly(cur_micro, 1)!=0, 0)) {
        logError("RiskMonitor(%s) %s:%s reported  NEW violates engine order count: %s",
                m_name.c_str(), algo.c_str(), mkt.c_str(), rle_e->toString().c_str());
        ret = false;
    }

    return ret;
}

void State::reportCancel (const std::string& algo,
                          const std::string& mkt) {
    // remove order count from both order and report side
    for (int i=0; i<2; ++i) {
        auto& rle_s(m_strat[algo][mkt][i].first);  // < rle, re >
        rle_s->removeOnly(1);
        auto& rle_e(m_eng[mkt][i].first);  // < rle, re >
        rle_e->removeOnly(1);
    }
}

void State::init() {
    // populate the limiters according to the config
    //
    // Engine level first, populate m_eng from m_cfg's m_eng_count and m_eng_rate_limit
    //
    //using LimitMap = std::unordered_map<
    //    std::string,         // per-mkt
    //        std::pair<std::shared_ptr<utils::RateLimiterEns>,     // order count limits
    //                  std::shared_ptr<utils::RateEstimator>>[2]>; // order rate/turn over limits
    for (const auto& mkt: m_cfg->listMarket()) {
        const auto& cl (m_cfg->m_eng_count[mkt]);  // vector< <ords, secs> >
        const auto& rl (m_cfg->m_eng_rate_limit[mkt]); // vector< vector <rate, secs> >
        
        // create the RateLimiterEns
        int hist_mul = 1;  // needs 1 times of history in case of cancels
        std::vector<std::tuple<int, time_t, int>> cl_cfg;
        for (const auto& cl_: cl) {
            cl_cfg.push_back(std::make_tuple(cl_.first, cl_.second, hist_mul));
        }
        std::shared_ptr<utils::RateLimiterEns> rle_1 (std::make_shared<utils::RateLimiterEns>(cl_cfg));
        std::shared_ptr<utils::RateLimiterEns> rle_2 (std::make_shared<utils::RateLimiterEns>(cl_cfg));

        // create the RateEstimator
        // here we are going to use these objects just to keep trace of current
        // rates in different look backs, i.e. 5m/15m/60m, the rates are checked
        // against a time-varying limit, specified at each 5m bar.
        // We specify a very high rate limit here just for simplicity at RateEstimator
        double high_rate = 1e+5;
        int bucket_seconds = 15; // rates are aggregated at 15 seconds
        std::vector<std::pair<double, int>> rl_cfg;
        if (rl.size() > 0) {
            for (const auto& rl_: rl[0]) {
                rl_cfg.push_back(std::make_pair(high_rate, rl_.second));
            }
        } else {
            // mkt not defined in trade rate csv, use a default here
            // its rate won't be checked
            rl_cfg.push_back(std::make_pair(high_rate, bucket_seconds*4));
        }
        std::shared_ptr<utils::RateEstimator> re_1 (std::make_shared<utils::RateEstimator>(rl_cfg, bucket_seconds));
        std::shared_ptr<utils::RateEstimator> re_2 (std::make_shared<utils::RateEstimator>(rl_cfg, bucket_seconds));
        m_eng[mkt][0] = std::make_pair(rle_1, re_1);
        m_eng[mkt][1] = std::make_pair(rle_2, re_2);
    }

    // populate the 
    //     std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<int, int>>> m_strat_count;
    //     std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<double, int>>> m_strat_rate_limit; // strat - mkt - vector< <turnover_rate, seconds> >
    // onto
    //     std::unordered_map<std::string, LimitMap> m_strat;
    //     std::unordered_map<std::string, LimitMap> m_flip;

    for (const auto& algo: m_cfg->listAlgo()) {
        for (const auto& mkt: m_cfg->listMarket()) {
            const auto& cl (m_cfg->m_strat_count[algo][mkt]); // vector< <orders, seconds> >
            const auto& rl (m_cfg->m_strat_rate_limit[algo][mkt]); // vector< turnover_rate, seconds> >
            const auto& fl (m_cfg->m_strat_flip[algo][mkt]); // vector< <flips, seconds> >
            
            // RateLimitEns
            int hist_mul = 2; // allow more cancel as needed
            std::vector<std::tuple<int, time_t, int>> cl_cfg;
            for (const auto& cl_: cl) {
                cl_cfg.push_back(std::make_tuple(cl_.first, cl_.second, hist_mul));
            }
            std::shared_ptr<utils::RateLimiterEns> rle_1 (std::make_shared<utils::RateLimiterEns>(cl_cfg));
            std::shared_ptr<utils::RateLimiterEns> rle_2 (std::make_shared<utils::RateLimiterEns>(cl_cfg));

            // RateEstimator for turnover
            std::vector<std::pair<double, int>> rl_cfg; // vector of rate limits
            for (const auto& rl_: rl) {
                int seconds = rl_.second;
                // rate is (number of turnover * max_pos_scaled * 2)/seconds
                double rate = rl_.first * m_cfg->m_strat_max_pos[algo][mkt] * 2.0/seconds;
                rl_cfg.push_back(std::make_pair(rate, seconds));
            }
            int bucket_seconds = 30; // rates are aggregated at bucket seconds
            std::shared_ptr<utils::RateEstimator> re_1 (std::make_shared<utils::RateEstimator>(rl_cfg, bucket_seconds));
            std::shared_ptr<utils::RateEstimator> re_2 (std::make_shared<utils::RateEstimator>(rl_cfg, bucket_seconds));
            m_strat[algo][mkt][0] = std::make_pair(rle_1, re_1);
            m_strat[algo][mkt][1] = std::make_pair(rle_2, re_2);

            // flip count, only has rate_limit at both sides
            hist_mul = 1; // allow more cancel as needed
            std::vector<std::tuple<int, time_t, int>> fl_cfg;
            for (const auto& fl_: fl) {
                fl_cfg.push_back(std::make_tuple(fl_.first, fl_.second, hist_mul));
            }
            std::shared_ptr<utils::RateLimiterEns> rlf_1 (std::make_shared<utils::RateLimiterEns>(fl_cfg));
            std::shared_ptr<utils::RateLimiterEns> rlf_2 (std::make_shared<utils::RateLimiterEns>(fl_cfg));
            m_flip[algo][mkt][0].first = rlf_1;
            m_flip[algo][mkt][1].first = rlf_1;
        }
    }
    logInfo("RiskMonitor State started!\n%s",toString().c_str());
}

bool State::checkTradeRate(time_t cur_utc, const std::string& mkt, int ord_cnt, std::shared_ptr<utils::RateEstimator>& re) {
    // current trade rate, all lookbacks
    const auto& rates (re->checkRates(cur_utc,ord_cnt));

    // current bar index
    int bar_idx = getBarIdx(cur_utc);

    const auto& iter(m_cfg->m_eng_rate_limit.find(mkt));
    if (__builtin_expect(iter==m_cfg->m_eng_rate_limit.end(),0)) {
        logInfo("RiskMonitor(%s) warning: %s trade rate not defined and couldn't check", m_name.c_str(), mkt.c_str());
        return true;
    }

    // check it against the m_cfg->m_eng_rate_limit on bar_idx
    const auto& rl (iter->second[bar_idx]); // vector<rate, seconds> at bar_idx
    for(size_t i=0; i<rates.size(); ++i) {
        if (__builtin_expect(rates[i]*rl[i].second > rl[i].first,0)) {
            logError("RiskMonitor(%s) %s engine trade rate violation for lookback of %d seconds: cur(%.2f orders), participation_limit(%.2f orders)",
                    m_name.c_str(), mkt.c_str(), (int)rl[i].second, rates[i]*rl[i].second, rl[i].first);
            return false;
        }
    }
    return true;
}

int State::getBarIdx(time_t cur_utc) const {
    static time_t utc0 (utils::TimeUtil::startUTC());
    int barsec = 300; // 5m
    int bar_idx = (((cur_utc + 24*3600) - utc0) % (24*3600))/barsec;
    int maxix = 275;
    if (__builtin_expect(bar_idx>maxix,0)) bar_idx=maxix;
    return bar_idx;
}

bool State::checkFlip(const std::string& algo, const std::string& mkt, long long new_pos_qty, bool is_fill_report, time_t cur_utc) {
    // check if the "new_pos_qty" causes a "flip", i.e.
    // a non-zero position with a different sign with previous non-zero position
    // if is_fill_report, update m_prev_direction[algo][mkt] from prev_pos_qty to new_pos_qty
    // usually only the report side updates the previous position

    // for the purpose of tracking position direction,
    // positions less or equal to MIN_QTY are ignored
    // for, i.e. overfill, etc
    static const int64_t MIN_QTY = 10;
    if (std::abs(new_pos_qty) <= MIN_QTY) {
        return true;
    }

    // new_pos_qty non-zero
    int64_t prev_pos_qty = 0;
    auto iter = m_prev_direction.find(algo);
    if (__builtin_expect( iter != m_prev_direction.end(), 1)) {
        auto iter2 = iter->second.find(mkt);
        if (__builtin_expect( iter2 != iter->second.end(), 1)) {
            prev_pos_qty = iter2->second;
            if (is_fill_report) {
                iter2->second = new_pos_qty;
            }
            if (__builtin_expect(prev_pos_qty * new_pos_qty < 0,0)) {
                // this would flip the direction of previous position
                // check the flip count
                bool ret = is_fill_report?
                    (m_flip[algo][mkt][1].first->updateOnly(cur_utc*1000000ULL, 1) == 0):
                    (m_flip[algo][mkt][0].first->checkOnly(cur_utc*1000000ULL, 1) == 0);
                if (__builtin_expect(!ret, 0)) {
                    logError("RiskMonitor(%s) %s:%s position flip detected on new position of %lld!", 
                            m_name.c_str(), algo.c_str(), mkt.c_str(), new_pos_qty);
                }
                return ret;
            }
        }
    }
    // not found, update non-zero qty
    if (is_fill_report) {
        m_prev_direction[algo][mkt] = new_pos_qty;
    }
    return true;
}

/**********
 * Monitor
 *********/
bool Monitor::checkLimitPrice(const std::string& symbol, double price, time_t cur_utc) const {
    // check current bbo
    double bidpx, askpx;
    int bidsz, asksz;
    if (__builtin_expect(!md::getBBO(symbol,  bidpx, bidsz, askpx, asksz),0)) {
        logError("RiskMonitor(%s) %s: Cannot get bbo", m_name.c_str(), symbol.c_str());
        return false;
    }

    if (__builtin_expect(bidsz*asksz*bidpx*askpx == 0, 0)) {
        logError("RiskMonitor(%s) %s: market has zero size or price on one or more sides: bid(%d@%.7f):ask(%d@%.7f)",
                m_name.c_str(), symbol.c_str(), bidsz, bidpx, asksz, askpx);
        return false;
    }

    const auto& mkt(utils::SymbolMapReader::get().getTradableMkt(symbol));
    if (__builtin_expect(!m_cfg->marketExists(mkt),0)) {
        logError("RiskMonitor(%s) %s: failed to find market from symbol", 
            m_name.c_str(), symbol.c_str());
        return false;
    }

    // check the price limit 
    const double diff_bid = std::fabs(bidpx-price);
    const double diff_ask = std::fabs(askpx-price);
    const double diff_px = _MIN_(diff_bid, diff_ask);
    if (__builtin_expect(diff_px > m_cfg->m_price_ticks.find(mkt)->second + 1e-10,0)) {
        logError("RiskMonitor(%s) %s: price (%.7f) is too far away from the BBO(%.7f:%.7f): %.7f>%.7f",
                m_name.c_str(), symbol.c_str(), price, bidpx, askpx, diff_px, m_cfg->m_price_ticks.find(mkt)->second);
        return false;
    }

    // check the per-bar spread limit
    const auto& iter(m_cfg->m_eng_spread_limit.find(mkt));
    if (__builtin_expect(iter==m_cfg->m_eng_spread_limit.end(),0)) {
        logInfo("RiskMonitor(%s) warning: %s spread not defined and not checked", m_name.c_str(), mkt.c_str());
        return true;
    }
    int bar_idx = m_state.getBarIdx(cur_utc);
    const double spd_limit = iter->second[bar_idx];
    const double spd = askpx-bidpx;

    if (__builtin_expect(spd > spd_limit + 1e-10, 0)) {
        logError("RiskMonitor(%s) %s: spread %.7f(%.7f:%.7f) exceed limit of %.7f at bar(%d)",
                m_name.c_str(), symbol.c_str(), spd, bidpx, askpx, spd_limit, bar_idx);
        return false;
    }
    return true;
}

} // namespace risk
} // namespace pm
