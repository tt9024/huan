#include "idbo_tf.h"
#include <cmath>
#include <stdexcept>

namespace algo {
    IDBO_TF::IDBO_TF (const std::string& name, const std::string& cfg, pm::FloorBase::ChannelType& channel, uint64_t cur_micro)
    : AlgoBase(name, channel),
      m_name(name),
      m_cfg(cfg),
      m_weight(0)
    {
        onReload(cur_micro, m_cfg);
        logInfo("%s created with cfg %s", m_name.c_str(), m_cfg.c_str());
    }

    IDBO_TF::IDBO_TF (const std::string& name, const std::string& cfg, pm::FloorBase::ChannelType& channel)
    : AlgoBase(name, channel),
      m_name(name),
      m_cfg(cfg)
    {
        logInfo("%s created with cfg %s, parameter and states are not loadeed! ", m_name.c_str(), m_cfg.c_str());
    }


    IDBO_TF::~IDBO_TF() {
        logInfo("%s destructed", m_name.c_str());
    }

    void IDBO_TF::onReload(uint64_t cur_micro, const std::string& config_file) {
        removeAllSymbol();
        m_param.clear(); // remnove all subscriptions
        try {
            // read the configuration file here
            const auto& reader = utils::ConfigureReader(config_file.c_str());

            // check the trading date
            const auto& cur_trading_day = utils::TimeUtil::curTradingDay2();
            if (reader.get<std::string>("update_date") != cur_trading_day) {
                throw std::runtime_error(std::string("Config update date ") + reader.get<std::string>("update_date") + " not equalt to current trading day " + cur_trading_day);
            }
            m_weight = reader.get<double>("strategy_weight");
            const auto& sym_reader = reader.getReader("symbols");
            for (const auto& key: sym_reader.listKeys()) {
                auto new_param( std::make_shared<Param> (key, sym_reader.getReader(key),  (time_t)(cur_micro/1000000ULL)));
                const auto& bcfg(new_param->_bcfg);
                new_param->_symid = addSymbol(bcfg->venue, bcfg->symbol, bcfg->type, IDBO_TF::Barsec);
                m_param[new_param->_symid] = new_param;
            }
        } catch (const std::exception& e) {
            logError("%s failed to load parameter file %s: %s", 
                    m_name.c_str(), m_cfg.c_str(), e.what());
            return;
        }

        // stop if it's currrently running
        bool need_restart = false;
        if (shouldRun()) {
            onStop(cur_micro);
            setShouldRun(false);
            need_restart = true;
        }

        // get historical bar so far
        initState(cur_micro/1000000);

        // restart the strategy if needed
        if (need_restart) {
            onStart(cur_micro);
        }
    }

    void IDBO_TF::onStart(uint64_t cur_micro) {
        setShouldRun(true);
        onOneSecond(cur_micro);
    }

    void IDBO_TF::onStop(uint64_t) {}

    void IDBO_TF::onOneSecond(uint64_t cur_micro) {
        for (const auto& p : m_param) {
            int symid = p.first;
            const auto& param (*p.second);
            auto& state (*m_state[symid]);

            // don't run if trigger not due yet
            const auto& trigger_time(state._trigger_time);
            int& next_idx (state._next_trigger_idx);

            if (trigger_time.size() <= (size_t) next_idx) {
                // done for the day
                continue;
            }
            if ((time_t)(cur_micro/1000000ULL) < trigger_time[next_idx]) {
                // not yet
                continue;
            }
            if ((size_t)++next_idx >= trigger_time.size()) {
                logInfo("%s running last bar (%s)", m_name.c_str(), param.toString(false).c_str());
            }
            updateState(param, state, cur_micro);
        }
    }

    void IDBO_TF::setPosition(State& state, time_t cur_utc) {
        auto tgt_position = state._tgt_pos;
        if (tgt_position != state._pos) {
            time_t tgt_utc = utils::TimeUtil::cur_utc() + 5*60;
            if (!setPositionTWAP2(state._symid, tgt_position, tgt_utc)) {
                logError("%s failed to set position to %lld from %lld", 
                        m_name.c_str(), (long long) tgt_position, 
                        (long long) state._pos);
            } else {  
                state._pos_td = m_param[state._symid]->_trade_day;
            }
        }
    }

    bool IDBO_TF::updateState(const Param& param, State& state, uint64_t cur_micro) {
        // update snap
        auto symid = state._symid;
        getSnap(symid, state._bookL1);
        getBar(symid,  state._bar);

        int64_t qty_done, qty_open;
        if (!getPosition(symid, qty_done, qty_open)) {
            logError("%s failed to get position!", m_name.c_str());
            return false;
        }
        state._pos = qty_done + qty_open;
        idbo(param, state, cur_micro);
        setPosition(state, cur_micro/1000000);
        state.persist();
        return true;
    }

    void IDBO_TF::idbo(const Param& param, State& state, uint64_t cur_micro) {
        time_t cur_utc = cur_micro/1000000;
        bool first_bar = (cur_utc == param._start_utc);
        // initialize state if needed
        if (state._last_updated == 0) {
            state._h = state._bar.high;
            state._l = state._bar.low;
        }
        state._last_updated = cur_utc;

        //state._tgt_pos = state._pos;
        if (state._pos != 0) {
            // active position
            if (first_bar) {
                // first bar
                if (state._pos > 0) {
                    state._stop += (param._inactive_bars*state._ic*(state._h - state._stop));
                } else {
                    state._stop += (param._inactive_bars*state._ic*(state._l - state._stop));
                }
            }

            // position close check
            // closee if stop levels have been hit
            if (((state._pos > 0) && (state._bar.low <= state._stop)) ||
                ((state._pos < 0) && (state._bar.high >= state._stop))) 
            {
                state._tgt_pos = 0;
                state._stop = 0;
                state._h = 0;
                state._l = 0;
                state._ic = 0;
                state._ds = 0;
                logInfo("%s(%s) clearing position: %s", m_name.c_str(), param._bcfg->toString().c_str(), state.toString(false).c_str());
            } else {
                // update stop level and speed
                if (state._pos > 0) {
                    state._stop += (state._ic * (std::max(state._bar.high, state._h) - state._stop));
                    // if new high - update speed and reference
                    if (state._bar.high > state._h) {
                        state._h = state._bar.high;
                        state._ic = std::min(state._ic + param._sar_ic, param._sar_cl);
                    }
                } else {
                    // position short, update stop
                    state._stop += (state._ic * (std::min(state._bar.low, state._l) - state._stop));
                    if (state._bar.low < state._l) {
                        state._l = state._bar.low;
                        state._ic = std::min(state._ic + param._sar_ic, param._sar_cl);
                    }
                }
            }
        } else {
            // no position - check if signal triggered
            if (param._signal_tf) {
                // new trade if threshold crossed
                int x_trade = 0;
                if (state._bar.high > param._thres_h) {
                    x_trade = 1;
                } else if (state._bar.low < param._thres_l) {
                    x_trade = -1;
                }

                if ((x_trade != 0) && (cur_utc < param._end_utc) &&
                    !( (state._pos_td == param._trade_day) &&
                       (state._side == x_trade) ) )
                {
                    state._side = x_trade;
                    state._h = state._bar.high;
                    state._l = state._bar.low;
                    state._ds = param._sar_ds;
                    state._stop = state._bar.close - x_trade * state._ds;
                    state._ic =  param._sar_ic;
                    state._tgt_pos = x_trade * param._pos_n;
                    logInfo("%s(%s) taking position: %s", m_name.c_str(), param._bcfg->toString().c_str(), state.toString(false).c_str());
                }
            }
        }
    }

    bool IDBO_TF::initState(time_t cur_sec) {
        m_state.clear();
        // init from m_param
        for (const auto& sp: m_param) {
            int symid = sp.first;
            std::string persist_file = persistFileName(symid);
            auto state (std::make_shared<State>(symid, persist_file));
            setupTriggerTime(*sp.second, *state, cur_sec);
            m_state[sp.first] = state;
        }
        return true;
    }

    IDBO_TF::Param::Param(const std::string& mts_symbol, const utils::ConfigureReader& cfg, time_t cur_utc) :
    _mts_symbol(mts_symbol),
    _symid(-1),
    _last_loaded(0) {
        int sh = cfg.get<int>("time_from")/100;
        int sm = cfg.get<int>("time_from")%100;
        int eh = cfg.get<int>("time_to")/100;
        int em = cfg.get<int>("time_to")%100;
        std::string day = utils::TimeUtil::tradingDay(
                cur_utc, 
                sh,  // start_hour
                sm,  // start_min
                eh,  // end_hour
                em,  // end_min
                0,   // day_offset
                2    // current or next trading day
            );
        _trade_day = std::stoi(day);
        time_t day_utc = (time_t) utils::TimeUtil::string_to_frac_UTC(day.c_str(), 0, "%Y%m%d");
        _start_utc = day_utc + ((sh*60) + sm)*60;
        _end_utc = day_utc + ((eh*60) + em)*60;

        _bcfg = std::make_shared<md::BookConfig>(mts_symbol, "L1");
        _signal_tf = cfg.get<int>("signal_tf");
        _thres_h = cfg.get<double>("thres_h");
        _thres_l = cfg.get<double>("thres_l");
        _sar_ds = cfg.get<double>("sar_ds");
        _sar_ic = cfg.get<double>("sar_ic");
        _sar_cl = cfg.get<double>("sar_cl");
        _pos_n = int(cfg.get<double>("pos_n") + 0.5);
        _inactive_bars = cfg.get<int>("inactive_bars");
        _last_loaded = cur_utc;
        logInfo("Parameter loaded: %s", 
                toString(false).c_str());
    }

    IDBO_TF::Param::~Param() {
        if (_symid>=0) {
            _symid = -1;
        }
        _last_loaded = 0;
    }

    std::string IDBO_TF::persistFileName(int symid) const {
        // IDBO_TF_mts_symbol_state.csv
        const auto iter = m_param.find(symid);
        const auto& sym = iter->second->_mts_symbol;
        const std::string path = plcc_getString("RecoveryPath") + "/strat";
        return path + "/" +m_name + "/IDBO_TF_" + sym + "_state.csv";
    }

    IDBO_TF::State::State(int symid, const std::string& persist_file)
    : _symid(symid),
      _pos(0),
      _tgt_pos(0),
      _stop(0),
      _h(0),
      _l(0),
      _ic(0),
      _ds(0),
      _side(0),
      _last_updated(0),
      _pos_td(0),
      _persist_file(persist_file) {
          retrieve();
      }

    bool IDBO_TF::setupTriggerTime(const Param& param, State& state, time_t cur_sec) {
        // setup trigger time based on barsec and a random offset
        auto& trigger_time(state._trigger_time);
        trigger_time.clear();
        state._next_trigger_idx = 0;
        const int bs = IDBO_TF::Barsec;
        cur_sec = (cur_sec/bs+1)*bs;

        if (param._start_utc/bs*bs != param._start_utc) {
            logError("%s start time not a multiple of bar period %d (%s)!", m_name.c_str(),
                    (int)param._start_utc,
                    param._bcfg->toString().c_str());
            return false;
        }
        if (cur_sec < param._start_utc) {
            logInfo("%s start before the start time %s", 
                    m_name.c_str(), utils::TimeUtil::frac_UTC_to_string(param._start_utc, 0).c_str());
            cur_sec = param._start_utc;
        } else if (cur_sec > param._end_utc) {
            logError("%s start after end time %s!",
                    m_name.c_str(), utils::TimeUtil::frac_UTC_to_string(param._end_utc, 0).c_str());
            return false;
        }

        trigger_time.push_back(cur_sec);
        cur_sec += bs;
        while (cur_sec <= param._end_utc) {
            trigger_time.push_back(cur_sec);
            cur_sec += bs;
        }
        return true;
    }

    std::string IDBO_TF::onDump() const {
        // dump the state book/bar/curpos
        // dump the param coef/barsec/st/et/maxpos
        std::string ret = "Dump: (strategy weigght = " + std::to_string(m_weight) + ")\n";
        for (const auto& sp: m_state) {
            const auto& state = *sp.second;
            const auto iter = m_param.find(sp.first);
            const auto& param(*(iter->second));
            ret += std::string("\n******\n") + param.toString(true) + "\nState: " + state.toString(true) + "\n.";
        }
        return ret.substr(0, ret.size()-1);
    }

    void IDBO_TF::State::persist() const {
        // don't persist if _pos and _tgt_pos are both zero
        if ((_pos == 0) && (_tgt_pos == 0)) {
            return;
        }

        // format of last_updated, pos, tgt_pos, stop, h, l, ic, ds, side, pos_td
        FILE* fp = fopen(_persist_file.c_str(), "a");
        fprintf(fp, "%d, %d, %d, %lf, %lf, %lf, %lf, %lf, %d, %d\n", 
                (int) _last_updated, (int) _pos, (int) _tgt_pos,
                _stop, _h, _l, _ic, _ds,
                _side, _pos_td);
        fclose(fp);
    }

    void IDBO_TF::State::retrieve() {
        const auto& file_token (utils::CSVUtil::read_file(_persist_file));
        if (file_token.size() > 0) {
            const auto& tk = file_token[file_token.size()-1];
            _tgt_pos = std::stoi(tk[2]);
            if (_tgt_pos == 0) {
                logInfo("IDBO_TF state cleared on zero tgt_pos from %s", _persist_file.c_str());
                return;
            }
            _last_updated = std::stoi(tk[0]);
            _pos = std::stoi(tk[1]);
            _stop = std::stod(tk[3]);
            _h = std::stod(tk[4]);
            _l = std::stod(tk[5]);
            _ic = std::stod(tk[6]);
            _ds = std::stod(tk[7]);
            _side = std::stoi(tk[8]);
            _pos_td = std::stoi(tk[9]);
            logInfo("IDBO_TF retrieved state file %s, current state: %s", 
                    _persist_file.c_str(), 
                    toString(true).c_str());
        } else {
            logInfo("IDBO_TF cannot retrieve state file %s, start cleared", 
                    _persist_file.c_str());
        }
    }

    std::string IDBO_TF::State::toString(bool verbose) const {
        char buf[512];
        size_t bytes = snprintf(buf, sizeof(buf), 
            "cur_pos=%lld, tgt_pos=%lld",
            (long long) _pos, (long long) _tgt_pos);
        if (verbose) {
            bytes+= snprintf(buf+bytes, sizeof(buf)-bytes,
            "\nsnap=%s\nbar=%s\n"
            "stop=%.8lf, h=%.8lf, l=%.8lf, "
            "ic=%.8lf, ds=%.8lf, persist=%s",
            _bookL1.toString().c_str(),
            _bar.toString().c_str(), 
            _stop, _h, _l, _ic, _ds, _persist_file.c_str());
        }
        return std::string(buf);
    }

    std::string IDBO_TF::Param::toString(bool verbose) const {
        char buf[1024];
        size_t bytes = snprintf(buf, sizeof(buf), 
                "symbol=%s, signal_tf=%d, pos_size=%d",
                _bcfg->toString().c_str(), (int) _signal_tf, (int)_pos_n);
        if (verbose) {
            bytes+= snprintf(buf+bytes, sizeof(buf)-bytes,
                "\nparameters: [thres_h=%f,thres_l=%f,sar_ds=%f,sar_ic=%f,sar_cl=%f]\n"
                "start_utc=%s, end_utc=%s, last_loaded=%s",
                _thres_h, _thres_l, _sar_ds, _sar_ic, _sar_cl,
                utils::TimeUtil::frac_UTC_to_string(_start_utc, 0).c_str(),
                utils::TimeUtil::frac_UTC_to_string(_end_utc, 0).c_str(),
                utils::TimeUtil::frac_UTC_to_string(_last_loaded, 0).c_str());
        }
        return std::string(buf);
    }

    std::string IDBO_TF::cfgFile() const {
        return m_cfg;
    }
}
