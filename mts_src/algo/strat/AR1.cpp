#include "AR1.h"
#include <cmath>
#include "RiskMonitor.h"
#include <stdexcept>

namespace algo {

    AR1::AR1 (const std::string& name, const std::string& cfg, pm::FloorBase::ChannelType& channel, uint64_t cur_micro)
    : AlgoBase(name, channel), 
      m_cfg(cfg),
      m_next_trigger_idx(0),
      m_symid(-1)
    {
        onReload(cur_micro, m_cfg);
        logInfo("%s created with cfg %s", m_name.c_str(), m_cfg.c_str());
    }

    AR1::~AR1() {
        logInfo("%s destructed", m_name.c_str());
    }

    void AR1::onReload(uint64_t cur_micro, const std::string& config_file) {
        // try reading the new parameter file
        auto m_param_new = m_param;
        try {
            m_param_new = std::make_shared<AR1Param>(m_name, config_file, (time_t)(cur_micro/1000000ULL));
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

        // replace the old parameter
        m_param = m_param_new;

        // subscribe market data for the A1 symbol
        if (m_symid >= 0) {
            removeSymbol(m_symid);
        }
        const auto& bc(m_param->_bcfg);
        m_symid = addSymbol(bc->venue, bc->symbol, bc->type, m_param->_barsec);

        // get historical bar so far
        initState(cur_micro);

        // restart the strategy if needed
        if (need_restart) {
            onStart(cur_micro);
        }
    }

    void AR1::onStart(uint64_t cur_micro) {
        // create the trigger time and update state
        if (!setupTriggerTime(cur_micro))
        {
            logError("%s failed to setup trigger time, "
                    "not started!", m_name.c_str());
            onStop(cur_micro);
            setShouldRun(false);
            return;
        }
        setShouldRun(true);
        updateState(cur_micro);
    }

    void AR1::onStop(uint64_t) {}

    void AR1::onOneSecond(uint64_t cur_micro) {

        // don't run if trigger not due yet
        const auto trigger_time = m_trigger_time[m_next_trigger_idx];
        if ((time_t)(cur_micro/1000000ULL) < trigger_time) {
            return;
        }
        if (++m_next_trigger_idx >= m_trigger_time.size()) {
            logInfo("%s finished day, stopped", m_name.c_str());
            onStop(cur_micro);
            setShouldRun(false);
            return;
        }

        // update the state
        updateState(cur_micro);

        // forecast
        AR1Forecast fcst;
        forecast(fcst);
        int64_t tgt_position = pop(fcst);

        // setPosition
        if (tgt_position != m_state._curPos) {
            double tgt_px = (tgt_position > m_state._curPos? 
                    (m_state._bookL1.getAsk()) :
                    (m_state._bookL1.getBid())
                    );
            if (!setPositionLimit(m_symid, tgt_position, tgt_px)) {
                logError("%s failed to set position to %lld from %lld", 
                        m_name.c_str(), (long long) tgt_position, 
                        (long long) m_state._curPos);
            }
        }
    }

    bool AR1::updateState(uint64_t cur_micro) {
        // update snap
        getSnap(m_symid, m_state._bookL1);
        getNextBar(m_symid, m_state._last_updated, m_state._barHist);
        size_t sz = m_state._barHist.size();
        if (sz<1) {
            logInfo("%s not enough bars received yet", m_name.c_str());
            return false;
        }
        m_state._last_updated = m_state._barHist[sz-1]->bar_time;

        int64_t qty_done, qty_open;
        if (!getPosition(m_symid, qty_done, qty_open)) {
            logError("%s failed to get position!", m_name.c_str());
            return false;
        }
        m_state._curPos = qty_done + qty_open;
        return true;
    }

    bool AR1::initState(uint64_t cur_micro) {
        // initialize the bar history
        m_state._barHist.clear();
        getHistBar(m_symid, m_param->_lookback, m_state._barHist);
        size_t sz = m_state._barHist.size();
        if (sz==0) {
            m_state._last_updated = (time_t)(cur_micro/1000000ULL)/m_param->_barsec*m_param->_barsec;
            logError("AR1 failed to get any history data!");
        } else {
            m_state._last_updated = m_state._barHist[sz-1]->bar_time;
        }
        m_state._curPos = 0;
        return true;
    }

    AR1::AR1Param::AR1Param(const std::string& name, const std::string& cfg, time_t cur_utc)
    : _name(name) {
        const auto& vc = utils::ConfigureReader(cfg.c_str());
        if (vc.getReader("symbols").arraySize() != (size_t) 1) {
            logError("AR1 configuration error: symbols size not 1!");
            throw std::invalid_argument("AR1 symbols size not 1!");
        }
        // expect to be in the format of
        /*
        symbols = [ 
            { 
              mts_symbol = SPX_N1
              # venue is needed because tp uses symbol.xml
              # and is different with xml_v2's venue
              # here the venue should be from symbol.xml
              venue = CME
              level = L1
              bar_sec = 5
            }
          ]
        */

        const auto& vl = vc.getReader("symbols[0]");
        const std::string mts_sym = vl.get<std::string>("mts_symbol");
        const auto* ti = utils::SymbolMapReader::get().getByMtsSymbol(mts_sym);
        const std::string& tradable (ti->_tradable);
        const std::string& symbol  (ti->_symbol);

        _bcfg = std::make_shared<md::BookConfig>(ti->_venue, tradable, vl.get<std::string>("level"));
        _barsec = vl.get<int>("bar_sec");

        // getting the start/stop time for current (or next) trading day
        auto start_str = utils::CSVUtil::read_line(vc.get<std::string>("StartTime"), ':');
        auto end_str   = utils::CSVUtil::read_line(vc.get<std::string>("EndTime"), ':');
        int sh = std::stoi(start_str[0]);
        int sm = std::stoi(start_str[1]);
        int eh = std::stoi(end_str[0]);
        int em = std::stoi(end_str[1]);

        std::string day = utils::TimeUtil::tradingDay(
                cur_utc, 
                sh,  // start_hour
                sm,  // start_min
                eh,  // end_hour
                em,  // end_min
                0,   // day_offset
                2    // current or next trading day
            );
        time_t day_utc = (time_t) utils::TimeUtil::string_to_frac_UTC(day.c_str(), 0, "%Y%m%d");
        _start_utc = day_utc + ((sh*60) + sm)*60;
        _end_utc = day_utc + ((eh*60) + em)*60;


        // getting the max position from the risk config, 
        // this could throw if the symbol is not defined
        _max_pos = (int64_t) pm::risk::Monitor::get().maxPosition(_name, symbol);

        auto cl = vc.getArr<std::string>("Coef");
        for (const auto& c : cl) {
            _A1Coef.push_back(std::stod(c));
        }
        _lookback = _A1Coef.size();
        _last_loaded = cur_utc;

        logInfo("Parameter loaded: %s", 
                toString().c_str());
    }

    bool AR1::setupTriggerTime(uint64_t cur_micro) {
        // setup trigger time based on barsec and a random offset
        m_trigger_time.clear();
        m_next_trigger_idx = 0;
        time_t cur_sec = (time_t)(cur_micro/1000000ULL) + 2;

        if (cur_sec < m_param->_start_utc) {
            logInfo("%s start before the start time %s", 
                    m_name.c_str(), utils::TimeUtil::frac_UTC_to_string(m_param->_start_utc, 0).c_str());
            cur_sec = m_param->_start_utc;
        } else if (cur_sec > m_param->_end_utc) {
            logError("%s start after end time %s!",
                    m_name.c_str(), utils::TimeUtil::frac_UTC_to_string(m_param->_end_utc, 0).c_str());
            return false;
        }

        m_trigger_time.push_back(cur_sec);

        // write trigger time on each bar, with a random shift
        int rand = (int) (cur_micro%1000);

        const int bs = m_param->_barsec;
        cur_sec = (cur_sec/bs+1) * bs;
        while (cur_sec < m_param->_end_utc) {
            m_trigger_time.push_back((time_t)(cur_sec + rand%bs));
            cur_sec += bs;
            rand = (rand*rand)%1000;
        }
        m_trigger_time.push_back(m_param->_end_utc);
        return true;
    }

    bool AR1::forecast(AR1Forecast& fcst) {
        const size_t sz = m_state._barHist.size();
        if (sz < m_param->_lookback) {
            logError("not enough hist to calculate");
            return false;
        }
        double f0 = 0;
        for (size_t i=0 ; i<m_param->_lookback; ++i) {
            double lr = std::log
                ( 
                    (double)m_state._barHist[sz-1-i]->open /
                    (double)m_state._barHist[sz-1-i]->close
                );
            f0 += (m_param->_A1Coef[i] * lr);
        }
        fcst._logRet = f0;
        
        // dump the ru

        char buf[512];
        size_t bytes = 0;
        for (size_t i = 0; i<  m_param->_lookback; ++i) {
            bytes += snprintf(buf+bytes, sizeof(buf)-bytes, " (%lu, %lf, %lf) ", 
                    (unsigned long)m_state._barHist[sz-i-1]->bar_time, 
                                   m_state._barHist[sz-i-1]->close, 
                                   std::log
                                   (
                                       (double)m_state._barHist[sz-1-i]->open /
                                       (double)m_state._barHist[sz-1-i]->close
                                   )
                              );
        }
        logInfo("AR1 got bar: %s", buf);
        logInfo("AR1 got forecast: %lf\n", fcst._logRet);

        return true;
    }

    int AR1::pop(const AR1Forecast& fcst) {
        // TODO - to be implemented
        if (fcst._logRet > 0)
            return 1;
        else 
            return -1;
    }

    std::string AR1::onDump() const {
        // dump the state book/bar/curpos
        // dump the param coef/barsec/st/et/maxpos
        return "Param: " + m_param->toString() + " State: " + m_state.toString();
    }

    std::string AR1::AR1State::toString() const {
        char buf[512];
        snprintf(buf, sizeof(buf), 
            "pos=%lld, snap=%s, bar=%s, updated=%s",
            (long long) _curPos,
            _bookL1.toString().c_str(),
            (_barHist.size()>0?
               _barHist[_barHist.size()-1]->toString().c_str():""),
            utils::TimeUtil::frac_UTC_to_string(_last_updated, 0).c_str());
        return std::string(buf);
    }

    std::string AR1::AR1Param::toString() const {
        char buf[512];
        snprintf(buf, sizeof(buf), 
                "symbol=%s, barsec=%d, lookback=%d, max_pos=%d,"
                "start_utc=%s, end_utc=%s, last_loaded=%s, name=%s",
                _bcfg->toString().c_str(),
                (int)_barsec,
                (int)_lookback,
                (int)_max_pos,
                utils::TimeUtil::frac_UTC_to_string(_start_utc, 0).c_str(),
                utils::TimeUtil::frac_UTC_to_string(_end_utc, 0).c_str(),
                utils::TimeUtil::frac_UTC_to_string(_last_loaded, 0).c_str(),
                _name.c_str()
            );
        return std::string(buf);
    }

    std::string AR1::cfgFile() const {
        return m_cfg;
    }
}
