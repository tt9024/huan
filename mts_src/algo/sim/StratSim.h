#include "md_bar.h"
#include "AlgoThreadMock.h"
#include <fstream>
#include <sstream>

namespace algo {

/*
 * publishes L1 BBO based on MTS History File (1 second bar), and
 * writes snap queue and bar data. 
 * The BBO is inferred from the close price at the bar time.
 * BBO size is a constant.
 */

class BarPub {
public:
    enum { BBO_SIZE = 10 , BAR_PERIOD = 1 };

    BarPub(const char* tradable, const char* mts_bar, time_t cur_second)
    // tradable is key of symbol map
    // mts_bar is the csv file of the 1S bar
    : _bar_file(mts_bar),
      _bcfg(tradable, "L1"), 
      _tick_size(_bcfg.getTradableInfo()->_tick_size),
      _bq(_bcfg, false), 
      _bw(_bq.theWriter()),
      _all_bar(std::move(loadMTSBar(mts_bar))),  // holds all bars from history file
      _bar(_bcfg, cur_second), // needs to be initialized after the first snap
      _next_ix(0), _bpx(0), _apx(0)
    {
        logInfo("Started BarPub for %s, bar file: %s", tradable, mts_bar);
    }

    // writes the snap and bar based on cur_sec
    void onOneSecond(time_t cur_sec) {
        if (__builtin_expect(_next_ix>=_all_bar.size(), 0)) {
            // done for the day
            return;
        }
        // gets the next bar
        const auto& b(_all_bar[_next_ix]);

        // we are at the next bar time
        if (__builtin_expect(cur_sec == b.bar_time, 1)) {
            pub(b, cur_sec);
            ++_next_ix;
            return;
        }

        // next bar time is earlier than the cur_sec
        if (cur_sec > b.bar_time) {
            // skip to the latest bar for cur_sec
            while ((cur_sec > _all_bar[_next_ix].bar_time) && 
                   (_next_ix < _all_bar.size()-1))  {
                ++_next_ix;
            }
            if (cur_sec < _all_bar[_next_ix].bar_time) {
                --_next_ix;
            }
            pub(_all_bar[_next_ix], cur_sec);
            ++_next_ix;
            return;
        }

        if (__builtin_expect(cur_sec == b.bar_time - BAR_PERIOD, 0)) {
            // publish open price of that bar
            pub_snap(b.open, cur_sec);

            // note in case this is the open time
            // the barWriter.onOneSecond() initializes
            // the open price without write.
            _bar.onOneSecond(cur_sec);
            return;
        }

        // wait for the next bar time
        return;
    }

    static const int MTSBarSecond = 1;
private:
    const std::string _bar_file;
    const md::BookConfig _bcfg;
    const double _tick_size;
    md::BookQ<utils::ShmCircularBuffer> _bq;
    md::BookQ<utils::ShmCircularBuffer>::Writer& _bw;
    const std::vector<md::BarPrice> _all_bar;
    md::BarWriter _bar;
    uint64_t _next_ix;
    double _bpx, _apx;

    std::vector<md::BarPrice> loadMTSBar(const char* mts_bar) {
        std::vector<md::BarPrice> all_bar;
        std::ifstream bf(mts_bar);
        std::string line;
        while (std::getline(bf, line)) {
            all_bar.emplace_back(line);
        }
        if (all_bar.size() < 2) {
            throw std::runtime_error(std::string("too few bars in ") + mts_bar);
        }

        // publish the open price before the bar writer is loaded
        // note the over-night return is visible if strategy request
        // previous day's bar
        pub_snap(all_bar[0].open, all_bar[0].bar_time-MTSBarSecond);
        return all_bar;
    }

    void pub(const md::BarPrice& bar, time_t cur_sec) {
        // publish the latest price to snap and update bar
        pub_snap(bar.close, bar.bar_time);
        pub_trade(bar.bvol, bar.svol, bar.last_price, bar.close);
        _bar.onOneSecond(cur_sec);
    }

    void pub_snap(double px, time_t bar_time) {
        // publish snap price
        double spd = _tick_size/2;

        // see if the price is at full tick or half tick
        double px0 = normalizePrice(px);
        if (std::abs(px0-px) <  _tick_size/10) {
            // mid price at a full tick, set spread as 2 ticks
            spd = _tick_size;
        }
        _bpx = normalizePrice(px - spd);
        _apx = normalizePrice(px + spd);
        _bw.updBBO(_bpx, BBO_SIZE, _apx, BBO_SIZE, bar_time*1000000);
    }

    void pub_trade(int32_t bvol, int32_t svol, double trdpx, double midpx) {
        if (trdpx > midpx) {
            pub_trd(svol, _bpx);
            pub_trd(bvol, trdpx);
        } else {
            pub_trd(bvol, _apx);
            pub_trd(svol, trdpx);
        }
    }

    void pub_trd(int32_t vol, double px) {
        if (vol != 0) {
            _bw.updTrade(px, vol);
        }
    }

    double normalizePrice(double px) {
        double px0 = std::abs(px);
        int ticks = int(px0/_tick_size+1e-2);
        px0 = _tick_size*ticks;
        return px>0? px0:-px0;
    }

};

/*
 * Set up AlgoThreadMock and BarPub with main.cfg
 */

class StratSim {
public:
    explicit StratSim(const char* trade_day, const char* cfg_file = NULL)
    : m_should_run(false) 
    {
        // performs the following taasks
        // 1. initialize the trading day and timer
        // 2. read the configuration and create market data from MtsBar
        // 3. create strategy from Strat
        // 4. initialize all strategies
        if ( (!cfg_file) || (!cfg_file[0]) ) {
            cfg_file = "config/main.cfg";
        }
        _main_cfg = std::string(cfg_file);
        const utils::ConfigureReader cr (cfg_file);

        // read the trading day
        _sim_name = cr.get<std::string>("SimName");
        _trade_day = cr.get<std::string>("SimDay");
        if (_trade_day != std::string(trade_day)) {
            throw std::runtime_error(std::string("trade day ") + trade_day + std::string(" mismatch with config file ") + cfg_file);
        }

        // Assuming the time has already been set to start of trading day
        _cur_micro = utils::TimeUtil::cur_micro();
        _end_micro = _cur_micro + 23*3600*1000000ULL; //run for 23 hours

        // read the market data inputs, given in format of
        // MtsBar = [ [ contract, bar_file ] , ... ]
        size_t symbols = cr.getReader("MtsBar").arraySize();
        char qstr[64];
        for (size_t i=0; i<symbols; ++i) {
            snprintf(qstr, sizeof(qstr), "MtsBar[%d][0]",(int)i);
            std::string tradable = cr.get<std::string>(qstr);
            snprintf(qstr, sizeof(qstr), "MtsBar[%d][1]",(int)i);
            std::string barfile = cr.get<std::string>(qstr);
            _pub.emplace_back(std::make_shared<BarPub>(tradable.c_str(), barfile.c_str(), _cur_micro/1000000));
        }

        // create AlgoThreadMock and kick off
        const auto& scfg = cr.get<std::string>("Strat");
        _sim = std::make_shared<algo::AlgoThreadMock>(_sim_name, scfg);
    }

    // Simulates AlgoThread's run()
    void run() {
        m_should_run = true;
        _sim->startAll();

        logInfo("Starting simulation %s on %s", _sim_name.c_str(), _trade_day.c_str());
        while (m_should_run && 
               (_cur_micro <= _end_micro)) {
            time_t cur_sec = _cur_micro/1000000;
            // bar update
            for (auto& pub : _pub) {
                pub->onOneSecond(cur_sec);
            }
            _sim->runOneSecond(_cur_micro);
            _cur_micro += 1000000;
            utils::TimeUtil::set_cur_time_micro(_cur_micro);
        }
        _sim->eodPosition();
        logInfo("Simulation done for day %s", _trade_day.c_str());
    }

    // ctrl-C from user, stop the run
    void stop() {
        _sim->stopAll();
        m_should_run = false;
        logInfo("Stopping simulation %s on %s", _sim_name.c_str(), _trade_day.c_str());
    }

    volatile bool m_should_run;
protected:
    std::string _main_cfg;
    std::string _sim_name, _trade_day;

    std::vector< std::shared_ptr<BarPub> > _pub;
    std::shared_ptr<AlgoThreadMock> _sim;

    uint64_t _cur_micro;
    uint64_t _end_micro;
};

}
