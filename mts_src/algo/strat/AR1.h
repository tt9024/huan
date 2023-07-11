#pragma once

#include "AlgoBase.h"
#include <cmath>

namespace algo {

    class AR1 : public AlgoBase {
    public:

        // create the algo in stopped state, config not read
        AR1(const std::string& name, const std::string& cfg, pm::FloorBase::ChannelType& channel, uint64_t cur_micro);
        ~AR1();

        void onStop(uint64_t cur_micro) override; 
        void onStart(uint64_t cur_micro) override; 

        // read config, remove all subscriptions, add all subscriptions
        void onReload(uint64_t cur_micro, const std::string& config_file = "") override;

        // check to see what to do for algo
        void onOneSecond(uint64_t cur_micro) override;

        std::string onDump() const override; 
        std::string cfgFile() const override;

    protected:
        struct AR1Param {
            explicit AR1Param(const std::string& name, const std::string& cfg, time_t cur_utc=0);
            std::shared_ptr<md::BookConfig> _bcfg;
            int _barsec;
            std::vector<double> _A1Coef;
            size_t _lookback;
            time_t _start_utc;
            time_t _end_utc;
            int64_t _max_pos;
            time_t _last_loaded;
            std::string _name;


            std::string toString() const;
        };

        struct AR1State {
            md::BookDepot _bookL1;
            std::vector<std::shared_ptr<md::BarPrice> > _barHist;
            int64_t _curPos;
            time_t _last_updated;

            std::string toString() const;
        };

        struct AR1Forecast {
            double _logRet;
            double _variance;
        };

        // update state with new market data
        bool initState(uint64_t cur_micro);
        bool updateState(uint64_t cur_micro);

        // perform forecast based on current state
        bool forecast(AR1Forecast& fcst);

        // position optimization code
        int pop(const AR1Forecast& fcst);

        // setup trigger times
        bool setupTriggerTime(uint64_t cur_micro);

        std::string m_cfg;
        std::shared_ptr<AR1Param> m_param;
        AR1State m_state;
        std::vector<time_t> m_trigger_time;
        size_t m_next_trigger_idx;
        int m_symid;
    };
}
