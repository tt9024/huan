#pragma once

#include "AlgoBase.h"
#include <cmath>

namespace algo {

    class IDBO_TF : public AlgoBase {
    public:

        // create the algo in stopped state, config not read
        IDBO_TF(const std::string& name, const std::string& cfg, pm::FloorBase::ChannelType& channel, uint64_t cur_micro);
        IDBO_TF(const std::string& name, const std::string& cfg, pm::FloorBase::ChannelType& channel); // created without parameter and state being loaded, need to call onReload() explicitly before start

        ~IDBO_TF();

        void onStop(uint64_t cur_micro) override; 
        void onStart(uint64_t cur_micro) override; 

        // read config, remove all subscriptions, add all subscriptions
        void onReload(uint64_t cur_micro, const std::string& config_file = "") override;

        // check to see what to do for algo
        void onOneSecond(uint64_t cur_micro) override;

        std::string onDump() const override; 
        std::string cfgFile() const override;
        const std::string m_name;

    protected:
        enum { Barsec = 300 } ;
        struct Param {
            explicit Param(const std::string& mts_symbol, const utils::ConfigureReader& cfg, time_t cur_utc);
            ~Param();

            std::shared_ptr<md::BookConfig> _bcfg;
            std::string _mts_symbol;
            int _symid;
            int _trade_day; // YYYYMMDD
            time_t _start_utc;
            time_t _end_utc;
            int64_t _max_pos;

            // live header
            int _signal_tf;
            double _thres_h;
            double _thres_l;
            double _sar_ds;
            double _sar_ic;
            double _sar_cl;
            int _pos_n; // pos_n * multiplier
            int _inactive_bars;

            time_t _last_loaded;
            std::string toString(bool verbose) const;
        };

        struct State {
            md::BookDepot _bookL1;
            md::BarPrice _bar;
            int _symid;
            std::vector<time_t> _trigger_time;
            int _next_trigger_idx;

            int32_t _pos;     // current position
            int32_t _tgt_pos; // desired position
            double _stop;
            double _h;
            double _l;
            double _ic;
            double _ds;
            int _side;
            time_t _last_updated;
            int _pos_td;
            std::string _persist_file;

            explicit State(int symid, const std::string& persist_file);
            std::string toString(bool verbose) const;
            void persist() const;
            void retrieve();
        };

        // update state with new market data
        bool initState(time_t cur_utc);
        bool updateState(const Param& param, State& state, uint64_t cur_micro);
        void setPosition(State& state, time_t cur_utc);

        // setup trigger times
        bool setupTriggerTime(const Param& param, State& state, time_t cur_utc);

        // signal function
        void idbo(const Param& para, State& state, uint64_t cur_micro);

        std::string m_cfg;
        std::map<int, std::shared_ptr<Param> > m_param;
        std::map<int, std::shared_ptr<State> > m_state;

        std::string persistFileName(int symid) const;

        double m_weight;

    };
}

