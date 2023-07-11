#pragma once

#include <set>
#include <vector>
#include <memory>
#include <map>

#include "FloorBase.h"
#include "md_snap.h"
#include "md_bar.h"

namespace algo {
    class AlgoBase {
    public:
        struct SymbolInfo {
            const md::BookConfig _bcfg;
            const int _barsec;
            std::shared_ptr<md::BookQType> _bq;
            std::shared_ptr<md::BookQReader> _snap_reader;
            std::shared_ptr<md::BarReader> _bar_reader;

            SymbolInfo( md::BookConfig&& bcfg, int barsec);
            std::string toString() const;
        };

        // create the algo in stopped state, config not read, channel is
        // out-going to the floor, and is share by strategies in AlgoThread
        AlgoBase(const std::string& name, pm::FloorBase::ChannelType& channel);
        virtual ~AlgoBase();

        // ==== functions used by Algo =====
        // getting current position 
        virtual bool getPosition(int symid, int64_t& qty_done, int64_t& qty_open);

        // set desired position 
        // market: achieve tgt position by market order
        // limit:  achieve by limit order
        // passive, achieve tgt position passively before tgt_utc
        // Refer to FloorBase.h for detail
        bool setPositionMarket(int symid, int64_t tgt_qty);
        bool setPositionLimit(int symid, int64_t tgt_qty, double tgt_px);
        bool setPositionPassive(int symid, int64_t tgt_qty, double tgt_px, time_t tgt_utc);
        bool setPositionTWAP(int symid, int64_t tgt_qty, time_t tgt_utc);
        bool setPositionTWAP2(int symid, int64_t tgt_qty, time_t tgt_utc);

        // set the position of symid to 0
        bool coverPosition(int symid);

        // utilities for for getting market data

        // get the latest bar
        bool getBar(int symid, md::BarPrice& bp);

        // get the latest snap price
        bool getSnap(int symid, md::BookDepot& sp);

        // The following two gets historical bars. 
        // get a fixed number of latest bars
        bool getHistBar(int symid, int numBars, std::vector<std::shared_ptr<md::BarPrice> >& bp);

        // get all bars since a given time (not inclusive), this is called
        // when a gap is detected on the latest bar, catching up. 
        int getNextBar(int symid, time_t since, std::vector<std::shared_ptr<md::BarPrice> >& bp);

        // for register symbols
        int addSymbol(const std::string& venue, const std::string& symbol, 
                      const std::string& snap_level, int barsec);
        void removeSymbol(int symid);
        void removeAllSymbol();

        // utilities
        void setShouldRun(bool should_run) { m_should_run = should_run; };
        bool shouldRun() const { return m_should_run;};
        std::string toString(bool dump_state = true) const;

        // ===== functions provided by Algo =====
        virtual void onStop(uint64_t cur_micro) = 0;
        virtual void onStart(uint64_t cur_micro) = 0;

        // read config, remove all subscriptions, add all subscriptions
        virtual void onReload(uint64_t cur_micro, const std::string& config_file = "") = 0;

        // check to see what to do for algo
        virtual void onOneSecond(uint64_t cur_micro) = 0;

        // dump current parameters and states
        virtual std::string onDump() const = 0;

        // return current config file name
        virtual std::string cfgFile() const = 0;

        // return a copy of SymbolInfo using symid
        SymbolInfo getSymbolInfo(int symid) const;

        // return name of this strategy
        std::string getName() const { return m_name; };

    protected:
        const std::string m_name;
        pm::FloorBase::ChannelType& m_channel;
        std::string m_cfg;
        volatile bool m_should_run;
        std::vector< std::shared_ptr<SymbolInfo> > m_symbols;
        virtual bool setPosition(const pm::FloorBase::PositionInstruction& pi);
    };
}
