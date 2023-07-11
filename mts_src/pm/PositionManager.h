#pragma once

#include "PositionData.h"
#include "PositionServer.h"
#include "FloorBase.h"

#include <memory>
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace pm {
    using PositionMap = std::map<std::string, std::map<std::string, std::shared_ptr<IntraDayPosition> > >;

    class PositionManager {
    public:
        explicit PositionManager(const std::string& name, const std::string& recover_path = "");

        // Allow compiler to generate destructor
        ~PositionManager() = default;

        std::string getLoadUtc() const;
        // read the latest positions from eod_csv_file, 
        // return the utc of the latest persistence

        bool loadRecovery(const std::string& recovery_file, bool persist_fill=false, bool update_risk=false);
        // loads execution reports from recovery file

        bool reconcile(const std::string& recovery_file, std::string& diff_logs, bool adjust=false) ;
        // reconciles this position with previous eod position updated with recovery_file. 

        bool update(const ExecutionReport& er, bool persist_fill=false, bool update_risk=false, bool do_pause_notify=false);
        // updates the intraday position from either fill or open, return true if it's a new fill
        // update the risk monitor if update_risk is true. In case risk violation, do notify pause
        // to other FloorCPR instances
 
        void deleteOO(const char* clOrdId);
        // delete an open order, not from execution report

        void resetPnl();
        // reset daily pnl to be 0
        
        bool persist() const;
        // writes the intraday positions to EoDPosition
        
        int64_t getPosition(const std::string& algo, const std::string& symbol, 
                           double* vap=nullptr, double* pnl=nullptr, int64_t* oqty=nullptr, 
                           int64_t* out_qty=nullptr) const ;
        // get position, vap, pnl given algo/symbol
        // set ptr to nullptr if not interested. 
        // Note on out_qty, if any, already included in the return
        // If symbol is N1, return in+out position, not including the open qty
        // If symbol is not N1, return its position, not including open qty

        int64_t getPosition(const std::string& symbol, 
                           double* vap=nullptr, double* pnl=nullptr, int64_t* oqty=nullptr) const;
        // get total position, vap, pnl given a symbol
        
        int64_t getPosition_Market(const std::string* algo_p = nullptr, const std::string* mkt_p=nullptr, double* mtm_pnl = nullptr) const;
        // get aggregated position (including open qty) of algo and market. market, i.e. WTI, includes all symbols of _N.
        // set mkt_p or algo_p to nullptr to aggreate all, mtm_pnl can also be retrieved.

        std::vector<std::shared_ptr<const IntraDayPosition> > listPosition(const std::string* ptr_algo=nullptr, const std::string* ptr_symbol=nullptr) const;
        // list all positions given either algo or symbol.  If none were given, list all

        std::vector<std::shared_ptr<const IntraDayPosition> > listOutInPosition(const std::string& algo, const std::string& symbol) const;
        // list all in/out positions given algo and symbol.  
        // If symbol is XXX_N1 and XXX_N0 (out) position exists, 
        // then return vector [out, in]. If only in is found, return [in], 
        // Otherwise, return emtpy vector

        std::vector<std::shared_ptr<const OpenOrder> > listOO     (const std::string* ptr_algo=nullptr, const std::string* ptr_symbol=nullptr, const bool* ptr_side=nullptr) const;
        std::vector<std::shared_ptr<const OpenOrder> > listOO_FIFO(const std::string* ptr_algo=nullptr, const std::string* ptr_symbol=nullptr, const bool* ptr_side=nullptr) const;
        // list all open positions given either algo or symbol, if none were given, list all
        // if ptr_side is given, only list buy OO if true, sell OO otherwise.
        // The _FIFO version returns OO in decreasing order according to open time in micro-seconds

        std::shared_ptr<const OpenOrder> getOO(const std::string& clOrdId) const;
        // get an open order by clOrdId, return empty pointer if not found
        
        std::vector<std::pair<std::shared_ptr<const OpenOrder>, int64_t> > matchOO(
                const std::string& symbol, 
                int64_t qty, 
                const double* px = nullptr,
                const std::string* algo = nullptr) const;
        // get a list of matching open orders to fill internally
        // qty is sign significant, px is a limit price to be matched
        // return a vector of pair, each with a pointer to the open order
        // and the qty (with sign) to be matched.  Total matched quantity
        // should be less or equal to the given qty.
        // If algo is given, then all open orders with same algo name are matched,
        // otherwise, only crossed open orders are matched.

        double getPnl(const std::string* ptr_algo=nullptr, 
                      const std::string* ptr_symbol=nullptr) const;
        // get aggregated pnl given algo or symbol, if none given, aggregate all

        std::string diff(const PositionManager& pm) const;
        // for each position, check with pm

        void operator=(const PositionManager& pm);
        bool operator==(const PositionManager& pm) const;
        const std::string& getName() const { return m_name;};
        std::string getRecoveryPath() const { return m_recovery_path; };

        std::vector<FloorBase::PositionServerDetails> getPositionServerDetails() const;

        std::string toString(const std::string* ptr_algo=nullptr, const std::string* ptr_symbol=nullptr, bool summary=false) const;

        // const utilities 
        std::string eod_csv() const;
        std::string daily_eod_csv() const;
        std::string eod_csv_mtm() const;
        std::string daily_eod_csv_mtm() const;
        // get name of eod position csv file
        std::string fill_csv() const;
        // get the name of fill csv file
        bool isMLegUnderlyingUpdate(const ExecutionReport& er) const;
        // whether this er is for underlying of a mleg order that is currently open
        // used to determine if open order map should be updated by this er
        
        const utils::CSVUtil::FileTokens loadEoD_CSVLines(const std::string& eod_file, std::string* latest_day_ptr) const;

        bool haveThisFill(const ExecutionReport& er) const;

    protected:
        const std::string m_name;
        const std::string m_recovery_path;
        PositionMap m_algo_pos;
        PositionMap m_symbol_pos;
        const std::string m_load_second;
        uint64_t m_last_micro;
        std::unordered_set<std::string> m_fill_execid;

        // this is a map used to fast find OO from clOrdId, otherwise
        // we will have to go through all the idp to find the OO.
        std::unordered_map<std::string, std::shared_ptr<const OpenOrder> > m_oo_map;

        std::string loadEoD();
        // for each position, check with pm
        
        template<typename V, typename MIter>
        void addAllMapVal(V& vec, const MIter& iter1, const MIter& iter2) const {
            for (MIter iter=iter1; iter != iter2; ++iter) {
                vec.push_back(iter->second);
            };
        }

        bool updateThisFill(const ExecutionReport& er);
    };
};

