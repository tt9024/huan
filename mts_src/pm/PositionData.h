#pragma once

#include <string>
#include <stdio.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <vector>
#include <memory>

#include "ExecutionReport.h"
#include "csv_util.h"
#include "plcc/PLCC.hpp"
#include "symbol_map.h"

// forward declaration 
namespace algo {
    class PositionManagerMock;
}

namespace pm {
    struct OpenOrder;
        // Forward declaration of open order data structure

    class IntraDayPosition {
    public:
        IntraDayPosition();
            // create an empty intra-day position

        IntraDayPosition(
                const std::string& symbol,
                const std::string& algo,
                int64_t qty = 0,
                double vap = 0,
                int64_t last_micro = 0 
                );
            // create an empty intra-day position, with optional
            // initial values

        explicit IntraDayPosition(const utils::CSVUtil::LineTokens& tokens);
            // create intra-day position from a csv line

        ~IntraDayPosition();

        void update(const ExecutionReport& er);
            // update from an execution report

        void deleteOO(const char* clOrdId = nullptr);
            // delete an open order (if exist) with clOrdId

        void resetPositionUnsafe(int64_t qty, double vap, uint64_t last_utc=0);
            // this resets the position to given qty, vap.
            // qty is sign significant; positive is buy, negative is sell
            // there will be only one sided position after this, i.e. 
            // the other side will be zero and the realized pnl will be 0

        void operator+=(const IntraDayPosition& idp);
        IntraDayPosition operator+(const IntraDayPosition& idp);
            // aggregate two intra-day position

        void resetPnl();
            // remove one side of qty/vwap and set pnl to 0

        std::string diff(const IntraDayPosition& idp, bool check_pnl=false) const;
            // finds difference with the given idp, 
            // return "" in case no difference is found 
            // set check_pnl to true to also check the realized pnl
            // from the position.  This may differ if one is loaded
            // from eod (with pnl reset)
            //
        int64_t tgt_fill(int64_t tgt_qty, double tgt_vap, double* fill_px=nullptr) const;
            // finds a qty and a px of a fill so that, with the additon of
            // this fill, tgt_qty and tgt_vap would be achieved.
            // both tgt_qty and returned qty are sign significant.
            // return the qty, set the px to fill_px
        
        bool operator==(const IntraDayPosition& idp) const;
            // compares two positions, in terms of position and vap

        utils::CSVUtil::LineTokens toCSVLine(bool use_mtm_pnl=false, double mtm_pnl_adjust=0, const double* ref_px=nullptr) const;
            // writes realized pnl to eod_pos.csv
        utils::CSVUtil::LineTokens toCSVLineMtm(const double* ref_px=nullptr) const; 
            // writes mtm pnl as an additional column into eod_pos_mtm.csv
        utils::CSVUtil::LineTokens toCSVLineMtmDaily(const utils::CSVUtil::FileTokens& prev_vec, const double* ref_px=nullptr) const; 
            // writes daily mtm pnl as pnl in same format of eod_pos.csv

        std::string toString() const;
            // writes the current position to a text line

        std::string dumpOpenOrder() const;
            // writes all open orders to multiple text lines

        bool hasPosition() const;
            // check if the position is empty

        int64_t getPosition(double* ptr_vap = nullptr, double* ptr_pnl = nullptr) const;
            // gets the current position, and optionally the volumn weighted price (vap)
            // and the realized pnl (pnl)
            // Note the return value is sign significant, i.e. positive is qty long
            // negative is qty short

        int64_t getOpenQty() const;
            // gets the total open qty. sign significant, i.e. positive is qty long
            // negative is qty short

        std::vector<std::shared_ptr<const OpenOrder> > listOO() const;
            // list all currently open orders

        std::shared_ptr<const OpenOrder> findOO(const std::string& clOrdId) const;

        double getRealizedPnl() const;
            // calculate current realized pnl
        
        double getMtmPnl(const double* ref_px = nullptr) const;
            // calculate realized pnl plus mark to market pnl
            // using the given ref_px
           
        int getPointValue() const;
        
        std::string get_symbol() const { return m_symbol ; };
        std::string get_algo() const   { return m_algo;    };

        [[nodiscard]] auto getVAP() const noexcept { return m_vap; }
        [[nodiscard]] auto getQuantity() const noexcept { return m_qty; }
        [[nodiscard]] auto getProfitAndLoss() const noexcept { return m_pnl; }
        [[nodiscard]] auto getMarkToMarket() const noexcept { return getMtmPnl(nullptr); }

        bool is_mleg() const { return utils::SymbolMapReader::get().isMLegSymbol(m_symbol); };

        static std::string get_pod_from_algo(const std::string& algo) { return algo.substr(0,algo.find("-")); };

    protected:
        std::string m_algo;
        std::string m_symbol;
        int m_contract_size;
        int64_t m_qty;
        double m_vap;
        double m_pnl;
        uint64_t m_last_micro;
        std::map<std::string, std::shared_ptr<OpenOrder> > m_oo;

        void addFill(int64_t qty, double px, uint64_t utc_micro);
        void addOO(const ExecutionReport& er);
        void updateOO(const char* clOrdId, int64_t qty);

        friend class PositionManagerMock;
        // purpose of testing and simulation
    };

    struct OpenOrder {
        const IntraDayPosition* m_idp;
        int64_t m_ord_qty;  // the inital qty upon open
        IDType m_clOrdId;
        //IDType m_execId;  // in case clOrdId is not unique (?)
        int64_t m_open_qty; // + buy - sell, from er
        double m_open_px;
        uint64_t m_open_micro;

        explicit OpenOrder(const IntraDayPosition* idp = nullptr);
            // create an empty open order

        OpenOrder(const IntraDayPosition* idp, const ExecutionReport& er);
            // create an open order from execution report

        std::string toString() const;
            // writes this open order to a text line
    };
}
