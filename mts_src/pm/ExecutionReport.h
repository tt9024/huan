#pragma once

#include "csv_util.h"
#include <string>
#include <stdexcept>
#include <memory>

namespace pm {
    static const int SymbolLen = 32;
    static const int IDLen = 64;
    static const int StatusLen = 4;
    using SymbolType = char[SymbolLen];
    using IDType = char[IDLen];
    using StatusType = char[StatusLen];
    using TEXTType = char[2*IDLen];

    struct ExecutionReport {
        ExecutionReport();
        ExecutionReport (
                const std::string& symbol, // exchange symbol
                const std::string& algo,   // fully qualified for position
                const std::string& clOrdId,
                const std::string& execId,
                const std::string& tag39,
                int qty, // sign significant: + buy, - sell
                double px,
                const std::string& utcTime, // YYYYMMDD-HH:MM:SS[.sss]
                const std::string& optionalTag, // reserved 
                uint64_t recv_micro,   // utc when it is received locally
                int64_t reserved       // currently used for signed cumQty
        );

        static std::string ERPersistFile();
        static ExecutionReport fromCSVLine(const utils::CSVUtil::LineTokens& token_vec);

        // functions to output to a csv line
        utils::CSVUtil::LineTokens toCSVLine() const;
        utils::CSVUtil::LineTokens toFillsCSVLine() const; // uses mts symbol format

        // this loads execution reports from the persist that are received
        // between given start_time_local and end_time_local and writes out_file
        static bool loadFromPersistence(const std::string& start_time_local, 
                                        const std::string& end_time_local, 
                                        const std::string& out_file, 
                                        const std::string& er_persist_file="");


        // static functions for unique client order id generation/parsing
        static std::string genClOrdId();
        static std::string genClOrdId(int pi_type, uint32_t woid);
        static std::string genReplaceClOrdId(const std::string& clOrdId);
        static std::string parseClOrdId(const char* clOrdId);
        static std::string parseClOrdId(const char* clOrdId, int& pi_type, uint32_t& woid);
        static const char* syntheticOrdIdPrefix(); // return MtsFloor

        // this generates a synthetic fill, with clOrdId being 
        // clOrdId set as "MtsFloor-" plus clOrdId_str and a timestamp
        // The clOrdId_str shows the purpose of this fill, such as
        // IntenalMatch, PaperTrade, or PositionAdjust, etc.
        static ExecutionReport genSyntheticFills(const std::string& symbol,
                                                 const std::string& algo,
                                                 int qty, // sign significant
                                                 double px,
                                                 const std::string& clOrdId_str = "",
                                                 const std::string& optional_tag = "");

        SymbolType m_symbol;
        SymbolType m_algo;
        IDType m_clOrdId;
        IDType m_execId;
        StatusType m_tag39;
        int m_qty;   // + buy, - sell
        double m_px;
        uint64_t m_utc_milli;
        TEXTType m_optional;
        uint64_t m_recv_micro;
        int64_t m_reserved; // reserved for filled qty in new 

        // helper functions
        bool isFill() const;
        bool isNew() const;
        bool isCancel() const;
        bool isReject() const;
        std::string toString() const;
        bool compareTo(const ExecutionReport& er, std::string* difflog=nullptr) const;



    };
}

