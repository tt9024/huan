#ifndef CPRICEMATRIX_HEADER_HPP

#define CPRICEMATRIX_HEADER_HPP

#include <string>
#include <set>
#include <map>
#include <vector>
#include "CSQLBestBidOffer.h"
#include "CDateTimeLessThan.h"
#include "CSymbol.h"
#include "CConfig.h"

namespace Mts {
    namespace TickData {
        class CPriceMatrix
        {
        public:
            CPriceMatrix(const std::vector<Mts::SQL::CSQLBestBidOffer> &objPriceHist);

            ~CPriceMatrix();

            unsigned int getNumDates() const;

            unsigned int getNumSymbols() const;

            const Mts::Core::CDateTime &getTimestamp(unsigned int iIndex) const;

            double getMidPx(unsigned int iTimestampIndex,
                    const Mts::Core::CSymbol &objSymbol) const;

            bool saveToFile(const std::string &strSymbolsOutputFile,
                    const std::string &strPricesOutputFile) const;

            bool saveToFileSpreads(const std::string &strSpreadsOutputFile) const;

        private:
            typedef std::set<Mts::Core::CDateTime, Mts::Core::CDateTimeLessThan> OrderedTimestamps;
            typedef std::set<unsigned int> OrderedSymbolIDs;
            typedef std::vector<Mts::Core::CDateTime> TimestampArray;

            TimestampArray m_TimestampsArray;
            OrderedTimestamps m_Timestamps;
            OrderedSymbolIDs m_SymbolIDs;

            int m_SymbolID2Column[Mts::Core::CConfig::MAX_NUM_SYMBOLS];

            unsigned int m_iNumDates;
            unsigned int m_iNumSymbols;
            Mts::SQL::CSQLBestBidOffer **m_Matrix;
        };
    }
}

#endif
