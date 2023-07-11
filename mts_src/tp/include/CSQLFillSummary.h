#ifndef CSQLFILLSUMMARY_HEADER

#define CSQLFILLSUMMARY_HEADER

#include "CDateTime.h"

namespace Mts {
    namespace SQL {
        class CSQLFillSummary
        {
        public:
            CSQLFillSummary(unsigned int iAlgoID,
                    unsigned int iFillCount,
                    double dTradedQtyUsdMM,
                    const Mts::Core::CDateTime &dtLastFillTimestamp)
                    : m_iAlgoID(iAlgoID),
                      m_iFillCount(iFillCount),
                      m_dTradedQtyUsdMM(dTradedQtyUsdMM),
                      m_dtLastFillTimestamp(dtLastFillTimestamp) {}

            unsigned int getAlgoID() const { return m_iAlgoID; }

            unsigned int getFillCount() const { return m_iFillCount; }

            double getTradedQtyUsdMM() const { return m_dTradedQtyUsdMM; }

            const Mts::Core::CDateTime &getLastFillTimestamp() const { return m_dtLastFillTimestamp; }

        private:
            unsigned int m_iAlgoID;
            unsigned int m_iFillCount;
            double m_dTradedQtyUsdMM;
            Mts::Core::CDateTime m_dtLastFillTimestamp;
        };
    }
}

#endif

