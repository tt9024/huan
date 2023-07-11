#ifndef CSQLMTM_HEADER

#define CSQLMTM_HEADER

#include "CDateTime.h"

namespace Mts {
    namespace SQL {
        class CSQLMTM
        {
        public:
            CSQLMTM(const Mts::Core::CDateTime &dtTimestamp,
                    unsigned int iAlgoID,
                    unsigned int iSymbolID,
                    unsigned long iLongQty,
                    unsigned long iShortQty,
                    double dLongWAP,
                    double dShortWAP,
                    const Mts::Core::CDateTime &dtTimestampMidPx,
                    double dMidPx,
                    double dCostUSD)
                    : m_dtTimestamp(dtTimestamp),
                      m_iAlgoID(iAlgoID),
                      m_iSymbolID(iSymbolID),
                      m_iLongQty(iLongQty),
                      m_iShortQty(iShortQty),
                      m_dLongWAP(dLongWAP),
                      m_dShortWAP(dShortWAP),
                      m_dtTimestampMidPx(dtTimestampMidPx),
                      m_dMidPx(dMidPx),
                      m_dCostUSD(dCostUSD) {}

            Mts::Core::CDateTime getTimestamp() const { return m_dtTimestamp; }

            unsigned int getAlgoID() const { return m_iAlgoID; }

            unsigned int getSymbolID() const { return m_iSymbolID; }

            unsigned long getLongQty() const { return m_iLongQty; }

            unsigned long getShortQty() const { return m_iShortQty; }

            double getLongWAP() const { return m_dLongWAP; }

            double getShortWAP() const { return m_dShortWAP; }

            Mts::Core::CDateTime getTimestampMidPx() const { return m_dtTimestampMidPx; }

            double getMidPx() const { return m_dMidPx; }

            double getCostUSD() const { return m_dCostUSD; }

        private:
            Mts::Core::CDateTime m_dtTimestamp;
            unsigned int m_iAlgoID;
            unsigned int m_iSymbolID;
            unsigned long m_iLongQty;
            unsigned long m_iShortQty;
            double m_dLongWAP;
            double m_dShortWAP;
            Mts::Core::CDateTime m_dtTimestampMidPx;
            double m_dMidPx;
            double m_dCostUSD;
        };
    }
}

#endif

