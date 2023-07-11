#ifndef CSQLPOSITION_HEADER

#define CSQLPOSITION_HEADER

#include "CDateTime.h"

namespace Mts {
    namespace SQL {
        class CSQLPosition
        {
        public:
            CSQLPosition(const Mts::Core::CDateTime &dtTimestamp,
                    unsigned int iAlgoID,
                    unsigned int iSymbolID,
                    unsigned long iLongQty,
                    unsigned long iShortQty,
                    double dLongWAP,
                    double dShortWAP)
                    : m_dtTimestamp(dtTimestamp),
                      m_iAlgoID(iAlgoID),
                      m_iSymbolID(iSymbolID),
                      m_iLongQty(iLongQty),
                      m_iShortQty(iShortQty),
                      m_dLongWAP(dLongWAP),
                      m_dShortWAP(dShortWAP) {}

            Mts::Core::CDateTime getTimestamp() const { return m_dtTimestamp; }

            unsigned int getAlgoID() const { return m_iAlgoID; }

            unsigned int getSymbolID() const { return m_iSymbolID; }

            unsigned long getLongQty() const { return m_iLongQty; }

            unsigned long getShortQty() const { return m_iShortQty; }

            double getLongWAP() const { return m_dLongWAP; }

            double getShortWAP() const { return m_dShortWAP; }

        private:
            Mts::Core::CDateTime m_dtTimestamp;
            unsigned int m_iAlgoID;
            unsigned int m_iSymbolID;
            unsigned long m_iLongQty;
            unsigned long m_iShortQty;
            double m_dLongWAP;
            double m_dShortWAP;
        };
    }
}

#endif

