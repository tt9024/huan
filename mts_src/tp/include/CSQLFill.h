#ifndef CSQLFILL_HEADER

#define CSQLFILL_HEADER

#include "CDateTime.h"

namespace Mts {
    namespace SQL {
        class CSQLFill
        {
        public:
            CSQLFill(const Mts::Core::CDateTime &dtFillTimestamp,
                    unsigned int iSymbolID,
                    char iDirection,
                    unsigned int iFillQty,
                    double dFillPrice,
                    unsigned int iOriginatorAlgoID)
                    : m_dtFillTimestamp(dtFillTimestamp),
                      m_iSymbolID(iSymbolID),
                      m_iDirection(iDirection),
                      m_iFillQty(iFillQty),
                      m_dFillPrice(dFillPrice),
                      m_iOriginatorAlgoID(iOriginatorAlgoID),
                      m_iProviderID(0) {}

            CSQLFill(const Mts::Core::CDateTime &dtFillTimestamp,
                    unsigned int iSymbolID,
                    char iDirection,
                    unsigned int iFillQty,
                    double dFillPrice)
                    : m_dtFillTimestamp(dtFillTimestamp),
                      m_iSymbolID(iSymbolID),
                      m_iDirection(iDirection),
                      m_iFillQty(iFillQty),
                      m_dFillPrice(dFillPrice),
                      m_iOriginatorAlgoID(0),
                      m_iProviderID(0) {}

            CSQLFill(const Mts::Core::CDateTime &dtFillTimestamp,
                    unsigned int iSymbolID,
                    char iDirection,
                    unsigned int iFillQty,
                    double dFillPrice,
                    unsigned int iOriginatorAlgoID,
                    unsigned int iProviderID)
                    : m_dtFillTimestamp(dtFillTimestamp),
                      m_iSymbolID(iSymbolID),
                      m_iDirection(iDirection),
                      m_iFillQty(iFillQty),
                      m_dFillPrice(dFillPrice),
                      m_iOriginatorAlgoID(iOriginatorAlgoID),
                      m_iProviderID(iProviderID) {}

            unsigned int getOriginatorAlgoID() const { return m_iOriginatorAlgoID; }

            const Mts::Core::CDateTime &getFillTimestamp() const { return m_dtFillTimestamp; }

            unsigned int getSymbolID() const { return m_iSymbolID; }

            char getDirection() const { return m_iDirection; }

            unsigned int getFillQty() const { return m_iFillQty; }

            double getFillPrice() const { return m_dFillPrice; }

            unsigned int getProviderID() const { return m_iProviderID; }

        private:
            unsigned int m_iOriginatorAlgoID;
            Mts::Core::CDateTime m_dtFillTimestamp;
            unsigned int m_iSymbolID;
            char m_iDirection;
            unsigned int m_iFillQty;
            double m_dFillPrice;
            unsigned int m_iProviderID;
        };
    }
}

#endif

