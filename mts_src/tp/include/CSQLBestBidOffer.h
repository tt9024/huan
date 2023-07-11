#ifndef CSQLBESTBIDOFFER_HEADER

#define CSQLBESTBIDOFFER_HEADER

#include "CDateTime.h"

namespace Mts {
    namespace SQL {
        class CSQLBestBidOffer
        {
        public:
            CSQLBestBidOffer()
                    : m_dtTimestamp(0),
                      m_iSymbolID(0),
                      m_dBidPx(0),
                      m_dAskPx(0),
                      m_iBidQty(0),
                      m_iAskQty(0),
                      m_iBidCPID(0),
                      m_iAskCPID(0),
                      m_dSpreadBps(0) {}

            CSQLBestBidOffer(const Mts::Core::CDateTime &dtTimestamp,
                    unsigned int iSymbolID,
                    double dBidPx,
                    double dAskPx,
                    unsigned int iBidQty,
                    unsigned int iAskQty,
                    unsigned int iBidCPID,
                    unsigned int iAskCPID,
                    double dSpreadBps)
                    : m_dtTimestamp(dtTimestamp),
                      m_iSymbolID(iSymbolID),
                      m_dBidPx(dBidPx),
                      m_dAskPx(dAskPx),
                      m_iBidQty(iBidQty),
                      m_iAskQty(iAskQty),
                      m_iBidCPID(iBidCPID),
                      m_iAskCPID(iAskCPID),
                      m_dSpreadBps(dSpreadBps) {}

            Mts::Core::CDateTime getTimestamp() const { return m_dtTimestamp; }

            unsigned int getSymbolID() const { return m_iSymbolID; }

            double getBidPx() const { return m_dBidPx; }

            double getAskPx() const { return m_dAskPx; }

            unsigned int getBidQty() const { return m_iBidQty; }

            unsigned int getAskQty() const { return m_iAskQty; }

            unsigned int getBidCPID() const { return m_iBidCPID; }

            unsigned int getAskCPID() const { return m_iAskCPID; }

            double getSpreadBps() const { return m_dSpreadBps; }

        private:
            Mts::Core::CDateTime m_dtTimestamp;
            unsigned int m_iSymbolID;
            double m_dBidPx;
            double m_dAskPx;
            unsigned int m_iBidQty;
            unsigned int m_iAskQty;
            unsigned int m_iBidCPID;
            unsigned int m_iAskCPID;
            double m_dSpreadBps;
        };
    }
}

#endif

