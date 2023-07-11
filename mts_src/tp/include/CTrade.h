#ifndef CTRADE_HEADER

#define CTRADE_HEADER

#include <cstring>
#include "CDateTime.h"
#include "CSymbol.h"
#include "CEvent.h"

namespace Mts {
    namespace OrderBook {
        class CTrade : public Mts::Event::CEvent
        {
        public:
            CTrade();

            CTrade(unsigned int iSymbolID,
                    unsigned int iProviderID,
                    const Mts::Core::CDateTime &dtMtsTimestamp,
                    const Mts::Core::CDateTime &dtExcTimestamp,
                    double dPrice,
                    unsigned int iSize,
                    const std::string &strExch,
                    const std::string &strExDest);

            unsigned int getSymbolID() const;

            unsigned int getProviderID() const;

            const Mts::Core::CDateTime &getMtsTimestamp() const;

            const Mts::Core::CDateTime &getExcTimestamp() const;

            double getPrice() const;

            unsigned int getSize() const;

            std::string getExch() const;

            std::string getExDest() const;

            void setSymbolID(unsigned int iSymbolID);

            void setProviderID(unsigned int iProviderID);

            void setMtsTimestamp(const Mts::Core::CDateTime &dtMtsTimestamp);

            void setExcTimestamp(const Mts::Core::CDateTime &dtExcTimestamp);

            void setPrice(double dPrice);

            void setSize(unsigned int iSize);

            void setExch(const std::string &strExch);

            void setExDest(const std::string &strExDest);

            std::string toString() const;

        private:
            unsigned int m_iSymbolID;
            unsigned int m_iProviderID;
            Mts::Core::CDateTime m_dtMtsTimestamp;
            Mts::Core::CDateTime m_dtExcTimestamp;
            double m_dPrice;
            unsigned int m_iSize;

            char m_szExch[16];
            char m_szExDest[16];
        };
    }
}

#include "CTrade.hxx"

#endif

