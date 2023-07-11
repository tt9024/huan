#ifndef CPOSITION_HEADER

#define CPOSITION_HEADER

#include "CSymbol.h"
#include "COrderFill.h"

namespace Mts {
    namespace Accounting {
        class CPosition
        {
        public:
            CPosition();

            explicit CPosition(const Mts::Core::CSymbol &objSymbol);

            CPosition(
                    const Mts::Core::CSymbol &objSymbol,
                    unsigned long ulLongQty,
                    double dLongWAP,
                    unsigned long ulShortQty,
                    double dShortWAP
            );

            CPosition(
                    const Mts::Core::CSymbol &objSymbol,
                    long iPosition,
                    double dWAP);

            // accessors
            long getPosition() const;

            long getGrossPosition() const;

            double getWAP() const;

            unsigned int getSymbolID() const;

            // operational
            double calcPnL(double dMktPrice);

            double getPnL() const;

            void updatePosition(
                    const Mts::Order::COrderFill &objFill
            );

            void updatePosition(
                    const Mts::Order::COrder::BuySell iDirection,
                    unsigned int iQuantity,
                    double dPrice
            );

            std::tuple<long, double, long, double, double> getSummary() const;

            std::string toString() const;

            bool fromString(const std::string &line);

        private:
            unsigned int m_uiSymbolID;
            unsigned long m_ulLongQty;
            double m_dLongWAP;
            unsigned long m_ulShortQty;
            double m_dShortWAP;
            double m_dPointValue;
            double m_dLastMktPrice;
            double m_dLastPnL;
        };
    }
}

#endif

