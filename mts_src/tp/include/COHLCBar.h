#ifndef COHLCBAR_HEADER

#define COHLCBAR_HEADER

#include "CDateTime.h"
#include "CBidAsk.h"

namespace Mts {
    namespace Indicator {
        class COHLCBar
        {
        public:
            COHLCBar();

            const Mts::Core::CDateTime &getTimestampOpen() const;

            const Mts::Core::CDateTime &getTimestampClose() const;

            double getOpen() const;

            double getHigh() const;

            double getLow() const;

            double getClose() const;

            void update(const Mts::OrderBook::CBidAsk &objBidAsk);

            void update(const Mts::Core::CDateTime &dtTimestamp);

            void roll();

        private:
            Mts::Core::CDateTime m_dtTimestampOpen;
            Mts::Core::CDateTime m_dtTimestampClose;

            double m_dOpen;
            double m_dHigh;
            double m_dLow;
            double m_dClose;

            bool m_bNewBar;
        };
    }
}

#endif

