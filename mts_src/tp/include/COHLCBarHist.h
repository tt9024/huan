#ifndef COHLCBARHIST_HEADER

#define COHLCBARHIST_HEADER

#include "COHLCBar.h"
#include "CCircularBuffer.h"

namespace Mts {
    namespace Indicator {
        class COHLCBarHist
        {
        public:
            COHLCBarHist(unsigned int iBarSizeSecs,
                    unsigned int iNumBars);

            void update(const Mts::OrderBook::CBidAsk &objBidAsk);

            void update(const Mts::Core::CDateTime &dtTimestamp);

            void roll();

            bool isFull() const;

            unsigned int getNumBars() const;

            std::tuple<Mts::Core::CDateTime, double, double, double, double> getOHLCBar(unsigned int iIndex) const;

            Mts::Core::CDateTime getBarTimestampClose(unsigned int iIndex) const;

            double getBarOpen(unsigned int iIndex) const;

            double getBarHigh(unsigned int iIndex) const;

            double getBarLow(unsigned int iIndex) const;

            double getBarClose(unsigned int iIndex) const;

        private:
            using OHLCBarHist = Mts::Indicator::CCircularBuffer<Mts::Indicator::COHLCBar>;

            unsigned int m_iRollMinute;
            unsigned int m_iNumBars;

            Mts::Core::CDateTime m_dtLastRoll;

            Mts::Indicator::COHLCBar m_CurrBar;

            OHLCBarHist m_BarHist;
        };
    }
}


#endif

