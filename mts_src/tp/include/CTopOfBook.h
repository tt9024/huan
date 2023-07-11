#ifndef CTOPOFBOOK_HEADER

#include "CBidAsk.h"

#define CTOPOFBOOK_HEADER

namespace Mts {
    namespace OrderBook {
        class CTopOfBook
        {
        public:
            CTopOfBook();

            void update(const CQuote &objBestBid,
                    const CQuote &objBestAsk);

            void update(const CBidAsk &objConsolidatedBook);

            bool isChanged() const;

            double getMidPx() const;

        private:
            double m_dBestBidPx;
            unsigned int m_iBestBidQty;
            double m_dBestAskPx;
            unsigned int m_iBestAskQty;
            bool m_bChanged;
        };
    }
}

#endif

