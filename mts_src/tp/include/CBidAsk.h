#ifndef CBIDASK_HEADER

#define CBIDASK_HEADER

#include <cstring>
#include "CQuote.h"
#include "CEvent.h"
#include "CDateTime.h"

namespace Mts {
    namespace OrderBook {
        class CBidAsk : public Mts::Event::CEvent
        {
        public:
            CBidAsk();

            CBidAsk(unsigned int iSymbolID,
                    const Mts::Core::CDateTime &dtTimestamp,
                    const CQuote &objBid,
                    const CQuote &objAsk);

            const Mts::Core::CDateTime &getTimestamp() const;

            unsigned int getSymbolID() const;

            const CQuote &getBid() const;

            const CQuote &getAsk() const;

            std::string toString() const;

            double getMidPx() const;

        private:
            unsigned int m_iSymbolID;
            Mts::Core::CDateTime m_dtTimestamp;
            CQuote m_Bid;
            CQuote m_Ask;
        };
    }
}

#endif

