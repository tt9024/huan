#ifndef IFEEDSUBSCRIBER_HEADER

#define IFEEDSUBSCRIBER_HEADER

#include "CQuote.h"
#include "CBidAsk.h"
#include "CTrade.h"
#include "CHeartbeat.h"
#include "CKeyValue.h"

namespace Mts {
    namespace Feed {
        class IFeedSubscriber
        {
        public:

            virtual void onFeedQuoteSim(const Mts::OrderBook::CBidAsk &objBidAsk) = 0;

            virtual void onFeedQuoteBidAsk(const Mts::OrderBook::CBidAsk &objBidAsk) = 0;

            virtual void onFeedQuote(const Mts::OrderBook::CQuote &quote) = 0;

            virtual void onFeedTrade(const Mts::OrderBook::CTrade &objTrade) = 0;

            virtual void onFeedHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat) = 0;

            virtual void onFeedLogon(unsigned int iProviderID) = 0;

            virtual void onFeedLogout(unsigned int iProviderID) = 0;

            virtual void onFeedNoMoreData() = 0;

            virtual void onFeedKeyValue(const Mts::OrderBook::CKeyValue &objKeyValue) = 0;
        };
    }
}

#endif



