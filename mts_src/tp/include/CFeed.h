#ifndef CFEED_HEADER

#define CFEED_HEADER

#include <vector>
#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>
#include "IFeedSubscriber.h"
#include "CBidAsk.h"
#include "CTrade.h"
#include "IRunnable.h"
#include "CHeartbeat.h"
#include "CKeyValue.h"

namespace Mts {
    namespace Feed {
        class CFeed : public Mts::Thread::IRunnable
        {
        public:
            CFeed();

            CFeed(bool bFIXProtocol);

            // implementation of IRunnable
            void run();

            void onStop();

            virtual bool connect() = 0;

            virtual bool disconnect() = 0;

            virtual void addSubscriber(const std::string &strProvider);

            virtual void removeSubscriber(const std::string &strProvider);

            virtual void initialize() = 0;

            void publishTrade(const Mts::OrderBook::CTrade &objTrade);

            void publishQuoteSim(const Mts::OrderBook::CBidAsk &objQuoteBidAsk);

            void publishQuote(const Mts::OrderBook::CBidAsk &objQuoteBidAsk);

            void publishHalfQuote(const Mts::OrderBook::CQuote &objQuote);

            void publishHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat);

            void publishLogon(unsigned int iProviderID);

            void publishLogout(unsigned int iProviderID);

            void publishNoMoreData();

            void publishKeyValue(const Mts::OrderBook::CKeyValue &objKeyValue);

            const Mts::Core::CDateTime &getHeartbeat() const;

            void setHeartbeat(const Mts::Core::CDateTime &dtNow);

            void addSubscriber(IFeedSubscriber *ptrSubscriber);

            bool isFIXProtocol() const;

        private:
            typedef std::vector<IFeedSubscriber *> FeedSubscriberArray;

            FeedSubscriberArray m_FeedSubscribers;
            boost::shared_ptr<boost::thread> m_ptrThread;

            bool m_bFIXProtocol;

            Mts::Core::CDateTime m_dtHeartbeat;
        };
    }
}

#endif

