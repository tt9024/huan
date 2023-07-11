#ifndef CALGORITHMFIXTEST_HEADER

#define CALGORITHMFIXTEST_HEADER

#include <boost/unordered_map.hpp>
#include <boost/thread/mutex.hpp>
#include "CAlgorithm.h"
#include "COrderCancelRequest.h"
#include "CConfig.h"

namespace Mts {
    namespace Algorithm {
        class CAlgorithmFIXTest : public CAlgorithm
        {
        public:
            // construction
            CAlgorithmFIXTest(unsigned int uiAlgoID,
                    const std::string &strAlgoName,
                    unsigned int uiEventBufferSizeBytes,
                    unsigned int uiUpdateMSec,
                    unsigned int uiTestID);

            ~CAlgorithmFIXTest();

            // overrides from CAlgorithm
            void onCreate();

            void onStart();

            void onStartOfDay(Mts::Core::CDateTime dtTimestamp);

            void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk);

            void onEvent(const Mts::OrderBook::CTrade &objTrade);

            void onEvent(const Mts::Order::COrderStatus &objOrderStatus);

            void onEvent(const Mts::Order::COrderFill &objOrderFill);

            void onEvent(const Mts::Core::CDateTimeEvent &objDateTime);

            void onEvent(const std::string &strUserCommand);

            void onEvent(const Mts::OrderBook::CKeyValue &objKeyValue);

            void onEndOfDay(Mts::Core::CDateTime dtTimestamp);

            void onStop();

            void onDestroy();

            void addTradedInstrument(const Mts::Core::CSymbol &objSymbol);

            std::vector<std::string> getMessageHistory() const;

        private:
            Mts::Order::COrder m_Orders[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            Mts::Order::COrderCancelRequest m_CancelReqs[Mts::Core::CConfig::MAX_NUM_SYMBOLS];

            unsigned int m_iTestID;
            bool m_bOrderSent;
            bool m_bCancelSent;

            // algo is single threaded but user commands are sent from a second thread hence synchronization is required
            boost::mutex m_Mutex;
        };
    }
}

#endif


