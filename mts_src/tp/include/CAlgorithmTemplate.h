#ifndef CALGORITHMTEMPLATE_HEADER

#define CALGORITHMTEMPLATE_HEADER

#include <boost/unordered_map.hpp>
#include <boost/thread/mutex.hpp>
#include "CAlgorithm.h"
#include "COrderCancelRequest.h"
#include "CConfig.h"
#include "CEmailer.h"
#include "CModel.h"

namespace Mts {
    namespace Algorithm {
        class CAlgorithmTemplate : public CAlgorithm
        {
        public:
            // construction
            CAlgorithmTemplate(unsigned int uiAlgoID,
                    const std::string &strAlgoName,
                    unsigned int uiEventBufferSizeBytes,
                    unsigned int uiUpdateMSec,
                    unsigned int uiExecAlgoID);

            ~CAlgorithmTemplate();

            // overrides from CAlgorithm
            void onCreate();

            void onStart();

            void onStartOfDay(Mts::Core::CDateTime dtTimestamp);

            void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk);

            void onEvent(const Mts::OrderBook::CTrade &objTrade);

            void onEvent(const Mts::Order::COrderStatus &objOrderStatus);

            void onEvent(const Mts::Order::COrderFill &objOrderFill);

            void onEvent(const Mts::Core::CDateTimeEvent &objDateTime);

            void onEvent(const Mts::OrderBook::CManualCommand &objMC);

            void onEvent(const Mts::OrderBook::CKeyValue &objKeyValue);

            void onEndOfDay(Mts::Core::CDateTime dtTimestamp);

            void onStop();

            void onDestroy();

            void addTradedInstrument(const Mts::Core::CSymbol &objSymbol);

            // algo specific
            bool onEvent_Internal(const Mts::OrderBook::CBidAsk &objBidAsk);

            void allocateFillToModels(const Mts::Order::COrderFill &objOrderFill);

            // accessors
            void addModel(boost::shared_ptr<Mts::Model::CModel> ptrModel);

            void setTradeTime(unsigned int iTimeOfDayMins);

            // return history of all messages sent from this algorithm
            std::vector<std::string> getMessageHistory() const;

            // making orders, cancel and position requests
            bool sendOrder(bool isBuy, const std::string &symbol,
                    long long qty, const std::string &priceStr);

            bool cancelOrder(const std::string &symbol);

            std::string getBidAsk(const std::string &symbol) const;

            std::string getPosition(const std::string &symbol) const;

            std::string getOpenOrders(const std::string &symbol) const;

        private:
            using ModelArray = std::vector<boost::shared_ptr<Mts::Model::CModel> >;

            enum
            {
                MINS_PER_DAY = 1440
            };

            enum
            {
                ALGO_TWAP = 0, ALGO_ICEBERG = 1
            };

            // sub-models
            ModelArray m_Models;

            // models are triggered on a time schedule, max update frequency is 1 minute
            unsigned int m_iTradeTimeMap[MINS_PER_DAY];
            Mts::Core::CDateTime m_dtLastUpdate;

            Mts::Order::COrder m_Orders[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            Mts::Order::COrderCancelRequest m_CancelReqs[Mts::Core::CConfig::MAX_NUM_SYMBOLS];

            double m_dLastBidPx[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            double m_dLastAskPx[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            Mts::Core::CDateTime m_dtLast[Mts::Core::CConfig::MAX_NUM_SYMBOLS];

            // execution algo ID
            unsigned int m_iExecAlgoID;

            // algo is single threaded but user commands are sent from a second thread hence synchronization is required
            boost::mutex m_Mutex;
        };
    }
}

#endif


