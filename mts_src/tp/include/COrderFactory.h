#ifndef CORDERFACTORY_HEADER

#define CORDERFACTORY_HEADER

#include <boost/unordered_map.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include "COrder.h"
#include "COrderCancelRequest.h"
#include "CLogBinary.h"

namespace Mts {
    namespace Order {
        class COrderFactory
        {
        public:
            static COrderFactory &getInstance();

            static void setModeSQL(bool bUseSQL);

            static void setLoadOrderHistory(bool bLoadOrderHistory);

            ~COrderFactory();

            const COrderCancelRequest &createCancellationRequest(
                    unsigned int iOriginatorAlgoID,
                    const COrder &objOrigOrder
            );

            const COrderCancelRequest &createCancellationRequest(
                    Mts::Core::CDateTime &dtCreateTime,
                    unsigned int iOriginatorAlgoID,
                    const COrder &objOrigOrder);

            const COrder &createIOCOrder(unsigned int iOriginatorAlgoID,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    COrder::BuySell iDirection,
                    unsigned int uiQuantity,
                    double dPrice,
                    const std::string &strOrderTag,
                    const std::string &strExecBrokerCode);

            const COrder &createIOCOrder(Mts::Core::CDateTime &dtCreateTime,
                    unsigned int iOriginatorAlgoID,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    COrder::BuySell iDirection,
                    unsigned int uiQuantity,
                    double dPrice,
                    const std::string &strOrderTag,
                    const std::string &strExecBrokerCode);

            void createIOCOrder(Mts::Core::CDateTime &dtCreateTime,
                    unsigned int iOriginatorAlgoID,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    COrder::BuySell iDirection,
                    unsigned int uiQuantity,
                    double dPrice,
                    const std::string &strOrderTag,
                    const std::string &strExecBrokerCode,
                    COrder &objOrder);

            const COrder &createGTCOrder(
                    const std::string &strMtsOrderId,
                    unsigned int iOriginatorAlgoID,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    COrder::BuySell iDirection,
                    unsigned int uiQuantity,
                    double dLimitPrice,
                    const std::string &strOrderTag,
                    const std::string &strExecBrokerCode);

            const COrder &createGTCOrder(Mts::Core::CDateTime &dtCreateTime,
                    const std::string &strMtsOrderId,
                    unsigned int iOriginatorAlgoID,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    COrder::BuySell iDirection,
                    unsigned int uiQuantity,
                    double dLimitPrice,
                    const std::string &strOrderTag,
                    const std::string &strExecBrokerCode);

            const COrder &createIcebergOrder(Mts::Core::CDateTime &dtCreateTime,
                    unsigned int iOriginatorAlgoID,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    COrder::BuySell iDirection,
                    unsigned int uiQuantity,
                    double dLimitPrice,
                    const std::string &strOrderTag,
                    const std::string &strExecBrokerCode);

            const COrder &createTWAPOrder(Mts::Core::CDateTime &dtCreateTime,
                    unsigned int iOriginatorAlgoID,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    COrder::BuySell iDirection,
                    unsigned int uiQuantity,
                    double dLimitPrice,
                    const std::string &strOrderTag,
                    const std::string &strExecBrokerCode);

            bool isOrderInCache(const char *pszMtsOrderID);

            bool isCancelRequestInCache(const char *pszMtsCancellationID);

            COrder &getOrder(const char *pszMtsOrderID);

            const COrder &getOrderFromCancellationID(const char *pszMtsCancellationID);

            unsigned long queryNextOrderID();

            unsigned long getNextOrderID();

            unsigned long getNextCancellationID();

            void loadNextOrderID();

            void setGenerateUniqueOrderID(bool bGenerateUniqueOrderID);

            void setNextOrderID(unsigned long ulNextOrderID);

            void setNextCancellationID(unsigned long ulNextCancellationID);

            std::string createMtsOrderID(Mts::Core::CDateTime &dtCreateTime,
                    unsigned long ulOrderID,
                    unsigned int iOriginatorAlgoID);

            std::string createMtsCancellationID(Mts::Core::CDateTime &dtCreateTime,
                    unsigned long ulCancellationID,
                    unsigned int iOriginatorAlgoID);

            void shutDown();

        private:
            COrderFactory();

            COrderFactory(const COrderFactory &objRhs);

            // restore state from log files
            void loadDailyHistory();

            // restore state from SQL
            void loadDailyHistorySQL();

        private:
            enum
            {
                MAX_NUM_ENGINES = 5
            };

            // Mts order ID -> order
            typedef boost::unordered_map<std::string, Mts::Order::COrder> OrdersMap;

            // Mts cancel req ID -> cancel request
            typedef boost::unordered_map<std::string, Mts::Order::COrderCancelRequest> CancelRequestMap;

            static bool m_bUseSQL;
            static bool m_bLoadOrderHistory;

            OrdersMap m_Orders;
            CancelRequestMap m_CancelRequests;

            bool m_bGenerateUniqueOrderID;
            unsigned long m_ulNextOrderID;
            unsigned long m_ulNextCancellationID;

            boost::mutex m_MutexOrders;
            boost::mutex m_MutexOrderID;
            boost::mutex m_MutexCancellationID;

            COrder m_Order;
            COrderCancelRequest m_CancelRequest;

            Mts::Log::CLogBinary m_OrderLog;
            Mts::Log::CLogBinary m_CancelRequestLog;
        };
    }
}

#endif

