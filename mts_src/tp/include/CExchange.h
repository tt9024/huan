#ifndef CEXCHANGE_HEADER

#define CEXCHANGE_HEADER

#include <vector>
#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>
#include "IExchangeSubscriber.h"
#include "COrder.h"
#include "COrderStatus.h"
#include "COrderFill.h"
#include "COrderCancelRequest.h"
#include "IRunnable.h"
#include "CBidAsk.h"
#include "CHeartbeat.h"
#include "CExecReport.h"
#include "IFIXSession.h"

namespace Mts {
    namespace Exchange {
        class CExchange : public Mts::Thread::IRunnable, public Mts::FIXEngine::IFIXSession
        {
        public:
            enum TimeInForce
            {
                GTC = 0, GTD = 1
            };

        public:
            CExchange();

            CExchange(bool bFIXProtocol);

            // implementation of IRunnable
            void run();

            // exchange specific implementation
            virtual bool connect() = 0;

            virtual bool disconnect() = 0;

            virtual bool submitCustomOrder(const Mts::Core::CDateTime &dtNow,
                    const std::string &strOrderSpec);

            virtual bool cancelCustomOrder(const Mts::Core::CDateTime &dtNow,
                    const std::string &strCancelReq);

            virtual bool submitLmtOrder(const Mts::Order::COrder &objOrder,
                    Mts::Exchange::CExchange::TimeInForce iTIF);

            virtual bool submitMktOrder(const Mts::Order::COrder &objOrder);

            virtual bool submitTWAPOrder(const Mts::Order::COrder &objOrder);

            virtual bool submitIcebergOrder(const Mts::Order::COrder &objOrder);

            virtual bool cancelOrder(const Mts::Order::COrderCancelRequest &objCancelReq) = 0;

            virtual bool cancelOrder(const std::string &origOrdId, const std::string &algo) { return false; };

            virtual bool replaceOrder(const std::string &origClOrdId, int64_t qty, double px, const std::string &algo,
                    const std::string &symbol, const std::string &newClOrdId) { return false; };


            // these methods should only be overridden by test exchanges which have to be fed market data from an external source
            virtual void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk);

            virtual unsigned int getProviderID() const = 0;

            void publishRiskViolation(const Mts::Order::COrderStatus &objOrderStatus);

            void publishOrderNew(const Mts::Order::COrder &objOrder);

            void publishOrderStatus(const Mts::Order::COrderStatus &objOrderStatus);

            void publishOrderFill(const Mts::Order::COrderFill &objOrderFill);

            void publishExecReport(const Mts::Order::CExecReport &objExecReport);

            void publishHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat);

            void publishLogon(unsigned int iProviderID);

            void publishLogout(unsigned int iProviderID);

            void publishExchangeMessage(const std::string &strMsg);

            void addSubscriber(IExchangeSubscriber *ptrSubscriber);

            bool isFIXProtocol() const;

            // the only method used in MTS - stateless publishing of an ExecutionReport,
            // without requring prior COrders, OriginalOrders at tpmain
            // thus more robust and efficient (fast)
            // TODO - dramatically clean up this code, removing most unused codes
            // and compiler warnings.
            void publishExecutionReport(const pm::ExecutionReport &er);

        private:
            bool approveOrder(const Mts::Order::COrder &objOrder);

        private:
            typedef std::vector<IExchangeSubscriber *> ExchangeSubscriberArray;
            ExchangeSubscriberArray m_ExchangeSubscribers; // handle the IFIX events
            boost::shared_ptr<boost::thread> m_ptrThread;

            bool m_bFIXProtocol;
        };
    }
}

#endif

