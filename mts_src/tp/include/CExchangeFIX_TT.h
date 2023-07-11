#ifndef CEXCHANGEFIX_TT_HEADER

#define CEXCHANGEFIX_TT_HEADER

#include <boost/lexical_cast.hpp>
#include <boost/thread/thread.hpp>
#include "CExchange.h"
#include "CProvider.h"
#include "COrder.h"
#include "COrderFill.h"
#include "COrderCancelRequest.h"
#include "COrderStatus.h"
#include "IFIXSession.h"

namespace Mts {
    namespace Exchange {
        class CExchangeFIX_TT : public Mts::Exchange::CExchange
        {
        public:
            CExchangeFIX_TT(const Mts::Core::CProvider &objProvider,
                    const std::string &strSenderCompID,
                    const std::string &strTargetCompID,
                    const std::string &strUsername,
                    const std::string &strPassword,
                    const std::string &strAccount,
                    unsigned int iHeartbeatsecs);

            void operator()();

            // CExchange overrides
            bool connect();

            bool disconnect();

            bool submitMktOrder(const Mts::Order::COrder &objOrder);

            bool submitLmtOrder(const Mts::Order::COrder &objOrder,
                    Mts::Exchange::CExchange::TimeInForce iTIF);

            bool submitTWAPOrder(const Mts::Order::COrder &objOrder);

            bool submitIcebergOrder(const Mts::Order::COrder &objOrder);

            // cancel
            bool cancelOrder(const Mts::Order::COrderCancelRequest &objCancelReq);

            bool cancelOrder(const std::string &origClOrdId, const std::string &algo);

            // replace
            bool replaceOrder(const std::string &origClOrdId, int64_t qty, double px, const std::string &algo,
                    const std::string &symbol, const std::string &newClOrdId) override;

            /*
            bool replaceOrderPxOnly(const std::string& origClOrdId, const std::string& algo, double px);
            bool replaceOrderQtyOnly(const std::string& origClOrdId, const std::string& algo, unsigned int qty);
            */

            unsigned int getProviderID() const;

        protected:
            void onCreate(const FIX::SessionID &objSessionID);

            void onLogon(const FIX::SessionID &objSessionID);

            void onLogout(const FIX::SessionID &objSessionID);

            void fromAdmin(const FIX::Message & objMessage,
                    const FIX::SessionID &objSessionID) CEXCEPT(FIX::FieldNotFound,
                                                                FIX::IncorrectDataFormat,
                                                                FIX::IncorrectTagValue,
                                                                FIX::RejectLogon);

            void toAdmin(FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::SecurityList &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::SecurityDefinition &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::MarketDataIncrementalRefresh &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::MarketDataSnapshotFullRefresh &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::MarketDataRequestReject &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::BusinessMessageReject &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::OrderCancelReject &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::OrderMassCancelReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::News &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::UserResponse &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::CollateralReport &objMessage,
                    const FIX::SessionID &objSessionID);

            std::string getSenderCompID() const;

            std::string getSessionQualifier() const;

            void queryHeader(FIX::Header &objHeader, bool set_tag50 = false);

            const std::string getStrAccount(const std::string &algo) const;

            void processExecutionReport_Fill(
                    const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void processExecutionReport_New(
                    const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void processExecutionReport_Cancel(
                    const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void processExecutionReport_Replace(
                    const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void processExecutionReport_Reject(
                    const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID);

            Mts::Order::COrder::OrderState tt2MtsOrderState(const std::string &strECNOrderState) const;

        protected:
            const Mts::Core::CProvider &m_Provider;
            FIX::SenderCompID m_SenderCompID;
            FIX::TargetCompID m_TargetCompID;
            std::string m_strUsername;
            std::string m_strPassword;
            std::string m_strAccount;

            unsigned int m_iHeartbeatsecs;
            std::string m_strTestReqID;

            Mts::Order::COrderFill m_OrderFill;
            Mts::Order::COrderStatus m_OrderStatus;

            bool m_bRecoveryComplete;

            mutable boost::mutex m_Mutex;
        };
    }
}

#endif

