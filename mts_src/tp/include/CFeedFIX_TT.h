#ifndef CFEEDFIX_TT_HEADER

#define CFEEDFIX_TT_HEADER

#include <boost/unordered_map.hpp>
#include <boost/lexical_cast.hpp>
#include "CFeed.h"
#include "CQuote.h"
#include "CTrade.h"
#include "CProvider.h"
#include "IFIXSession.h"
#include <set>

namespace Mts {
    namespace Feed {
        class CFeedFIX_TT : public Mts::Feed::CFeed,
                            public Mts::FIXEngine::IFIXSession
        {
        public:
            CFeedFIX_TT(const Mts::Core::CProvider &objProvider,
                    const std::string &strSenderCompID,
                    const std::string &strTargetCompID,
                    const std::string &strUserName,
                    const std::string &strPassword,
                    unsigned int iHeartbeMtsecs);

            void addSubscription(const Mts::Core::CSymbol &objSymbol);

            void operator()();

            // CFeed overrides
            bool connect();

            bool disconnect();

            void initialize();

        private:
            void onCreate(const FIX::SessionID &objSessionID);

            void onLogon(const FIX::SessionID &objSessionID);

            void onLogout(const FIX::SessionID &objSessionID);

            void fromAdmin(const FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID) EXCEPT(FIX::FieldNotFound,
            FIX::IncorrectDataFormat,
            FIX::IncorrectTagValue,
            FIX::RejectLogon);

            void toAdmin(FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID);

            void requestMktData(const std::string &strSymbol,
                    bool bSubscribe,
                    unsigned int iSubscriptionType);

            void requestSecurityDef(const std::string &strSymbol);

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

            void queryHeader(FIX::Header &objHeader);

        private:
            typedef std::set<std::string> SubscribedSymbols;
            typedef boost::unordered_map<std::string, unsigned int> MDEntryID2SymbolIDMap;
            typedef boost::unordered_map<std::string, Mts::OrderBook::CQuote::Side> MDEntryID2SideMap;

            const Mts::Core::CProvider &m_Provider;
            FIX::SenderCompID m_SenderCompID;
            FIX::TargetCompID m_TargetCompID;
            std::string m_strUserName;
            std::string m_strPassword;

            SubscribedSymbols m_Subscriptions;
            SubscribedSymbols m_MatchedSubscriptions;

            unsigned int m_iHeartbeMtsecs;
            std::string m_strTestReqID;

            Mts::OrderBook::CTrade m_Trade;
            Mts::OrderBook::CQuote m_QuoteBid;
            Mts::OrderBook::CQuote m_QuoteAsk;
        };
    }
}

#endif

