#ifndef CFIXENGINE_HEADER

#define CFIXENGINE_HEADER

#include <boost/unordered_map.hpp>
#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include "quickfix/Application.h"
#include "quickfix/Session.h"
#include "quickfix/MessageCracker.h"
#include "IRunnable.h"
#include "IFIXSession.h"
#include "CExcept.h"

namespace Mts {
    namespace FIXEngine {
        class CFIXEngine : public FIX::Application,
                           public FIX::MessageCracker,
                           public Mts::Thread::IRunnable
        {
        public:
            CFIXEngine(const std::string &strSettingsFile);

            ~CFIXEngine() { stopFIXEngine(); };

            void addSession(boost::shared_ptr<IFIXSession> ptrSession);

            int getNumSessions() const;

            // inherited from Mts::Thread::IRunnable
            void run();

            void operator()();

            void stopFIXEngine();

            bool isShutdown() const;

        private:
            // inherited from FIX::Application
            void onCreate(const FIX::SessionID &objSessionID);

            void onLogon(const FIX::SessionID &objSessionID);

            void onLogout(const FIX::SessionID &objSessionID);

            void toAdmin(FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID);

            void toApp(FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID) EXCEPT(FIX::DoNotSend);

            void fromAdmin(const FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID) EXCEPT(FIX::FieldNotFound,
                                                               FIX::IncorrectDataFormat,
                                                               FIX::IncorrectTagValue,
                                                               FIX::RejectLogon);

            void fromApp(const FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID) EXCEPT(FIX::FieldNotFound,
                                                               FIX::IncorrectDataFormat,
                                                               FIX::IncorrectTagValue,
                                                               FIX::UnsupportedMessageType);

            void onMessage(const FIX44::MarketDataIncrementalRefresh &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::MarketDataSnapshotFullRefresh &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::MarketDataRequestReject &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::OrderCancelReject &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::SecurityList &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::SecurityDefinition &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::UserResponse &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::CollateralReport &objMessage,
                    const FIX::SessionID &objSessionID);

            void onMessage(const FIX44::News &objMessage,
                    const FIX::SessionID &objSessionID);

            boost::shared_ptr<IFIXSession> getSession(const FIX::SessionID &objSessionID);

        private:
            // SenderCompID -> session
            typedef boost::unordered_map<std::string, boost::shared_ptr<IFIXSession> > SessionMap;

            std::string m_strSettingsFile;
            SessionMap m_Sessions;
            boost::shared_ptr<boost::thread> m_ptrThread;
            volatile bool m_bRunning;
            volatile bool m_bShutdown;
        };
    }
}

#endif

