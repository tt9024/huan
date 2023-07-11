#pragma once

#include "CExchangeFIX_TT.h"
#include "CExchangeRecovery.h"

namespace Mts
{
    namespace Exchange
    {
        class CDropCopyFIX_TT : public Mts::Exchange::CExchangeFIX_TT
        {
        public:
            CDropCopyFIX_TT(
                const Mts::Core::CProvider & objProvider,
                const std::string & strSenderCompID,
                const std::string &	strTargetCompID,
                const std::string &	strUsername,
                const std::string &	strPassword,
                const std::string &	strAccount,
                unsigned int iHeartbeatsecs);

            bool isDone() const { return m_bDone; };
            void setDone(bool isDone) { m_bDone = isDone; };
            const CExchangeRecovery& getRecoveryFills() const { return m_recover; };
            void setRequest(const std::string& startLocalTime, const std::string& endLocalTime);

            static bool getOrdId(const FIX44::ExecutionReport& objMessage, std::string& ordId); 
            static bool getSymbol(const FIX44::ExecutionReport& objMessage, std::string& symbol);
            static bool isWebOrder(const FIX44::ExecutionReport &objMessage);


        protected:

            void onLogon(const FIX::SessionID & objSessionID) override;
            void onLogout(const FIX::SessionID & objSessionID) override;

            void onMessage(
                const FIX44::ExecutionReport & objMessage,
                const FIX::SessionID & objSessionID) override ;

            void onMessage(
                const FIX44::BusinessMessageReject & objMessage,
                const FIX::SessionID &objSessionID) override;

            void processExecutionReport(
                    const FIX44::ExecutionReport & objMessage);

        private:
            CExchangeRecovery m_recover;
            volatile bool m_bDone;
            void sendRecoveryRequet();
        };
    }
}
