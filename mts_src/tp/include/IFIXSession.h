#ifndef IFIXSESSION_HEADER

#define IFIXSESSION_HEADER

#include "quickfix/fix44/ExecutionReport.h"
#include "quickfix/fix44/OrderCancelReject.h"
#include "quickfix/fix44/NewOrderSingle.h"
#include "quickfix/fix44/OrderCancelRequest.h"
#include "quickfix/fix44/OrderCancelReplaceRequest.h"
#include "quickfix/fix44/OrderMassCancelRequest.h"
#include "quickfix/fix44/OrderMassCancelReport.h"
#include "quickfix/fix44/OrderStatusRequest.h"
#include "quickfix/fix44/MarketDataRequest.h"
#include "quickfix/fix44/MarketDataIncrementalRefresh.h"
#include "quickfix/fix44/MarketDataSnapshotFullRefresh.h"
#include "quickfix/fix44/MarketDataRequestReject.h"
#include "quickfix/fix44/BusinessMessageReject.h"
#include "quickfix/fix44/SecurityListRequest.h"
#include "quickfix/fix44/SecurityList.h"
#include "quickfix/fix44/Logon.h"
#include "quickfix/fix44/Logout.h"
#include "quickfix/fix44/News.h"
#include "quickfix/fix44/UserRequest.h"
#include "quickfix/fix44/UserResponse.h"
#include "quickfix/fix44/CollateralReport.h"
#include "quickfix/fix44/CollateralInquiry.h"
#include "quickfix/fix44/SecurityDefinitionRequest.h"
#include "quickfix/fix44/SecurityDefinition.h"
#include "quickfix/fix44/SecurityStatusRequest.h"
#include "quickfix/fix44/SecurityStatus.h"

#ifdef _WIN32
#include <Windows.h>
#define sleep_milli(ms) Sleep(ms)
#else

#include <unistd.h>

#define sleep_milli(ms) usleep((ms)*1000)
#endif

namespace Mts {
    namespace FIXEngine {
        class IFIXSession
        {
        public:
            // overrides required by both feeds and venues
            virtual void onCreate(const FIX::SessionID &objSessionID) = 0;

            virtual void onLogon(const FIX::SessionID &objSessionID) = 0;

            virtual void onLogout(const FIX::SessionID &objSessionID) = 0;

            virtual void fromAdmin(const FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID) EXCEPT(FIX::FieldNotFound,
                                                               FIX::IncorrectDataFormat,
                                                               FIX::IncorrectTagValue,
                                                               FIX::RejectLogon) = 0;

            virtual void toAdmin(FIX::Message &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            // overrides for security data
            virtual void onMessage(const FIX44::SecurityList &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::SecurityDefinition &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            // overrides for market data
            virtual void onMessage(const FIX44::MarketDataRequestReject &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::MarketDataSnapshotFullRefresh &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::MarketDataIncrementalRefresh &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            // overrides for order routing
            virtual void onMessage(const FIX44::ExecutionReport &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::OrderCancelReject &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::OrderMassCancelReport &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            // misc
            virtual void onMessage(const FIX44::UserResponse &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::CollateralReport &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::BusinessMessageReject &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual void onMessage(const FIX44::News &objMessage,
                    const FIX::SessionID &objSessionID) = 0;

            virtual std::string getSenderCompID() const = 0;

            virtual std::string getSessionQualifier() const = 0;
        };
    }
}

#endif


