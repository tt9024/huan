#ifndef CTCPSERVER_HEADER

#define CTCPSERVER_HEADER

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/thread/thread.hpp>
#include <vector>
#include "CAlgorithm.h"
#include "CTCPConnection.h"
#include "CHeartbeat.h"
#include "CConsolidatedPositionManager.h"

using boost::asio::ip::tcp;

namespace Mts {
    namespace Engine {
        class CEngine;
    }
}

namespace Mts {
    namespace Networking {
        class CTCPServer : public Mts::Algorithm::CAlgorithm
        {
        public:
            CTCPServer(boost::shared_ptr<Mts::Engine::CEngine> ptrEngine,
                    unsigned int iTCPServerBufferSize,
                    unsigned int iTCPServerPort);

            // TCP broadcasting
            void startServer();

            void runIOService();

            void start_accept();

            void handle_accept(boost::shared_ptr<CTCPConnection> ptrConnection,
                    const boost::system::error_code &objError);

            void sendStaticData(boost::shared_ptr<CTCPConnection> ptrConnection);

            void sendTransactionHistory(boost::shared_ptr<CTCPConnection> ptrConnection);

            void sendStatusMessage(const std::string &strMessage);

            void sendGUIShutdownMessage();

            bool hasClients() const;

            void disconnectClient(const std::string &strClientName);

            void sendErrorMessage(const std::string &strMsg);

            void sendUserProfile(boost::shared_ptr<CTCPConnection> ptrConnection,
                    const std::string &strUID);

            // from CAlgorithm
            void onCreate();

            void onStart();

            void onStartOfDay(Mts::Core::CDateTime dtTimestamp);

            void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk);

            void onEvent(const Mts::Order::COrder &objOrder);

            void onEvent(const Mts::Order::COrderStatus &objOrderStatus);

            void onEvent(const Mts::Order::COrderFill &objOrderFill);

            void onEvent(const Mts::Order::CExecReport &objExecReport);

            void onEndOfDay(Mts::Core::CDateTime dtTimestamp);

            void onIncludeProvider(const Mts::Core::CProvider &objProvider);

            void onExcludeProvider(const Mts::Core::CProvider &objProvider);

            void onStop();

            void onDestroy();

            // admin
            void onHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat);

            void onAlgoState(boost::shared_ptr<CTCPConnection> ptrConnection,
                    unsigned int iAlgoID,
                    Mts::Algorithm::CAlgorithm::OperationalMode iAlgoState);

            void onAlgoState(unsigned int iAlgoID,
                    Mts::Algorithm::CAlgorithm::OperationalMode iAlgoState);

            void onAlgorithmMessage(const std::string &strMsg);

        private:
            void createOrderStatusMessage(char *pszBuffer,
                    const Mts::Order::COrderStatus &objOrderStatus);

            void createOrderFillMessage(char *pszBuffer,
                    const Mts::Order::COrderFill &objOrderFill);

            void createExecReportMessage(char *pszBuffer,
                    const Mts::Order::CExecReport &objExecReport);

        private:
            enum BroadcastMessage
            {
                STATIC_DATA_SYMBOLS = 0,
                BOOK_UPDATE = 1,
                POSITION_UPDATE = 2,
                PNL_UPDATE = 3,
                ORDER_STATUS_UPDATE = 4,
                FILL_UPDATE = 5,
                STATIC_DATA_ALGOS = 6,
                STATIC_DATA_CCYS = 7,
                CCY_POSITION_UPDATE = 8,
                ALGO_PNL_UPDATE = 9,
                LEVEL_II_UPDATE = 10,
                NEW_ORDER_UPDATE = 11,
                HEARTBEAT = 12,
                STATIC_DATA_PROVIDERS = 13,
                ALGO_STATE_UPDATE = 14,
                ALGO_MESSAGE = 15,
                STATUS_MESSAGE = 16,
                STATIC_DATA_ACCOUNTS = 17,
                PROVIDER_UPDATE = 18,
                PERFORMANCE_UPDATE = 19,
                STATIC_DATA_GUI_HB = 20,
                ERROR_MESSAGE = 21,
                ALLOCATION_STATUS = 22,
                STATIC_DATA_ALGO_INIT = 23,
                LEVEL_II_LP_UPDATE = 24,
                STATIC_DATA_TICKSIZES = 25,
                PROVIDER_ALLOCATION = 26,
                STATIC_DATA_ALLOCATION = 27,
                USER_PROFILE = 28,
                GUI_SHUTDOWN = 29,
                STATIC_DATA_ENGINE_ID = 30,
                ANALYTICS_SPREADS = 31,
                ANALYTICS_HEATMAP = 32,
                CLOSING_TRADES_UPDATE = 33,
                EXEC_REPORT = 34,
                ASMF_EXECUTION_ORDER = 35
            };

            template<typename T>
            void broadcast(const T &objDataItem,
                    BroadcastMessage iMsgType);

            void broadcast(const std::string &strMessage,
                    BroadcastMessage iMsgType);

            void broadcast(boost::shared_ptr<CTCPConnection> ptrConnection,
                    const std::string &strDataItem,
                    BroadcastMessage iMsgType);

        private:
            typedef std::vector<boost::shared_ptr<CTCPConnection> > ClientArray;
            typedef std::vector<std::string> ClientNameArray;

            boost::shared_ptr<Mts::Engine::CEngine> m_ptrEngine;

            boost::asio::io_service m_IOService;
            tcp::acceptor m_Acceptor;

            boost::mutex m_MutexClients;
            ClientArray m_Clients;
            ClientNameArray m_ClientNames;

            // might be better off in CEngine but want to keep the main thread lightweight and for messaging only
            Mts::Accounting::CConsolidatedPositionManager m_ConsolidatedPositionManager;
        };
    }
}

#include "CTCPServer.hxx"

#endif

