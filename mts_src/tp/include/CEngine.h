#ifndef CENGINE_HEADER

#define CENGINE_HEADER

#include <vector>
#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include "CExchangeBroker.h"
#include "CExchange.h"
#include "IExchangeSubscriber.h"
#include "CFeed.h"
#include "IFeedSubscriber.h"
#include "CAlgorithm.h"
#include "CFIXEngine.h"
#include "IFIXSession.h"
#include "CLogBinaryBuffered.h"
#include "CLogBinary.h"
#include "CLog.h"
#include "CHeartbeat.h"
#include "CEvent.h"
#include "CEventQueue.h"
#include "CEmailer.h"
#include "CPriceMatrix.h"
#include "CMarketDataManager.h"

// market data publish
#include "md_snap.h"
#include "md_bar.h"
#include "thread_utils.h"
#include <unordered_map>

// floor
#include "TPClient.h"

#define TTBookLevel 1
namespace Mts {
    namespace Networking {
        class CTCPEngineController;
    }
}

namespace Mts {
    namespace Engine {
        class CEngine : public Mts::Exchange::IExchangeSubscriber,
                        public Mts::Feed::IFeedSubscriber,
                        public Mts::Algorithm::IAlgorithmSubscriber
        {
        public:
            CEngine(unsigned int iEngineID,
                    const std::string &strEngineName,
                    const std::string &strFIXConfig,
                    const std::string &strMarketDataQuoteLogFilename,
                    const std::string &strMarketDataTradeLogFilename,
                    unsigned int iEngineControlPort,
                    bool bNoLoggingFlag,
                    bool bSimulationMode,
                    bool bSimulationModelFullReporting,
                    unsigned int iTCPServerBufferSize,
                    unsigned int iTCPServerPort,
                    unsigned int iGuiHBMins,
                    unsigned int iGuiUpdateMSec,
                    bool bLoadPositionsOnStartup,
                    bool bLoadPositionsFromSQL,
                    bool bLoadOrderHistoryOnStartup,
                    bool bUseDateTimeEvents,
                    unsigned int iEventLoopSleepMs,
                    bool bProduction,
                    unsigned int iMktDataMgrBufferSize,
                    unsigned int iMktDataMgrUpdateMSec);

            ~CEngine();

            // starts/stops event pump
            void run();

            void startEngine();

            void shutdownEngine();

            void addFIXSession(boost::shared_ptr<Mts::FIXEngine::IFIXSession> ptrSession);

            void addExchange(boost::shared_ptr<Mts::Exchange::CExchange> ptrExchange);

            void addFeed(boost::shared_ptr<Mts::Feed::CFeed> ptrFeed);

            void addAlgorithm(boost::shared_ptr<Mts::Algorithm::CAlgorithm> ptrAlgorithm);

            void addAlgorithm(const std::string &algoName);

            // event handlers (IFeedSubscriber)
            void onFeedQuoteSim(const Mts::OrderBook::CBidAsk &objBidAsk);

            void onFeedQuoteBidAsk(const Mts::OrderBook::CBidAsk &objBidAsk);

            void onFeedQuote(const Mts::OrderBook::CQuote &objQuote);

            void onFeedTrade(const Mts::OrderBook::CTrade &objTrade);

            void onFeedHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat);

            void onFeedHeartbeat(unsigned int iProviderID,
                    double dJulianDateTime);

            void onFeedLogon(unsigned int iProviderID);

            void onFeedLogout(unsigned int iProviderID);

            void onFeedNoMoreData();

            void onFeedKeyValue(const Mts::OrderBook::CKeyValue &objKeyValue);


            // event handlers (IExchangeSubscriber)
            void onExchangeExecReport(const Mts::Order::CExecReport &objExecReport);

            void onExchangeOrderNew(const Mts::Order::COrder &objOrder);

            void onExchangeOrderStatus(const Mts::Order::COrderStatus &objOrderStatus);

            void onExchangeOrderFill(const Mts::Order::COrderFill &objOrderFill);

            void onExchangeHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat);

            void onExchangeHeartbeat(unsigned int iProviderID,
                    double dJulianDateTime);

            void onExchangeLogon(unsigned int iProviderID);

            void onExchangeLogout(unsigned int iProviderID);

            void onExchangeMessage(const std::string &strMsg);

            // event handlers (IAlgorithmSubscriber)
            void onAlgorithmMessage(const std::string &strMsg);

            void onAlgorithmPositionUpdate(unsigned int iAlgoID,
                    const Mts::Accounting::CPosition &objPosition);

            void onAlgorithmInternalError(unsigned int iAlgoID,
                    const std::string &strMsg);

            // operational
            void setActive(unsigned int iAlgoID);

            void setPassive(unsigned int iAlgoID);

            void setLiquidate(unsigned int iAlgoID);

            void setPosition(unsigned int iAlgoID,
                    const Mts::Core::CSymbol &objSymbol,
                    long iPosition,
                    double dWAP);

            void setPosition(unsigned int iAlgoID, const Mts::Core::CSymbol &objSymbol,
                    const Mts::Accounting::CPosition &objPosition);

            void sendUserCommandToAlgo(unsigned int iAlgoID, const std::string &strUserCommand);

            void sendRecoveryRequestToDropCopy(const std::string &start_utc, const std::string &end_utc,
                    const std::string &out_file);

            void aggregationIncludeProvider(unsigned int iProviderID);

            void aggregationExcludeProvider(unsigned int iProviderID);

            void initAlgorithms(const Mts::TickData::CPriceMatrix &objPriceHist5Min,
                    const Mts::TickData::CPriceMatrix &objPriceHist1Min);

            void resetAlgorithmStops();

            void saveUserProfile(const std::string &strUID,
                    const std::string &strDefaultCcys);

            void addSubscriber(const std::string &strProvider);

            void removeSubscriber(const std::string &strProvider);
            //void publishMktUpdate(const Mts::OrderBook::CBidAsk & objBidAsk);
            //void publishTrade(const Mts::OrderBook::CTrade & objTrade);

            // accessors (note, these methods are called from another thread so pass by value is used to avoid an additional synchronization overhead)
            std::vector<Mts::Order::COrderFill> getOrderFillHistory() const;

            std::vector<Mts::Order::COrderStatus> getOrderStatusHistory() const;

            std::vector<boost::shared_ptr<Mts::Algorithm::CAlgorithm> > getAlgorithms() const;

            std::vector<std::string> getAllAlgoState() const;

            unsigned int getGuiHBMins() const;

            bool isProduction() const;

            unsigned int getEngineID() const;

            std::string getEngineName() const;

            // diagnostics
            void onGUIHeartbeat();

            void checkForDroppedConnections();

            void disconnectFeedsAndExchanges();

            void connectFeedsAndExchanges();

            void dumpInboundMessages();

            void disconnectClient(const std::string &strClientName);

            // interface to engine controller .exe
            //void setTCPEngineController(boost::shared_ptr<Mts::Networking::CTCPEngineController>  ptrTCPEngineController);

            // recovery process
            void checkAndLoadRecoveryFile();

            // used by CExchange to send execution report
            void sendExecutionReport(const pm::ExecutionReport &er);

        private:
            void initializeEngine();

            void onProviderInactive(unsigned int iProviderID);

            void setAlgorithmMode(unsigned int iAlgoID, Mts::Algorithm::CAlgorithm::OperationalMode iMode);

            // Market data
            void addBookWriter(const Mts::Core::CSymbol &csymbol,
                    const std::string &level = "L1");

            void writeBook(const Mts::OrderBook::CBidAsk &quote);

            void writeBook(const Mts::OrderBook::CQuote &quote);

            void writeBook(const Mts::OrderBook::CTrade &trade);

            // Order Routing
            std::string sendOrder(const char *data);

            std::string requestReplay(const char *data, int size, std::string &ready_file);

            std::string requestOpenOrder(const char *data, int size);

            // Execution Report with Floor
            bool persistExecutionReport(const pm::ExecutionReport &er) const;

        private:
            typedef std::vector<boost::shared_ptr<Mts::Feed::CFeed> > FeedArray;
            typedef std::vector<boost::shared_ptr<Mts::Algorithm::CAlgorithm> > AlgorithmArray;
            typedef boost::unordered_map<unsigned int, boost::shared_ptr<Mts::Algorithm::CAlgorithm> > AlgorithmMap;
            typedef std::vector<Mts::Order::COrderFill> InboundOrderFillArray;
            typedef std::vector<Mts::Order::COrderStatus> InboundOrderStatusArray;
            typedef std::vector<Mts::Order::COrderStatus> RiskViolationsArray;

            Mts::FIXEngine::CFIXEngine m_FIXEngine;

            // engine instance identification
            unsigned int m_iEngineID;
            const std::string &m_strEngineName;

            // collection of execution venues running in the engine
            Mts::Exchange::CExchangeBroker &m_ExchangeBroker;

            // collection of market data feeds running in the engine
            FeedArray m_Feeds;

            // collection of algorithms running in the engine
            AlgorithmArray m_Algorithms;
            AlgorithmMap m_AlgoID2Algorithm;

            volatile bool m_bEngineShutdown;

            // logging
            std::string m_strMarketDataQuoteLogFilename;
            std::string m_strMarketDataTradeLogFilename;
            bool m_bNoLoggingFlag;
            Mts::Log::CLog m_MarketDataQuoteLogCSV;
            Mts::Log::CLog m_MarketDataTradeLogCSV;
            Mts::Log::CLogBinaryBuffered m_InboundOrderFillsLogBinary;
            Mts::Log::CLogBinaryBuffered m_InboundOrderStatusLogBinary;
            Mts::Log::CLog m_InboundPositionLogCSV;
            Mts::Log::CLog m_OutboundAllocationsLogCSV;

            // transaction/message history (both protected by locks as they may be accessed from multiple threads)
            mutable boost::mutex m_MutexOrderFillHistory;
            mutable boost::mutex m_MutexOrderStatusHistory;
            mutable boost::mutex m_MutexAllocatedTradeHistory;
            InboundOrderFillArray m_OrderFillHistory;
            InboundOrderStatusArray m_OrderStatusHistory;

            // market data manager, persists bars to the database
            Mts::TickData::CMarketDataManager m_MarketDataManager;

            // engine control interface (engctrl -> engine)
            //boost::shared_ptr<Mts::Networking::CTCPEngineController> m_ptrTCPEngineController;

            // last heartbeat from the GUI (any GUI)
            mutable boost::mutex m_MutexGuiHB;
            Mts::Core::CDateTime m_dtLastGuiHB;
            unsigned int m_iGuiHBMins;

            // throttle (millseconds) for GUI level II updates
            Mts::Core::CDateTime m_dtLastGuiUpdate[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            unsigned int m_iGuiUpdateMSec;

            // sleep duration in main event loop
            unsigned int m_iEventLoopSleepMs;

            // last heartbeat from each feed or execution venue
            Mts::LifeCycle::CHeartbeat m_ExchangeHB[Mts::Core::CConfig::MAX_NUM_PROVIDERS];
            Mts::LifeCycle::CHeartbeat m_FeedHB[Mts::Core::CConfig::MAX_NUM_PROVIDERS];
            bool m_DataOnlyProvider[Mts::Core::CConfig::MAX_NUM_PROVIDERS];

            // status of feeds and executive venues
            bool m_ExchangeActive[Mts::Core::CConfig::MAX_NUM_PROVIDERS];
            bool m_FeedActive[Mts::Core::CConfig::MAX_NUM_PROVIDERS];

            // booleans denoting which providers to include in the aggregated book
            bool m_IncludeProviderInAggregation[Mts::Core::CConfig::MAX_NUM_PROVIDERS];


            // if true, algorithm positions will be loaded from a flat file (or database) upon engine initialization otherwise they'll start flat
            bool m_bLoadPositionsOnStartup;
            bool m_bLoadPositionsFromSQL;
            bool m_bLoadOrderHistoryOnStartup;

            // flag, if true the engine will periodically send datetime events to algos
            bool m_bUseDateTimeEvents;

            // emailer
            Mts::Mail::CEmailer m_Emailer;

            // flag, true if the engine instance is for production
            bool m_bProduction;

            // flag, true if the engine start sequence completed
            bool m_bEngineStarted;
            bool m_bRecoveryDone;

            // tick-by-tick book writer indexed by symbol_id
            using BookOutputType = md::BookQ<TTBookLevel>::Writer;
            using BookWriterType = md::BookWriter<BookOutputType, TTBookLevel>;

            std::unordered_map<unsigned int, std::shared_ptr<md::BookQ<TTBookLevel>>> m_bq;
            std::unordered_map<unsigned int, std::shared_ptr<BookWriterType>> m_book_writer;

            using BarWriter = md::BarWriterLive<TTBookLevel>;
            BarWriterLive<TTBookLevel> m_bar_writer;
            utils::ThreadWrapper<BarWriterLive<TTBookLevel>> m_bar_writer_thread;

            // interface to the floor
            std::unordered_map<std::string, unsigned int> m_exch_contract_map;
            pm::FloorClientOrder<CEngine> m_floor;

            friend class pm::FloorClientOrder<CEngine>;

            FILE *m_er_fp;
        };

    }

}

#endif


