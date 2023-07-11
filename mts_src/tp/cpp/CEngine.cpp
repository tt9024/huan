#include <boost/thread/thread.hpp>
#include "CEngine.h"
#include "CPosition.h"
#include "CLogText.h"
#include "CAlgorithmFactory.h"
#include "CApplicationLog.h"
#include "CDateTimeEvent.h"
#include "COrderFactory.h"
#include "CStringTokenizer.h"
#include <unistd.h>

#include "RiskMonitor.h"

using namespace Mts::Engine;


CEngine::CEngine(unsigned int                   iEngineID,
                 const std::string &    strEngineName,
                 const std::string &    strFIXConfig,
                 const std::string &    strMarketDataQuoteLogFilename,
                 const std::string &    strMarketDataTradeLogFilename,
                 unsigned int                   iEngineControlPort,
                 bool                                   bNoLoggingFlag,
                 bool                                   bSimulationMode,
                 bool                                   bSimulationModelFullReporting,
                 unsigned int                   iTCPServerBufferSize,
                 unsigned int                   iTCPServerPort,
                 unsigned int                   iGuiHBMins,
                 unsigned int                   iGuiUpdateMSec,
                 bool                                   bLoadPositionsOnStartup,
                 bool                                   bLoadPositionsFromSQL,
                 bool                                   bLoadOrderHistoryOnStartup,
                 bool                                   bUseDateTimeEvents,
                 unsigned int                   iEventLoopSleepMs,
                 bool                                   bProduction,
                 unsigned int                   iMktDataMgrBufferSize,
                 unsigned int                   iMktDataMgrUpdateMSec)
: m_iEngineID(iEngineID),
    m_strEngineName(strEngineName),
    m_FIXEngine(strFIXConfig),
    m_strMarketDataQuoteLogFilename(strMarketDataQuoteLogFilename),
    m_strMarketDataTradeLogFilename(strMarketDataTradeLogFilename),
    m_ExchangeBroker(Mts::Exchange::CExchangeBroker::getInstance()),
    m_bNoLoggingFlag(bNoLoggingFlag),
    m_iGuiHBMins(iGuiHBMins),
    m_iGuiUpdateMSec(iGuiUpdateMSec),
    m_bLoadPositionsOnStartup(bLoadPositionsOnStartup),
    m_bLoadPositionsFromSQL(bLoadPositionsFromSQL),
    m_bLoadOrderHistoryOnStartup(bLoadOrderHistoryOnStartup),
    m_bUseDateTimeEvents(bUseDateTimeEvents),
    m_iEventLoopSleepMs(iEventLoopSleepMs),
    m_bProduction(bProduction),
    m_MarketDataManager(iMktDataMgrBufferSize, iMktDataMgrUpdateMSec, false),
    m_bEngineStarted(false),
    m_bRecoveryDone(false),
    m_bar_writer(),
    m_bar_writer_thread(m_bar_writer),
    m_floor("tp", *this),
    m_er_fp(fopen(pm::ExecutionReport::ERPersistFile().c_str(), "a"))

{
    if (!m_er_fp) {
        AppLogError("Cannot open file %s for append!", plcc_getString("ERPersistFile").c_str());
        throw std::runtime_error(std::string("Cannot open Exectuion Report Persist File ") +
            plcc_getString("ERPersistFile") + std::string("for write!"));
    }

    initializeEngine();

    for (auto& provider : m_DataOnlyProvider) {
        provider = false;
    }
}


CEngine::~CEngine() {

    shutdownEngine();
}


void CEngine::startEngine()
{
    AppLog("Starting engine");
    AppLog("Initializing algorithms");

    // initialize algorithms
    //AlgorithmArray::iterator iter = m_Algorithms.begin();

    /*
    for (; iter != m_Algorithms.end(); ++iter) {
        (*iter)->onStart();
        (*iter)->run();
    }
    */

    // this is for GUI connection
    /*
    AppLog("Initializing GUI connection");

    m_TCPServer.onStart();
    m_TCPServer.run();
    m_TCPServer.startServer();
    */

    // initializing market data manager
    // check the symbol id map using the price matrix
    //AppLog("Initializing market data manager");
    //m_MarketDataManager.onStart();
    //m_MarketDataManager.run();

    // initialize venues
    AppLog("Initializing exchanges");

    // this is currently a noop, since each CExchange doesn't run a thread
    m_ExchangeBroker.initiateAllExchanges();  

    // initialize FIX connections to feeds
    AppLog("Initializing feeds");

    for (auto iterFd : m_Feeds) {
        iterFd->run();
    }

    //FeedArray::iterator iterFd = m_Feeds.begin();
    //for (; iterFd != m_Feeds.end(); ++iterFd)
    //    (*iterFd)->run();

    // setup for order routing for all subscribed symbols (in tradable)
    const auto& msv_all(utils::SymbolMapReader::get().getSubscriptions("").first);
    for (const auto& ms: msv_all) {
        const Mts::Core::CSymbol & csymbol = Mts::Core::CSymbol::getSymbol(ms);
        unsigned int symbol_id = csymbol.getSymbolID();
        const std::string& symbol = csymbol.getContractExchSymbol();
        m_exch_contract_map[symbol] = symbol_id;
    }

    // creates book writers and bar writers for the primary feed subscriptions
    // see also CFeedFactory.cpp:createFeedFIX_TT()
    const auto& msv_pair(utils::SymbolMapReader::get().getSubscriptions("TTFix"));
    const auto& msv(msv_pair.first);
    for (const auto& ms : msv) {
        const Mts::Core::CSymbol & objSymbol = Mts::Core::CSymbol::getSymbol(ms);
        addBookWriter(objSymbol);
    }

    // key by the "symbol" in symbol definition
    // boost::unordered_map<std::string, boost::shared_ptr<CSymbol> >   CSymbol::m_SymbolsByName;
    // boost::unordered_map<unsigned int, boost::shared_ptr<CSymbol> >  CSymbol::m_SymbolsByID;
    //

    // this kicks of all connections defined in fix config, connections and logon
    // For feeds, it requests security def upon logon
    m_FIXEngine.run();

    // For feeds, request market data subscription
    for (auto iterFd : m_Feeds) {
        iterFd->initialize();
    }
    //for (iterFd = m_Feeds.begin(); iterFd != m_Feeds.end(); ++iterFd)
    //    (*iterFd)->initialize();



    // configure market data logs

    /*
    Mts::Core::CDateTime dtNow = Mts::Core::CDateTime::now();
    char szBuffer[255];

    sprintf(szBuffer, "%s_%llu.txt", m_strMarketDataQuoteLogFilename.c_str(), dtNow.getCMTime());
    m_MarketDataQuoteLogCSV.setLogFile(szBuffer);
    m_MarketDataQuoteLogCSV.startLogging();

    sprintf(szBuffer, "%s_%llu.txt", m_strMarketDataTradeLogFilename.c_str(), dtNow.getCMTime());
    m_MarketDataTradeLogCSV.setLogFile(szBuffer);
    m_MarketDataTradeLogCSV.startLogging();


    // configure inbound fills and order status logs

    std::string strLogDir = Mts::Core::CConfig::getInstance().getLogFileDirectory();

    sprintf(szBuffer, "%s//InboundOrderFillLog.dat", strLogDir.c_str());
    m_InboundOrderFillsLogBinary.setLogFile(szBuffer);
    m_InboundOrderFillsLogBinary.startLogging();

    sprintf(szBuffer, "%s//InboundOrderStatusLog.dat", strLogDir.c_str());
    m_InboundOrderStatusLogBinary.setLogFile(szBuffer);
    m_InboundOrderStatusLogBinary.startLogging();

    sprintf(szBuffer, "%s//InboundPositionLog.txt", strLogDir.c_str());
    m_InboundPositionLogCSV.setLogFile(szBuffer);
    m_InboundPositionLogCSV.startLogging();

    // configure outbound allocations log
    sprintf(szBuffer, "%s//OutboundAllocationsLog.txt", strLogDir.c_str());
    m_OutboundAllocationsLogCSV.setLogFile(szBuffer);
    m_OutboundAllocationsLogCSV.startLogging();

    */

    pm::risk::Monitor::get().set_instance_name("CEngine");
    m_bEngineStarted = true;

    //m_Emailer.sendEmail("**** TRADING ALERT ****", "Engine started");
}


void CEngine::run() {

    AppLog("Engine awaiting command");

    AppLog("Engine Starting");
    startEngine();

    // start the bar writer
    m_bar_writer_thread.run(NULL);

    //m_ptrTCPEngineController->startServer();
    m_bEngineShutdown = false;
    time_t cur_utc = utils::TimeUtil::cur_utc();
    const time_t read_pause_itvl_sec = 10;
    while (!m_bEngineShutdown) {

        // run the floor
        if(!m_floor.run_one_loop(m_floor)) {
            utils::TimeUtil::micro_sleep(1000);

            time_t cur_utc0 = utils::TimeUtil::cur_utc();
            if (__builtin_expect(cur_utc0-cur_utc>read_pause_itvl_sec,0)) {
                pm::risk::Monitor::get().status().statusLoop();
                cur_utc = cur_utc0;
            }
        }

        //boost::this_thread::sleep(boost::posix_time::milliseconds(m_iEventLoopSleepMs));

        // there is no need to check anything here
        // quickfix retries, algo monitors market data
        //checkForDroppedConnections();

        /*
        if (m_bUseDateTimeEvents == true) {
            Mts::Core::CDateTime dtNow;
            Mts::Core::CDateTimeEvent objDateTime(dtNow);

            AlgorithmArray::iterator iter = m_Algorithms.begin();

            for (; iter != m_Algorithms.end(); ++iter) {
                (*iter)->onDateTime(objDateTime);
            }
        }
        */
    }
}

void CEngine::addFIXSession(boost::shared_ptr<Mts::FIXEngine::IFIXSession> ptrSession) {
    m_FIXEngine.addSession(ptrSession);
}

void CEngine::addExchange(boost::shared_ptr<Mts::Exchange::CExchange> ptrExchange) {

    m_ExchangeBroker.addExchange(ptrExchange->getProviderID(), ptrExchange);
    ptrExchange->addSubscriber(this);

    if (ptrExchange->isFIXProtocol()) {
        //note CExchange is subclass of IFIXSession
        //IFIXSession is used to get FIXEngine callback
        //CExchange adds to it with order sending capabilities
        addFIXSession(ptrExchange);
    }
}


void CEngine::addFeed(boost::shared_ptr<Mts::Feed::CFeed> ptrFeed) {

    m_Feeds.push_back(ptrFeed);
    ptrFeed->addSubscriber(this);

    if (ptrFeed->isFIXProtocol()) {

        addFIXSession(boost::dynamic_pointer_cast<Mts::FIXEngine::IFIXSession>(ptrFeed));
    }
}

void CEngine::addAlgorithm(boost::shared_ptr<Mts::Algorithm::CAlgorithm> ptrAlgorithm) {

    m_Algorithms.push_back(ptrAlgorithm);
    ptrAlgorithm->addAlgorithmSubscriber(this);
    ptrAlgorithm->onCreate();

    m_AlgoID2Algorithm[ptrAlgorithm->getAlgoID()] = ptrAlgorithm;
}

void CEngine::addAlgorithm(const std::string& algoName) {
    const std::string algoFile = MTSConfigInstance.getAlgoConfigFile(algoName);
    boost::shared_ptr<Mts::Algorithm::CAlgorithm> ptrAlgo = \
        Mts::Algorithm::CAlgorithmFactory::getInstance().createAlgorithm(algoFile);
    addAlgorithm(ptrAlgo);
}

// Events coming from algorithm
void CEngine::onAlgorithmMessage(const std::string & strMsg) {
    AppLog("On Algo Message: %s", strMsg.c_str());
    //m_TCPServer.onAlgorithmMessage(strMsg);
}


void CEngine::onAlgorithmPositionUpdate(
    unsigned int iAlgoID,    
    const Mts::Accounting::CPosition & objPosition
) {
    char szBuffer[255];
    sprintf(szBuffer, "%d,%s", iAlgoID, objPosition.toString().c_str());
    m_InboundPositionLogCSV.log(szBuffer);
}


// this handler is intended for use with a simulated feed and exchange
void CEngine::onFeedQuoteSim(const Mts::OrderBook::CBidAsk & objBidAsk) {

    //publishMktUpdate(objBidAsk);
}

void CEngine::onFeedQuoteBidAsk(const Mts::OrderBook::CBidAsk & objBidAsk) {
    writeBook(objBidAsk);

    /*
    unsigned int iProviderID = objBidAsk.getBid().getProviderID();
    AppLog("Got Quote: %s", objBidAsk.toString().c_str());

    // only process quotes we can actually trade against, if venue is down but data feed is ok, ignore quote
    if ((m_DataOnlyProvider[iProviderID] == true || (m_FeedActive[iProviderID] == true && m_ExchangeActive[iProviderID] == true)) &&
             m_IncludeProviderInAggregation[iProviderID] == true) {

        //publishMktUpdate(objBidAsk);
    } else {
        AppLog("But not publishing!");
    }


    //onFeedHeartbeat(iProviderID, objBidAsk.getBid().getMtsTimestamp().getValue());

    if (m_bNoLoggingFlag == false) {

        m_MarketDataQuoteLogCSV.log(objBidAsk.getBid().toString());
        m_MarketDataQuoteLogCSV.log(objBidAsk.getAsk().toString());
    }
    */
}

void CEngine::onFeedQuote(const Mts::OrderBook::CQuote & objQuote) {
    writeBook(objQuote);
}


void CEngine::onFeedTrade(const Mts::OrderBook::CTrade & objTrade) {
    writeBook(objTrade);

    // trade event propagation to algorithms is disabled for now, uncomment if needed
    //publishTrade(objTrade);
    //
    //AppLog("Got Trade: %s", objTrade.toString().c_str());
    /*
    if (m_bNoLoggingFlag == false) {
        m_MarketDataTradeLogCSV.log(objTrade.toString());
        m_MarketDataManager.onTrade(objTrade);
    }
    */
}

// FIXME right now the only place
// publishes keyvalue is feedPollDB
void CEngine::onFeedKeyValue(const Mts::OrderBook::CKeyValue & objKeyValue) {

    AppLog("Got KeyValue: %s", objKeyValue.toString().c_str());

    /*
    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter) {
        bool bRet = (*iter)->onKeyValue(objKeyValue);

        if (bRet == false)
            setPassive((*iter)->getAlgoID());
    }
    */
}

/*
void CEngine::publishTrade(const Mts::OrderBook::CTrade & objTrade) {

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter) {
        if ((*iter)->isTradedInstrument(objTrade.getSymbolID()))
        { 
            //if (m_bSimulationMode == true)
            //  (*iter)->onEvent(objTrade);
            //else {
                bool bRet = (*iter)->onTrade(objTrade);

                if (bRet == false)
                    setPassive((*iter)->getAlgoID());
            //}
        }
    }
}
*/

/*
void CEngine::publishMktUpdate(const Mts::OrderBook::CBidAsk & objBidAsk) {

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter) {
        if ((*iter)->isTradedInstrument(objBidAsk.getSymbolID()))
        {
            //if (m_bSimulationMode == true)
            //    (*iter)->onEvent(objBidAsk);
            //else {

                // Note this is on socket thread!
                bool bRet = (*iter)->onMktBidAsk(objBidAsk);

                if (bRet == false)
                    setPassive((*iter)->getAlgoID());
            //}
        }
    }

    m_ExchangeBroker.feedTestExchanges(objBidAsk);

    FeedArray::iterator iterFd = m_Feeds.begin();

    for (; iterFd != m_Feeds.end(); ++iterFd) {
        (*iterFd)->setHeartbeat(objBidAsk.getTimestamp());
    }

    // flat file logging, database updates, GUI updates
    if (m_bNoLoggingFlag == false) {

        static double GUI_UPDATE_INTERVAL_DAYFRAC = static_cast<double>(m_iGuiUpdateMSec) / (24.0 * 60.0 * 60.0 * 1000.0);

        if (objBidAsk.getTimestamp().getValue() - m_dtLastGuiUpdate[objBidAsk.getSymbolID()].getValue() >= GUI_UPDATE_INTERVAL_DAYFRAC) {

            m_dtLastGuiUpdate[objBidAsk.getSymbolID()] = objBidAsk.getTimestamp();
            //m_TCPServer.onMktBidAsk(objBidAsk);
            m_MarketDataManager.onMktBidAsk(objBidAsk);
        }
    }
}
*/

bool CEngine::persistExecutionReport(const pm::ExecutionReport& er) const
{
    return utils::CSVUtil::write_file(er.toCSVLine(), m_er_fp);
}

void CEngine::sendExecutionReport(const pm::ExecutionReport& er) {

    // 
    // debug for test risk
    //
    //pm::ExecutionReport er(er0);
    //er.m_qty = 10000;

    pm::FloorBase::MsgType msg (
            pm::FloorBase::ExecutionReport,
            (const char*)&er,
            sizeof(pm::ExecutionReport)
        );

    // this is not blocking
    m_floor.sendMsg(msg);

    // check tp's report side without pm
    const bool do_pause_notify = true;
    if (__builtin_expect(!pm::risk::Monitor::get().updateER(er, do_pause_notify), 0)) {
        AppLogError("RiskMonitor %s:%s failed at FeedHandler on updateER() for incoming er(%s), pause notified",
                er.m_algo, er.m_symbol, er.toString().c_str());
    };
    AppLog("Sent %s", er.toString().c_str());
    if (!persistExecutionReport(er)) {
        AppLogError("Failed to persist the execution report " + er.toString());
    }
}

void CEngine::onExchangeOrderNew(const Mts::Order::COrder & objOrder) {
    /*
    AppLog("Got OrderNew %s", objOrder.toString().c_str());

    const std::string& clOrdId (objOrder.getMtsOrderID());
    const std::string& execId (objOrder.getExcExecID());
    const std::string& algo (objOrder.getOrderTag());
    const auto& csymbol (Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID()));
    const std::string& exch_symbol (csymbol.getContractExchSymbol());

    bool isBuy ( objOrder.getDirection() == Mts::Order::COrder::BUY);
    unsigned int qty (objOrder.getQuantity());

    // note the price in COrder doesn't have TT's multiplier
    // it's the normal price.
    double px (objOrder.getPrice());

    // TODO - add TransactTime parse in COrder
    uint64_t new_micro = utils::TimeUtil::cur_micro();

    pm::ExecutionReport er_new (
            exch_symbol, // symbol
            algo,        // algo
            clOrdId,     // tag 11 clOrdId
            execId,      // execId, together with tag 11 form a hash in pm
            "0",  // tag39 = new
            (int)qty * (isBuy? 1:-1), // qty + or -
            px, // px
            utils::TimeUtil::frac_UTC_to_string(new_micro/1000ULL, 3, NULL, true), // exchange time gmt
            "",  // additional tag
            new_micro  // receive local micro
    );

    sendExecutionReport(er_new);
    */
    AppLogError("This should never be called! Got OrderNew %s", objOrder.toString().c_str());
}

// it is called when cancel comes in
void CEngine::onExchangeOrderStatus(const Mts::Order::COrderStatus & objOrderStatus) {
    /*
    AppLog("Got OrderStatus: %s", objOrderStatus.toString().c_str());

    const auto& objOrder (objOrderStatus.getOrigOrder());
    const std::string& clOrdId (objOrder.getMtsOrderID());
    const std::string& execId (objOrder.getExcExecID());
    const std::string& algo (objOrder.getOrderTag());
    const auto& csymbol (Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID()));
    const std::string& exch_symbol (csymbol.getContractExchSymbol());

    // TODO - add TransactTime parse in COrder
    uint64_t new_micro = utils::TimeUtil::cur_micro();

    // assert the status is cancel
    if (objOrder.getOrderState() != Mts::Order::COrder::CANCELLED) {
        AppLogError("Recieved OrderStatus on non-cancel");
        return;
    }


    pm::ExecutionReport er_cancel (
            exch_symbol, // symbol
            algo,        // algo
            clOrdId,     // tag 11 clOrdId
            execId,      // execId, together with tag 11 form a hash in pm
            "4",         // tag39 = cancel
            0,           // qty + or -
            0,           // px
            utils::TimeUtil::frac_UTC_to_string(new_micro/1000ULL, 3, NULL, true), // exchange time gmt
            "",  // additional tag
            new_micro  // receive local micro
    );

    sendExecutionReport(er_cancel);
    */
    AppLogError("This should never be called! Got OrderStatus: %s", objOrderStatus.toString().c_str());
}

void CEngine::onExchangeOrderFill(const Mts::Order::COrderFill & objOrderFill) {
    /*
    AppLog("Got OrderFill: %s", objOrderFill.toString().c_str());

    const auto& objOrder (objOrderFill.getOrigOrder());
    const std::string& clOrdId (objOrder.getMtsOrderID());
    const std::string& execId (objOrder.getExcExecID());
    const std::string& algo (objOrder.getOrderTag());
    const auto& csymbol (Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID()));
    const std::string& exch_symbol (csymbol.getContractExchSymbol());

    // TODO - add TransactTime parse in COrder
    uint64_t new_micro = utils::TimeUtil::cur_micro();

    bool isBuy ( objOrder.getDirection() == Mts::Order::COrder::BUY);
    int qty = (int) objOrderFill.getFillQuantity() * (isBuy?1:-1);
    double px = objOrderFill.getFillPrice();

    // assert the status is cancel
    std::string tag39 = "2";
    if (objOrder.getOrderState() == Mts::Order::COrder::PARTIALLY_FILLED) {
        tag39 = "1";
    }

    pm::ExecutionReport er_fill (
            exch_symbol, // symbol
            algo,        // algo
            clOrdId,     // tag 11 clOrdId
            execId,      // execId, together with tag 11 form a hash in pm
            tag39,       // tag39 = fill/partial_fill
            qty,           // qty + or -
            px,           // px
            utils::TimeUtil::frac_UTC_to_string(new_micro/1000ULL, 3, NULL, true), // exchange time gmt
            "",  // additional tag
            new_micro  // receive local micro
    );

    sendExecutionReport(er_fill);
    */
    AppLogError("This should never be called! Got OrderFill: %s", objOrderFill.toString().c_str());
}

void CEngine::onExchangeExecReport(const Mts::Order::CExecReport & objExecReport)
{
    AppLog("Got ExecReport: %s", objExecReport.toString().c_str());
}

void CEngine::onFeedLogon(unsigned int iProviderID)
{
    AppLog("onFeedLogon provider=%u", iProviderID);
    /*
    m_FeedActive[iProviderID] = true;
    if (m_ExchangeActive[iProviderID] == false)
        return;

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        (*iter)->onProviderActive(iProviderID);

    const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider(iProviderID);
    m_Emailer.sendEmail("**** TRADING ALERT ****", "Feed + Exchange up - " + objProvider.getName());
    */
}

void CEngine::onFeedLogout(unsigned int iProviderID)
{
    AppLog("onFeedLogout provider=%u", iProviderID);

    /*
    if (m_FeedActive[iProviderID] == true) {

        const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider(iProviderID);
        m_Emailer.sendEmail("**** TRADING ALERT ****", "Feed down - " + objProvider.getName());
    }

    m_FeedActive[iProviderID] = false;

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        (*iter)->onProviderInactive(iProviderID);

    onProviderInactive(iProviderID);
    */
}

void CEngine::onExchangeLogon(unsigned int iProviderID)
{

    AppLog("onExchangeLogon provider=%u", iProviderID);

    /*
    m_ExchangeActive[iProviderID] = true;

    if (m_FeedActive[iProviderID] == false)
        return;

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        (*iter)->onProviderActive(iProviderID);

    const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider(iProviderID);
    m_Emailer.sendEmail("**** TRADING ALERT ****", "Feed + Exchange up - " + objProvider.getName());
    */
}


void CEngine::onExchangeLogout(unsigned int iProviderID)
{
    AppLog("onExchangeLogout %d=iProviderID", iProviderID);

    /*
    if (m_ExchangeActive[iProviderID] == true) {

        const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider(iProviderID);
        m_Emailer.sendEmail("**** TRADING ALERT ****", "Exchange down - " + objProvider.getName());
    }

    m_ExchangeActive[iProviderID] = false;

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        (*iter)->onProviderInactive(iProviderID);

    onProviderInactive(iProviderID);
    */
}


void CEngine::initializeEngine() {

    for (auto i = 0u; i != Mts::Core::CConfig::MAX_NUM_PROVIDERS; ++i) {
        m_FeedHB[i].setProviderID(i);
        m_FeedHB[i].setIsFeed(true);
        m_ExchangeHB[i].setProviderID(i);
        m_ExchangeHB[i].setIsFeed(false);
        m_FeedActive[i] = false;
        m_ExchangeActive[i] = false;
        m_IncludeProviderInAggregation[i] = true;
    }
}


void CEngine::shutdownEngine() {

    if ( (m_bEngineStarted == false) || m_bEngineShutdown ) {
        return;
    }
    m_bEngineShutdown = true;
    m_bEngineStarted = false;
    m_bar_writer.stop();
    utils::TimeUtil::micro_sleep(100000);  

    std::cout << "Disconnecting feeds and exchanges" << std::endl;

    try {
        disconnectFeedsAndExchanges();
    } catch (const std::exception& e) {
        AppLogError("problem disconnect feeds and exchange: %s", e.what());
    }

    std::cout << "Stopping FIX engine" << std::endl;

    try {
        m_FIXEngine.stopFIXEngine();
    } catch (const std::exception& e) {
        AppLogError("Problem shutting down FIXEngine: %s", e.what());
    }

    utils::TimeUtil::micro_sleep(100000);
    if (m_er_fp) {
        fclose(m_er_fp);
        m_er_fp = nullptr;
    }

}


void CEngine::onFeedHeartbeat(unsigned int iProviderID,    double dJulianDateTime) {
    m_FeedHB[iProviderID].setJulianDateTime(dJulianDateTime);
}


void CEngine::onFeedHeartbeat(const Mts::LifeCycle::CHeartbeat & objHeartbeat) {
    m_FeedHB[objHeartbeat.getProviderID()].setJulianDateTime(objHeartbeat.getJulianDateTime());
}


void CEngine::onExchangeHeartbeat(unsigned int iProviderID, double dJulianDateTime) {
    m_ExchangeHB[iProviderID].setJulianDateTime(dJulianDateTime);
}


void CEngine::onExchangeHeartbeat(const Mts::LifeCycle::CHeartbeat & objHeartbeat) {
    m_ExchangeHB[objHeartbeat.getProviderID()].setJulianDateTime(objHeartbeat.getJulianDateTime());
}


/*void CEngine::setTCPEngineController(boost::shared_ptr<Mts::Networking::CTCPEngineController> ptrTCPEngineController) {
    m_ptrTCPEngineController = ptrTCPEngineController;
}*/


void CEngine::setActive(unsigned int iAlgoID) {
    AppLog("CEngine::setActive");
    setAlgorithmMode(iAlgoID, Mts::Algorithm::CAlgorithm::ACTIVE);
}


void CEngine::setLiquidate(unsigned int iAlgoID) {
    AppLog("CEngine::setLiquidate");
    setAlgorithmMode(iAlgoID, Mts::Algorithm::CAlgorithm::LIQUIDATE);
}


void CEngine::setPassive(unsigned int iAlgoID) {
    AppLog("CEngine::setPassive");
    setAlgorithmMode(iAlgoID, Mts::Algorithm::CAlgorithm::PASSIVE);
}


void CEngine::setAlgorithmMode(unsigned int iAlgoID, 
    Mts::Algorithm::CAlgorithm::OperationalMode iMode) 
{
    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        if ((*iter)->getAlgoID() == iAlgoID || iAlgoID == -1) {
            (*iter)->setOperationalMode(iMode);
            //m_TCPServer.onAlgoState((*iter)->getAlgoID(), iMode);
        }
}

void CEngine::onAlgorithmInternalError(unsigned int iAlgoID, const std::string & strMsg) {
    setPassive(iAlgoID);
    m_Emailer.sendEmail("**** TRADING ALERT ****", strMsg);
}

void CEngine::sendUserCommandToAlgo(unsigned int iAlgoID, const std::string & strUserCommand) {
    AppLog(std::string("CEngine::sendUserCommandToAlgo ") + strUserCommand);
    //m_TCPServer.sendStatusMessage("Executing user command: " + strUserCommand);

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    // this runs on TCPServer's thread, should just queue an event 
    // to be picked up by algo in its thread with necessary context
    for (; iter != m_Algorithms.end(); ++iter)
        if ((*iter)->getAlgoID() == iAlgoID || iAlgoID == -1)
            (*iter)->onCommand(Mts::OrderBook::CManualCommand(Mts::Core::CDateTime::now(), strUserCommand));
}

void CEngine::sendRecoveryRequestToDropCopy(const std::string& startLocalTime, const std::string& endLocalTime, const std::string& out_file) {

    AppLog(std::string("CEngine::sendRecoveryRequestToDropCopy started! From ") + startLocalTime + " to " + endLocalTime + " dump to " + out_file);

#ifdef USE_TT_RECOVERY
    // This runs on the CEngine's main thread from running FloorClient
    AppLog("Using TT Recovery Service!")
    const std::string dropcopy_fix_config = CConfig::getInstance().getDropCopyFIXConfigFile();
    const std::string dropcopy_TT_config = CConfig::getInstance().getDropCopySessionConfigFile();
    auto exc = Mts::Exchange::CDropCopyRecovery(dropcopy_fix_config, dropcopy_TT_config);
    exc.getToFile(startLocalTime, endLocalTime, out_file);
#else
    // This reads the ERPersistFile and dump each fills
    AppLog(std::string("Using persist from ") + pm::ExecutionReport::ERPersistFile());
    pm::ExecutionReport::loadFromPersistence(startLocalTime, endLocalTime, out_file);
#endif

    AppLog(std::string("CEngine::sendRecoveryRequestToDropCopy from ") + startLocalTime + " to " + endLocalTime + " finished! Saved to " + out_file);
}



void CEngine::resetAlgorithmStops() {

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        (*iter)->resetStop();
}


void CEngine::setPosition(
        unsigned int iAlgoID,
        const Mts::Core::CSymbol &  objSymbol,
        long iPosition,
        double dWAP) 
{

    Mts::Accounting::CPosition objNewPos(objSymbol, iPosition, dWAP);
    setPosition(iAlgoID, objSymbol, objNewPos);
}


void CEngine::setPosition(unsigned int iAlgoID,
                          const Mts::Core::CSymbol & objSymbol,
                          const Mts::Accounting::CPosition & objPosition) {

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        if ((*iter)->getAlgoID() == iAlgoID) {
            (*iter)->setPosition(objSymbol, objPosition);
            return;
        }
}


void CEngine::initAlgorithms(const Mts::TickData::CPriceMatrix & objPriceHist5Min, 
    const Mts::TickData::CPriceMatrix & objPriceHist1Min) {

    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter) {

        // some algos required only 5 min data, otherwise both
        (*iter)->onEngineStart(objPriceHist5Min);
        (*iter)->onEngineStart(objPriceHist5Min, objPriceHist1Min);
    }
}


void CEngine::checkForDroppedConnections() {

    static double HB_INTERVAL_DAYFRAC = 1.0 / 1440.0;

    Mts::Core::CDateTime dtNow = Mts::Core::CDateTime::now();

    for (int i = 0; i != Mts::Core::CConfig::MAX_NUM_PROVIDERS; ++i) {

        // bit of a hack, assume we'll get one heartbeat
        if (m_FeedHB[i].getJulianDateTime() > 0) {

            double dDelta = dtNow.getValue() - m_FeedHB[i].getJulianDateTime();

            if (dDelta > HB_INTERVAL_DAYFRAC) {
                m_FeedHB[i].setActiveFlag(false);
                //m_TCPServer.onHeartbeat(m_FeedHB[i]);
            }
            else {
                m_FeedHB[i].setActiveFlag(true);
                //m_TCPServer.onHeartbeat(m_FeedHB[i]);
            }
        }

        // bit of a hack, assume we'll get one heartbeat
        if (m_ExchangeHB[i].getJulianDateTime() <= 0)
            continue;

        double dDelta = dtNow.getValue() - m_ExchangeHB[i].getJulianDateTime();

        if (dDelta > HB_INTERVAL_DAYFRAC) {
            m_ExchangeHB[i].setActiveFlag(false);
            //m_TCPServer.onHeartbeat(m_ExchangeHB[i]);
        }
        else {
            m_ExchangeHB[i].setActiveFlag(true);
            //m_TCPServer.onHeartbeat(m_ExchangeHB[i]);
        }
    }


    // if GUI HB interval is set to -1 we aren't worried about a human monitoring risk (e.g. when running UAT/SIM)
    if (m_iGuiHBMins == 0)
        return;

    // check for GUI HB to ensure someone is monitoring the risk (no contact for a duration of 2x expected HB will trigger PASSIVE mode)
    static double   HB_GUI_DAY_FRAC = static_cast<double>(m_iGuiHBMins * 2) / 1440.0;
    bool                    bNoGuiHB                = false;

    {
        boost::mutex::scoped_lock lock(m_MutexGuiHB);

        bNoGuiHB            = m_dtLastGuiHB.getValue() > 0 && dtNow.getValue() - m_dtLastGuiHB.getValue() > HB_GUI_DAY_FRAC;
    }

    // set all algorithms to PASSIVE if we lose HB from GUI
    if (bNoGuiHB == true) {

        AppLogError("Lost contact with GUI. Setting all algorithms to PASSIVE");
        setPassive(-1);
    }
}


std::vector<Mts::Order::COrderFill> CEngine::getOrderFillHistory() const {

    boost::mutex::scoped_lock lock(m_MutexOrderFillHistory);
    return m_OrderFillHistory;
}


std::vector<Mts::Order::COrderStatus> CEngine::getOrderStatusHistory() const {

    boost::mutex::scoped_lock lock(m_MutexOrderStatusHistory);
    return m_OrderStatusHistory;
}


std::vector<boost::shared_ptr<Mts::Algorithm::CAlgorithm> > CEngine::getAlgorithms() const {

    return m_Algorithms;
}


void CEngine::disconnectFeedsAndExchanges() {

    FeedArray::iterator iterFd = m_Feeds.begin();

    for (; iterFd != m_Feeds.end(); ++iterFd) {
        (*iterFd)->disconnect();
    }

    m_ExchangeBroker.disconnectFromAllExchanges();
}


void CEngine::connectFeedsAndExchanges() {

    FeedArray::iterator iterFd = m_Feeds.begin();

    for (; iterFd != m_Feeds.end(); ++iterFd) {
        (*iterFd)->connect();
    }

    m_ExchangeBroker.connectToAllExchanges();
}


// called if connectivity to an ECN is lost, algorithms must handle their own state update, engine will send a state update to the GUI
void CEngine::onProviderInactive(unsigned int iProviderID) {
    std::cout << "Provider Inactive Called " << std::endl;
}


void CEngine::dumpInboundMessages() {

    Mts::Log::CLogText objLogOrderStatus("InboundOrderStatus.txt");

    for (size_t i = 0; i != m_OrderStatusHistory.size(); ++i) {

        objLogOrderStatus.log(m_OrderStatusHistory[i].toString());
    }

    Mts::Log::CLogText objLogOrderFill("InboundOrderFill.txt");

    for (size_t i = 0; i != m_OrderFillHistory.size(); ++i) {

        objLogOrderFill.log(m_OrderFillHistory[i].toString());
    }
}


void CEngine::aggregationIncludeProvider(unsigned int iProviderID) {

    m_IncludeProviderInAggregation[iProviderID] = true;

    const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider(iProviderID);
    //m_TCPServer.onIncludeProvider(objProvider);
}


void CEngine::aggregationExcludeProvider(unsigned int iProviderID) {

    m_IncludeProviderInAggregation[iProviderID] = false;

    const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider(iProviderID);
    //m_TCPServer.onExcludeProvider(objProvider);
}


void CEngine::onGUIHeartbeat() {

    boost::mutex::scoped_lock lock(m_MutexGuiHB);

    m_dtLastGuiHB = Mts::Core::CDateTime::now();
}


unsigned int CEngine::getGuiHBMins() const {

    return m_iGuiHBMins;
}


void CEngine::disconnectClient(const std::string & strClientName) {

    //m_TCPServer.disconnectClient(strClientName);
}


std::vector<std::string> CEngine::getAllAlgoState() const {

    std::vector<std::string> objAllState;

    AlgorithmArray::const_iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter) {

        std::vector<std::string> objState = (*iter)->getState();

        for (size_t i = 0; i != objState.size(); ++i) {

            objAllState.push_back(objState[i]);
        }
    }

    return objAllState;
}


void CEngine::saveUserProfile(const std::string & strUID, 
                                                            const std::string & strDefaultCcys) {
}


bool CEngine::isProduction() const {

    return m_bProduction;
}


unsigned int CEngine::getEngineID() const {

    return m_iEngineID;
}


std::string CEngine::getEngineName() const {

    return m_strEngineName;
}


void CEngine::addSubscriber(const std::string & strProvider) {

    char szBuffer[512];
    sprintf(szBuffer, "Adding subscriber %s", strProvider.c_str());

    AppLog(szBuffer);

    FeedArray::iterator iterFd = m_Feeds.begin();

    for (; iterFd != m_Feeds.end(); ++iterFd)
        (*iterFd)->addSubscriber(strProvider);
}


void CEngine::removeSubscriber(const std::string & strProvider) {

    char szBuffer[512];
    sprintf(szBuffer, "Removing subscriber %s", strProvider.c_str());

    AppLog(szBuffer);

    FeedArray::iterator iterFd = m_Feeds.begin();

    for (; iterFd != m_Feeds.end(); ++iterFd)
        (*iterFd)->removeSubscriber(strProvider);
}

// this event handler is triggered by a simulated feed and is used to signal the end of a simulation
void CEngine::onFeedNoMoreData() {
/*
    for (auto objAlgo : m_Algorithms) {

        m_ptrSimulator->reportPerformance(objAlgo->getPositionManager());
    }
*/
    std::cout << "end of simulation" << std::endl;
}


void CEngine::onExchangeMessage(const std::string & strMsg) {

    /*
    AlgorithmArray::iterator iter = m_Algorithms.begin();

    for (; iter != m_Algorithms.end(); ++iter)
        (*iter)->onRecoveryDone();

    */
    AppLog("Recovery Done");
    m_bRecoveryDone = true;
    std::cout << "CEngine::onExchangeMessage() - Recovery Done" << std::endl;
}

void CEngine::checkAndLoadRecoveryFile() {
}

void CEngine::addBookWriter(const Mts::Core::CSymbol& csymbol,
                            const std::string& level
                           )
{
    unsigned int symbol_id = csymbol.getSymbolID();
    const std::string& venue = csymbol.getExchange();
    const std::string& symbol = csymbol.getContractExchSymbol();
    const md::BookConfig bcfg (venue, symbol, level);
    auto bq=std::make_shared<md::BookQ<TTBookLevel>> (bcfg, false);
    m_bq[symbol_id]=bq;
    m_book_writer[symbol_id]=std::make_shared<BookWriterType>(bcfg, bq->theWriter());
    m_bar_writer.add(bq);
}

void CEngine::writeBook(const Mts::OrderBook::CQuote& quote) {
    unsigned int symid = quote.getSymbolID();
    bool is_bid = (quote.getSide() == Mts::OrderBook::CQuote::BID);
    double px = quote.getPrice();
    long long sz = (long long) quote.getSize();

    auto iter = m_book_writer.find(symid);
    if (iter == m_book_writer.end()) {
        AppLogError("BBO update failed: symbolid %u not found!", symid);
        return;
    }
    auto& writer = iter->second;
    writer->updBBO
        (
            px,
            sz, 
            is_bid,
            utils::TimeUtil::cur_micro()
        );
}

void CEngine::writeBook(const Mts::OrderBook::CBidAsk& quote) {
    unsigned int symid = quote.getSymbolID();
    const auto& bid = quote.getBid();
    const auto& ask = quote.getAsk();

    auto iter = m_book_writer.find(symid);
    if (iter == m_book_writer.end()) {
        AppLogError("BBO update failed: symbolid %u not found!", symid);
        return;
    }

    auto& writer = iter->second;
    writer->updBBO
        (
            bid.getPrice(), bid.getSize(), 
            ask.getPrice(), ask.getSize(), 
            utils::TimeUtil::cur_micro()
        );
}

void CEngine::writeBook(const Mts::OrderBook::CTrade& trade) {
    unsigned int symid = trade.getSymbolID();
    double px = trade.getPrice();
    unsigned int qty = trade.getSize();

    auto iter = m_book_writer.find(symid);
    if (iter == m_book_writer.end()) {
        AppLogError("BBO update failed: symbolid %u not found!", symid);
        return;
    }
    auto& writer = iter->second;
    writer->updTrade(px, qty, utils::TimeUtil::cur_micro());
}

std::string CEngine::sendOrder(const char* data) {
    // expect a string: "B|S algo, symbol, qty, px, ClOrdId"
    //               or "C   ClOrdId, algo"
    //               or "R   ClOrdId, qty, px, algo, symbol, ReplaceClOrdId"
    // sign of qty:
    //    * for B|S, always positive
    //    * for R,   sign significant
    // data is a null-terminated string
    
    if (!m_bRecoveryDone) {
        AppLogInfo("Received order before recovery done : %s", data);
        //return std::string("Recovery not done yet, wait...");
    }
    std::string errstr = "";

    try {
        const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider("TT");
        unsigned int iProviderID = objProvider.getProviderID();
        Mts::Core::CDateTime dtNow = Mts::Core::CDateTime::now();
        std::string	strExecBrokerCode = "TT";
        boost::optional<std::reference_wrapper<Mts::Exchange::CExchange> > objExchange = Mts::Exchange::CExchangeBroker::getInstance().getExchange(iProviderID);

        bool bSent = false;
        switch (data[0]) {
        case 'B':
        case 'S':
        {
            bool isBuy = (data[0] == 'B'? true:false);
            auto line = utils::CSVUtil::read_line(std::string(data+1));
            Mts::Order::COrder::BuySell iDirection = isBuy ? Mts::Order::COrder::BUY : Mts::Order::COrder::SELL;

            auto iter = m_exch_contract_map.find(line[1]);
            if (iter == m_exch_contract_map.end()) {
                AppLogError("SendOrder Failed: symbol %s not found!", line[1].c_str());
                return "Symbol " + line[1] + " not found!";
            }
            unsigned int symbol_id = iter->second;
            const std::string& algo(line[0]);
            int qty = std::stoi(line[2]);
            double dPx = std::stod(line[3]);
            const std::string& clOrdId (line[4]);

            // check the tp side
            const auto& symbol(line[1]);
            if (__builtin_expect(!pm::risk::Monitor::get().checkNewOrder(algo, symbol, isBuy?qty:-qty),0)) {
                AppLogError("RiskMonitor %s:%s failed at FeedHandler checkNewOrder() for incoming new order %s, order not sent",
                        algo.c_str(), symbol.c_str(), data);
                bSent = false; break;
            }
            const auto cur_utc (utils::TimeUtil::cur_utc());

            if (__builtin_expect(!pm::risk::Monitor::get().config().isManualStrategy(algo), 1)) {
                if (__builtin_expect(!pm::risk::Monitor::get().checkLimitPrice(symbol, dPx, cur_utc),0)) {
                    AppLogError("RiskMonitor %s:%s failed at FeedHandler checkLimitPrice() for incoming new order %s, order not sent",
                            algo.c_str(), symbol.c_str(), data);
                    bSent = false; break;
                }
            }

            // check paper trading
            if (pm::risk::Monitor::get().config().isPaperTrading(algo)) {
                const auto er (pm::ExecutionReport::genSyntheticFills(line[1], algo, isBuy?qty:-qty,dPx, clOrdId));
                sendExecutionReport(er);
                AppLog("Generate Fill for paper trading on %s", data);
                bSent = true;
                break;
            }

            int algo_id_unused = 0;
            auto order = Mts::Order::COrderFactory::getInstance().createGTCOrder(
                dtNow, clOrdId, algo_id_unused, symbol_id, iProviderID, iDirection,
                (unsigned int)qty, dPx, algo, strExecBrokerCode
            );

            bSent = objExchange.value().get().submitLmtOrder(order, Mts::Exchange::CExchange::TimeInForce::GTC);
            break;
        }
        case 'C':
        {
            //"C   clOrdId, algo"
            auto line = utils::CSVUtil::read_line(std::string(data+1));
            bSent = objExchange.value().get().cancelOrder(line[0], line[1]);
            break;
        }

        case 'R':
        {
            //"R  ClOrdId, qty, px, algo, symbol, replaceClOrdId"
            //qty has to carry the sign
            auto line = utils::CSVUtil::read_line(std::string(data+1));
            bSent = objExchange.value().get().replaceOrder(line[0], std::stoll(line[1]), std::stod(line[2]), line[3], line[4], line[5]);
            if (!bSent) {
                errstr = "Failed in replace";
                AppLogError("Failed to send replace: %s", data);
            };
            break;
        }
        case 'Z' :
        {
            /* deprecated !
            
            // uses cancel + new as replacement, deprecated by replace
            //"Z  ClOrdId, qty, px, algo, symbol, replaceClOrdId"
            //qty has to carry the sign
            auto line = utils::CSVUtil::read_line(std::string(data+1));

            bSent = objExchange.value().get().cancelOrder(line[0], line[3]);
            if (!bSent) {
                errstr = "Failed in submitOrder";
                AppLogError("Failed to send order: %s", data);
                break;
            }
            int64_t qty = std::stoll(line[1]);
            if (qty == 0) {
                AppLog("nothing to be replaced, just cancel: %s", data);
                break;
            }
            auto ordStr = std::string(qty>0?"B ":"S ") + line[3] + ", " + line[4] + ", " + std::to_string(std::abs(qty)) + ", " + line[2] + ", " + line[5];
            errstr = sendOrder(ordStr.c_str());
            */
            AppLogError("CEngine Z deprecated! use R instead!");
            break;
        }
        default :
            AppLogError("unknown command: %s", data);
            errstr = std::string("unknown command: ") + std::string(data);
        }
        if (bSent) {
            AppLog("Order sent: %s", data);
        }
        else {
            AppLogError("Failed to send order: %s", data);
            errstr = "Failed in submitOrder";
        }
    } catch (const std::exception& e) {
        return e.what();
    }
    return errstr;
}

std::string CEngine::requestOpenOrder(const char* data, int size) {
    return "";
}

std::string CEngine::requestReplay(const char* data, int size, std::string& ready_file) {
    // data is in format of "from_utc, filepath"

    auto line = utils::CSVUtil::read_line(std::string(data));
    const std::string& fromLocalTime(line[0]);
    const std::string endLocalTime = utils::TimeUtil::frac_UTC_to_string(0,0);
    const std::string& dump_file (line[1]);

    try {
        sendRecoveryRequestToDropCopy(fromLocalTime, endLocalTime, dump_file);
    } catch (const std::exception& e) {
        return e.what();
    }
    ready_file = dump_file;
    return "";
}

