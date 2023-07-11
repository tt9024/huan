#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>

#include "CApplication.h"
#include "CEngine.h"
#include "CFeedFactory.h"
#include "CExchangeFactory.h"
#include "COrderFactory.h"
#include "CConfig.h"
#include "CXMLParser.h"
#include "CApplicationLog.h"
#include "CFeedManager.h"

using namespace Mts::Application;

CApplication::CApplication(const std::string& version_string) : 
    m_version_string(version_string),
    m_should_run(false)
{ }

void CApplication::run(const std::string & strEnginePath)
{
	try {
		run_(strEnginePath);
	} catch (const std::exception & e) {
		std::cout << "Exception thrown" << std::endl;
		std::cout << e.what() << std::endl;
		try {
			Mts::Log::CApplicationLog & objApplicationLog = Mts::Log::CApplicationLog::getInstance();
			AppLog(std::string("Exception thrown") + e.what());
		}
		catch (const std::exception & ) {
			std::cout << "Cannot log exception!" << std::endl;
		}
	}
}

void CApplication::stop() {
    if (m_should_run) {
        m_should_run = false;
        try {
            m_ptrEngine->shutdownEngine();
        } catch (const std::exception& e) {
            fprintf(stderr, "problem shutting down engine: %s\n", e.what());
        }
    }
}

CApplication::~CApplication() {
    stop();
}

void CApplication::run_(const std::string & strEnginePath)
{

    Mts::Core::CDateTime dtNow = Mts::Core::CDateTime::now();

    std::cout << "MTS version " << m_version_string << std::endl;

    std::cout << "Instantiating configuration" << std::endl;
    Mts::Core::CConfig& objConfig = MTSConfigInstance;

    Mts::XML::CXMLParser objParser;
    objConfig.setEnginePath(strEnginePath);
    const std::string strEngineXML = objConfig.getEngineConfigFile();
    boost::property_tree::ptree objPTree = objParser.parseXMLFile(strEngineXML);

    // engine
    unsigned int        iEngineID               = objPTree.get<unsigned int>("engine.general.engineid");
    std::string         strName                 = objPTree.get<std::string>("engine.general.name");
    std::string         strFIXConfig            = MTSConfigFile(objPTree.get<std::string>("engine.general.fixconfig"));
    std::string         strRiskXML              = MTSConfigFile(objPTree.get<std::string>("engine.general.risklimits"));
    unsigned int        iTCPBufferSize          = objPTree.get<unsigned int>("engine.general.tcpserverbuffer");
    unsigned int        iTCPPort                = objPTree.get<unsigned int>("engine.general.tcpserverport");
    unsigned int        iEngCtrlPort            = objPTree.get<unsigned int>("engine.general.engctrlport");
    unsigned int        iGuiHBMins              = objPTree.get<unsigned int>("engine.general.guihbmins");
    unsigned int        iGuiUpdateMSec          = objPTree.get<unsigned int>("engine.general.guiupdatethrottlemsec");
    unsigned int        iUseDateTimeEvents      = objPTree.get<unsigned int>("engine.general.usedatetimeevents");
    unsigned int        iEventLoopSleepMs       = objPTree.get<unsigned int>("engine.general.eventloopsleepms");
    unsigned int        iProduction             = objPTree.get<unsigned int>("engine.general.production");
    unsigned int        iMktDataMgrBufferSize   = objPTree.get<unsigned int>("engine.general.mktdatamgrbuffer");
    unsigned int        iMktDataMgrUpdateMSec   = objPTree.get<unsigned int>("engine.general.mktdatamgrupdatemsec");
    unsigned int        iMktDataMgrUseSQL       = objPTree.get<unsigned int>("engine.general.mktdatamgrusesql");

    objConfig.setEngineID(iEngineID);

    // simulation
    std::string         strSimulatorXML                = MTSConfigFile(objPTree.get<std::string>("engine.simulation.simulatorxml"));
    unsigned int        iSimulationMode                = objPTree.get<unsigned int>("engine.simulation.simulationmode");
    unsigned int        iSimulationModeFullReporting   = objPTree.get<unsigned int>("engine.simulation.simulationmodefullreporting");
    unsigned int        iSimulationModeInitAlgos       = objPTree.get<unsigned int>("engine.simulation.simulationmodeinitalgos");

    // reporting
    unsigned int        iLoadPositions                 = objPTree.get<unsigned int>("engine.reporting.loadpositionsonstart");
    unsigned int        iLoadPositionsFromSQL          = objPTree.get<unsigned int>("engine.reporting.loadpositionsfromsql");
    unsigned int        iLoadOrderHistory              = objPTree.get<unsigned int>("engine.reporting.loadorderhistory");
    std::string         strRecoveryDir                 = objPTree.get<std::string>("engine.reporting.recoverydir");

    objConfig.setRecoveryFileDirectory(strRecoveryDir);
    

    // email
    std::string         strSMTPServer           = objPTree.get<std::string>("engine.email.smtpserver");
    std::string         strFromEmailAddr        = objPTree.get<std::string>("engine.email.fromemailaddr");
    std::string         strToEmailAddr          = objPTree.get<std::string>("engine.email.toemailaddr");
    unsigned int        iEnableEmailer          = objPTree.get<unsigned int>("engine.email.enableemailer");

    objConfig.setSMTPServer(strSMTPServer);
    objConfig.setFromEmailAddr(strFromEmailAddr);
    objConfig.setToEmailAddr(strToEmailAddr);
    objConfig.setEnableEmailer(iEnableEmailer == 1);

    std::cout << "Creating logs" << std::endl;

    // logging
    objConfig.setLogFileDirectory(objPTree.get<std::string>("engine.logging.directory"));
    std::string         strLogDir           = objConfig.getLogFileDirectory();
    std::string         strMktDataQuoteLog  = objPTree.get<std::string>("engine.logging.marketdataquotelog");
    std::string         strMktDataTradeLog  = objPTree.get<std::string>("engine.logging.marketdatatradelog");
    unsigned int        iBufferSize         = objPTree.get<unsigned int>("engine.logging.buffersize");
    unsigned int        iIntervalSecs       = objPTree.get<unsigned int>("engine.logging.intervalsecs");
    unsigned int        iNoLoggingFlag      = objPTree.get<unsigned int>("engine.logging.nologging");

    objConfig.setLogBufferSize(iBufferSize);
    objConfig.setLogIntervalSec(iIntervalSecs);
	std::cout << "Instantiating application log" << std::endl;
	Mts::Log::CApplicationLog& objApplicationLog = Mts::Log::CApplicationLog::getInstance();

    std::cout << "Loading static data" << std::endl;

    // load static data
    std::string         strCcyDictXML       = MTSConfigFile(objPTree.get<std::string>("engine.staticdata.ccydef"));
    std::string         strSymbolDictXML    = MTSConfigFile(objPTree.get<std::string>("engine.staticdata.symboldef"));
    std::string         strProviderDictXML  = MTSConfigFile(objPTree.get<std::string>("engine.staticdata.providerdef"));

    Mts::Core::CCurrncy::load(strCcyDictXML);
    Mts::Core::CSymbol::load(strSymbolDictXML);
    Mts::Core::CProvider::load(strProviderDictXML);

    // identifiers
    unsigned int        iGenerateOrderID   = objPTree.get<unsigned int>("engine.identifiers.generateorderid");
    unsigned long       iNextOrderID       = objPTree.get<unsigned long>("engine.identifiers.nextorderid");

    std::cout << "Instantiating order factory" << std::endl;

    COrderFactory::setLoadOrderHistory(iLoadOrderHistory == 1);

    // force creation of singletons on main thread
    Mts::Order::COrderFactory& objOrderFactory  = Mts::Order::COrderFactory::getInstance();
    objOrderFactory.setGenerateUniqueOrderID(iGenerateOrderID == 1);
    //use a random starting id unique within a month
    //objOrderFactory.setNextOrderID(iNextOrderID);

    std::cout << "Engine ID: " << iEngineID << "  Order ID: " << objOrderFactory.queryNextOrderID() << std::endl;
	char buf[128];
	snprintf(buf, sizeof(buf), "Engine ID: %llu Order ID: %llu", 
		                       (unsigned long long )iEngineID, 
		                       (unsigned long long) objOrderFactory.queryNextOrderID());
	AppLog(buf);

    std::cout << "Instantiating exchange broker" << std::endl;
    AppLog("Instantiating exchange broker");
    Mts::Exchange::CExchangeBroker& objExchBroker = Mts::Exchange::CExchangeBroker::getInstance();

    /*
    std::cout << "Instantiating algorithm factory" << std::endl;
    AppLog("Instantiating algorithm factory");
    Mts::Algorithm::CAlgorithmFactory & objAlgoFactory = Mts::Algorithm::CAlgorithmFactory::getInstance();
    */

    std::cout << "Instantiating feed factory" << std::endl;
    AppLog("Instantiating feed factory");
    Mts::Feed::CFeedFactory& objFeedFactory = Mts::Feed::CFeedFactory::getInstance();

    std::cout << "Instantiating exchange factory" << std::endl;
    AppLog("Instantiating exchange factory");
    Mts::Exchange::CExchangeFactory& objExchangeFactory = Mts::Exchange::CExchangeFactory::getInstance();

    /*
    std::cout << "Instantiating user manager" << std::endl;
    AppLog("Instantiating user manager");
    Mts::User::CUserManager::setModeSQL(false);
    Mts::User::CUserManager &   objUserManager = Mts::User::CUserManager::getInstance();
    */

    std::cout << "Initializing engine" << std::endl;
    AppLog("Initializing engine");


    /*
    boost::shared_ptr<Mts::Engine::CEngine>   ptrEngine(new Mts::Engine::CEngine(\
        iEngineID, 
        strName,strFIXConfig,
        PathJoin(strLogDir,strMktDataQuoteLog), 
        PathJoin(strLogDir,strMktDataTradeLog),
        iEngCtrlPort,
        iNoLoggingFlag == 1,
        iSimulationMode == 1,
        iSimulationModeFullReporting == 1, 
        iTCPBufferSize,iTCPPort, 
        iGuiHBMins, 
        iGuiUpdateMSec, 
        iLoadPositions == 1, 
        iLoadPositionsFromSQL == 1, 
        iLoadOrderHistory == 1,
        iUseDateTimeEvents == 1, 
        iEventLoopSleepMs, 
        iProduction == 1, 
        iMktDataMgrBufferSize, 
        iMktDataMgrUpdateMSec));
    */

    m_ptrEngine.reset(new Mts::Engine::CEngine(
        iEngineID, 
        strName,strFIXConfig,
        PathJoin(strLogDir,strMktDataQuoteLog), 
        PathJoin(strLogDir,strMktDataTradeLog),
        iEngCtrlPort,
        iNoLoggingFlag == 1,
        iSimulationMode == 1,
        iSimulationModeFullReporting == 1, 
        iTCPBufferSize,iTCPPort, 
        iGuiHBMins, 
        iGuiUpdateMSec, 
        iLoadPositions == 1, 
        iLoadPositionsFromSQL == 1, 
        iLoadOrderHistory == 1,
        iUseDateTimeEvents == 1, 
        iEventLoopSleepMs, 
        iProduction == 1, 
        iMktDataMgrBufferSize, 
        iMktDataMgrUpdateMSec));

    m_should_run = true;


    /*
    std::cout << "Loading algorithms" << std::endl;
    AppLog("Loading algorithms");
    BOOST_FOREACH(const boost::property_tree::ptree::value_type & v, objPTree.get_child("engine.algorithms")) {
        const boost::property_tree::ptree & objChild = v.second;
        std::string algoname = objChild.data();
        std::cout << "Loading " << algoname << std::endl;
        AppLog("Loading " + algoname);
        m_ptrEngine->addAlgorithm(algoname);
    }
    */

    std::cout << "Loading feeds" << std::endl;
    AppLog("Loading feeds");

    for (const auto& v : objPTree.get_child("engine.feeds")) {
        const boost::property_tree::ptree & objChild = v.second;
        std::string strFeedXML = MTSConfigFile(objChild.data());

        std::cout << "Loading " << strFeedXML << std::endl;
        AppLog("Loading " + strFeedXML);

        boost::shared_ptr<Mts::Feed::CFeed> ptrFeed = Mts::Feed::CFeedFactory::getInstance().createFeed(strFeedXML);
        m_ptrEngine->addFeed(ptrFeed);

        Mts::Feed::CFeedManager::getInstance().addFeed(strFeedXML, ptrFeed);
    }

    std::cout << "Loading exchanges" << std::endl;
    AppLog("Loading exchanges");

    for (const auto& v : objPTree.get_child("engine.exchanges")) {
        const boost::property_tree::ptree& objChild = v.second;
        std::string strExchangeXML = MTSConfigFile(objChild.data());

        std::cout << "Loading " << strExchangeXML << std::endl;
        AppLog("Loading " + strExchangeXML);

        boost::shared_ptr<Mts::Exchange::CExchange> ptrExchange = Mts::Exchange::CExchangeFactory::getInstance().createExchange(strExchangeXML);
        m_ptrEngine->addExchange(ptrExchange);
    }

    //boost::shared_ptr<Mts::Networking::CTCPEngineController> ptrTCPEngineController(new Mts::Networking::CTCPEngineController(iEngCtrlPort, m_ptrEngine));
    //m_ptrEngine->setTCPEngineController(ptrTCPEngineController);

    m_ptrEngine->run();
    
};

