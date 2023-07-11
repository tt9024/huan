#include <boost/foreach.hpp>
#include "CFeedFactory.h"
#include "CXMLParser.h"
#include "CFeedEnginePersistedTOB.h"
#include "CFeedTickDataTOB.h"
#include "CFeedFIX_TT.h"
#include "CStringTokenizer.h"
#include "CMtsException.h"
#include "symbol_map.h"

using namespace Mts::Feed;


CFeedFactory &CFeedFactory::getInstance() {

    static CFeedFactory theInstance;
    return theInstance;
}


boost::shared_ptr<CFeed> CFeedFactory::createFeed(const std::string &strFeedDefXML) {

    Mts::XML::CXMLParser objParser;
    boost::property_tree::ptree objPTree = objParser.parseXMLFile(strFeedDefXML);

    std::string strClassName = objPTree.get<std::string>("feed.class");

    if (strClassName == "Mts::Feed::CFeedEnginePersistedTOB")
        return createFeedEnginePersistedTOB(objPTree);

    if (strClassName == "Mts::Feed::CFeedTickDataTOB")
        return createFeedTickDataTOB(objPTree);

    if (strClassName == "Mts::Feed::CFeedFIX_TT")
        return createFeedFIX_TT(objPTree);
    /*
    if (strClassName == "Mts::Feed::CFeedPollDB")
        return createFeedPollDB(objPTree);
    */

    throw Mts::Exception::CMtsException("unknown feed");
}


boost::shared_ptr<CFeed> CFeedFactory::createFeedEnginePersistedTOB(const boost::property_tree::ptree &objPTree) {

    std::string strProvider = objPTree.get<std::string>("feed.provider");
    std::string strCounterparty = objPTree.get<std::string>("feed.counterparty");
    std::string strName = objPTree.get<std::string>("feed.name");
    unsigned int iSize = boost::lexical_cast<unsigned int>(objPTree.get<std::string>("feed.size"));
    unsigned int iStreamFromAllProviders = boost::lexical_cast<unsigned int>(
            objPTree.get<std::string>("feed.streamfromallproviders"));
    double dSpreadMultiplier = boost::lexical_cast<double>(objPTree.get<std::string>("feed.spreadmultiplier"));
    unsigned int iQuoteThrottleSecs = boost::lexical_cast<unsigned int>(
            objPTree.get<std::string>("feed.quotethrottlemsecs"));
    std::string strTimeWnd = objPTree.get<std::string>("feed.timewindow");

    // only quotes within a defined time window will be streamed
    Mts::Core::CStringTokenizer objStringSplitter;

    std::vector<std::string> objTokensWnd = objStringSplitter.splitString(strTimeWnd, "-");

    std::string strStartTime = objTokensWnd[0];
    std::string strEndTime = objTokensWnd[1];

    unsigned int iStartHR = boost::lexical_cast<unsigned int>(strStartTime.substr(0, 2));
    unsigned int iStartMI = boost::lexical_cast<unsigned int>(strStartTime.substr(2, 2));
    unsigned int iEndHR = boost::lexical_cast<unsigned int>(strEndTime.substr(0, 2));
    unsigned int iEndMI = boost::lexical_cast<unsigned int>(strEndTime.substr(2, 2));
    unsigned int iStartInMin = iStartHR * 60 + iStartMI;
    unsigned int iEndInMin = iEndHR * 60 + iEndMI;

    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider(strProvider);
    boost::shared_ptr<Mts::Feed::CFeed> ptrFeed(
            new Mts::Feed::CFeedEnginePersistedTOB(objProvider, strCounterparty, iSize, iStreamFromAllProviders == 1,
                                                   dSpreadMultiplier, iQuoteThrottleSecs, iStartInMin, iEndInMin));

    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("feed.subscription")) {
                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = objChild.data();
                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);

                    (boost::static_pointer_cast<Mts::Feed::CFeedEnginePersistedTOB>(ptrFeed))->addSymbol(
                            objSymbol.getSymbolID());
                }

    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("feed.inputfiles")) {
                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strInputFile = objChild.data();

                    (boost::static_pointer_cast<Mts::Feed::CFeedEnginePersistedTOB>(ptrFeed))->addInputFile(
                            strInputFile);
                }

    return ptrFeed;
}


boost::shared_ptr<CFeed> CFeedFactory::createFeedTickDataTOB(const boost::property_tree::ptree &objPTree) {

    std::string strProvider = objPTree.get<std::string>("feed.provider");
    std::string strName = objPTree.get<std::string>("feed.name");
    double dSpreadMultiplier = boost::lexical_cast<double>(objPTree.get<std::string>("feed.spreadmultiplier"));
    unsigned int iQuoteThrottleSecs = boost::lexical_cast<unsigned int>(
            objPTree.get<std::string>("feed.quotethrottlemsecs"));
    std::string strTimeWnd = objPTree.get<std::string>("feed.timewindow");
    unsigned int iTradeFileFormat = boost::lexical_cast<unsigned int>(
            objPTree.get<std::string>("feed.tradefileformat"));

    // only quotes within a defined time window will be streamed
    Mts::Core::CStringTokenizer objStringSplitter;

    std::vector<std::string> objTokensWnd = objStringSplitter.splitString(strTimeWnd, "-");

    std::string strStartTime = objTokensWnd[0];
    std::string strEndTime = objTokensWnd[1];

    unsigned int iStartHR = boost::lexical_cast<unsigned int>(strStartTime.substr(0, 2));
    unsigned int iStartMI = boost::lexical_cast<unsigned int>(strStartTime.substr(2, 2));
    unsigned int iEndHR = boost::lexical_cast<unsigned int>(strEndTime.substr(0, 2));
    unsigned int iEndMI = boost::lexical_cast<unsigned int>(strEndTime.substr(2, 2));
    unsigned int iStartInMin = iStartHR * 60 + iStartMI;
    unsigned int iEndInMin = iEndHR * 60 + iEndMI;

    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider(strProvider);
    boost::shared_ptr<Mts::Feed::CFeed> ptrFeed(
            new Mts::Feed::CFeedTickDataTOB(objProvider, dSpreadMultiplier, iQuoteThrottleSecs, iStartInMin, iEndInMin,
                                            iTradeFileFormat == 1));

    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("feed.subscription")) {
                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = objChild.data();
                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);

                    (boost::static_pointer_cast<Mts::Feed::CFeedTickDataTOB>(ptrFeed))->addSymbol(
                            objSymbol.getSymbolID());
                }

    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("feed.inputfiles")) {
                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strInputFile = objChild.data();

                    (boost::static_pointer_cast<Mts::Feed::CFeedTickDataTOB>(ptrFeed))->addInputFile(strInputFile);
                }

    return ptrFeed;
}


boost::shared_ptr<CFeed> CFeedFactory::createFeedFIX_TT(const boost::property_tree::ptree &objPTree) {
    std::string strProvider = objPTree.get<std::string>("feed.provider");
    std::string strName = objPTree.get<std::string>("feed.name");
    std::string strSenderCompID = objPTree.get<std::string>("feed.fix.SenderCompID");
    std::string strTargetCompID = objPTree.get<std::string>("feed.fix.TargetCompID");
    std::string strUsername = objPTree.get<std::string>("feed.fix.username");
    std::string strPassword = objPTree.get<std::string>("feed.fix.password");
    unsigned int iHeartbeat = objPTree.get<unsigned int>("feed.fix.heartbeat");

    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider(strProvider);
    boost::shared_ptr<Mts::Feed::CFeed> ptrFeed(new Mts::Feed::CFeedFIX_TT(objProvider,
                                                                           strSenderCompID,
                                                                           strTargetCompID,
                                                                           strUsername,
                                                                           strPassword,
                                                                           iHeartbeat));

    const auto &msv_pair(utils::SymbolMapReader::get().getSubscriptions("TTFix"));
    // only subscribe to the primary
    const auto &msv(msv_pair.first);
    for (const auto &ms: msv) {
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(ms);
        (boost::static_pointer_cast<Mts::Feed::CFeedFIX_TT>(ptrFeed))->addSubscription(objSymbol);
    }
    return ptrFeed;
}

/*
boost::shared_ptr<CFeed> CFeedFactory::createFeedPollDB(const boost::property_tree::ptree & objPTree) {

    std::string     strName = objPTree.get<std::string>("feed.name");
    
    boost::shared_ptr<Mts::Feed::CFeed> ptrFeed(new Mts::Feed::CFeedPollDB());

    BOOST_FOREACH(const boost::property_tree::ptree::value_type & v, objPTree.get_child("feed.queries")) {
        const boost::property_tree::ptree & objChild = v.second;

        std::string strKey                  = objChild.get_child("key").data();
        unsigned int iPollFreqSecs  = boost::lexical_cast<unsigned int>(objChild.get_child("pollfreqsecs").data());
        std::string strSQL                  = objChild.get_child("sql").data();
        std::string strDSN                  = objChild.get_child("dsn").data();

        (boost::static_pointer_cast<Mts::Feed::CFeedPollDB>(ptrFeed))->addQuery(strKey, iPollFreqSecs, strSQL, strDSN);
    }

    return ptrFeed;
}
*/

