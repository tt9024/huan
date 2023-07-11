#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "CDropCopyFIX_TT.h"
#include "CExchangeFactory.h"
#include "CXMLParser.h"
#include "CExchangeFIX_TT.h"
#include "CAlgorithmFactory.h"


using namespace Mts::Exchange;


CExchangeFactory &CExchangeFactory::getInstance() {

    static CExchangeFactory theInstance;
    return theInstance;
}


boost::shared_ptr<CExchange> CExchangeFactory::createExchange(const std::string &strExchangeDefXML) {

    Mts::XML::CXMLParser objParser;
    boost::property_tree::ptree objPTree = objParser.parseXMLFile(strExchangeDefXML);

    std::string strClassName = objPTree.get<std::string>("exchange.class");

    if (strClassName == "Mts::Exchange::CFillSimulator")
        return createFillSimulator(objPTree);

    if (strClassName == "Mts::Exchange::CExchangeFIX_TT")
        return createExchangeFIX_TT(objPTree);

    throw Mts::Exception::CMtsException("unknown exchange");
}


boost::shared_ptr<CExchange> CExchangeFactory::createFillSimulator(const boost::property_tree::ptree &objPTree) {
    return nullptr;
    /*
	std::string		strProvider						= objPTree.get<std::string>("exchange.provider");
	std::string		strName								= objPTree.get<std::string>("exchange.name");

	// this the the provider that the fill simulator is assumed to be replacing
	const Mts::Core::CProvider & objProvider = Mts::Core::CProvider::getProvider(strProvider);

	boost::shared_ptr<Mts::Exchange::CExchange> ptrExchange(new Mts::Exchange::CFillSimulator(objProvider));

	return ptrExchange;
    */
}


boost::shared_ptr<CExchange> CExchangeFactory::createExchangeFIX_TT(const boost::property_tree::ptree &objPTree) {

    std::string strProvider = objPTree.get<std::string>("exchange.provider");
    std::string strName = objPTree.get<std::string>("exchange.name");
    std::string strSenderCompID = objPTree.get<std::string>("exchange.fix.SenderCompID");
    std::string strTargetCompID = objPTree.get<std::string>("exchange.fix.TargetCompID");
    std::string strUsername = objPTree.get<std::string>("exchange.fix.username");
    std::string strPassword = objPTree.get<std::string>("exchange.fix.password");
    std::string strAccount = objPTree.get<std::string>("exchange.fix.account");
    unsigned int iHeartbeat = objPTree.get<unsigned int>("exchange.fix.heartbeat");

    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider(strProvider);

    boost::shared_ptr<Mts::Exchange::CExchangeFIX_TT> ptrExchange(
            new Mts::Exchange::CExchangeFIX_TT(
                    objProvider,
                    strSenderCompID,
                    strTargetCompID,
                    strUsername,
                    strPassword,
                    strAccount,
                    iHeartbeat
            )
    );

    return ptrExchange;
}

boost::shared_ptr<CDropCopyFIX_TT> CExchangeFactory::createDropCopy(const std::string &strDropCopyXml) {
    Mts::XML::CXMLParser objParser;
    boost::property_tree::ptree objPTree = objParser.parseXMLFile(strDropCopyXml);
    std::string strClassName = objPTree.get<std::string>("exchange.class");
    if (strClassName != "Mts::Exchange::CDropCopyFIX_TT")
        throw Mts::Exception::CMtsException(std::string("unknown dropcopy class type: ") + strClassName);

    std::string strProvider = objPTree.get<std::string>("exchange.provider");
    std::string strName = objPTree.get<std::string>("exchange.name");
    std::string strSenderCompID = objPTree.get<std::string>("exchange.fix.SenderCompID");
    std::string strTargetCompID = objPTree.get<std::string>("exchange.fix.TargetCompID");
    std::string strUsername = objPTree.get<std::string>("exchange.fix.username");
    std::string strPassword = objPTree.get<std::string>("exchange.fix.password");
    std::string strAccount = objPTree.get<std::string>("exchange.fix.account");
    unsigned int iHeartbeat = objPTree.get<unsigned int>("exchange.fix.heartbeat");

    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider(strProvider);

    return boost::shared_ptr<CDropCopyFIX_TT>(new Mts::Exchange::CDropCopyFIX_TT(
            objProvider,
            strSenderCompID,
            strTargetCompID,
            strUsername,
            strPassword,
            strAccount,
            iHeartbeat));
}



