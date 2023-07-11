#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <set>
#include "CAlgorithmFactory.h"
#include "CXMLParser.h"
#include "CAlgorithmTemplate.h"
#include "CAlgorithmFIXTest.h"
#include "CSymbol.h"
#include "CMtsException.h"
#include "CStringTokenizer.h"
#include "CModelFactory.h"
#include "CMath.h"


using namespace Mts::Algorithm;


CAlgorithmFactory &CAlgorithmFactory::getInstance() {

    static CAlgorithmFactory theInstance;
    return theInstance;
}


boost::shared_ptr<CAlgorithm> CAlgorithmFactory::createAlgorithm(const std::string &strAlgoDefXML) {

    Mts::XML::CXMLParser objParser;

    boost::property_tree::ptree objPTree = objParser.parseXMLFile(strAlgoDefXML);

    std::string strClassName = objPTree.get<std::string>("algorithm.class");

    if (strClassName == "Mts::Algorithm::CAlgorithmTemplate")
        return createAlgorithmTemplate(objPTree);

    if (strClassName == "Mts::Algorithm::CAlgorithmFIXTest")
        return createAlgorithmFIXTest(objPTree);

    throw Mts::Exception::CMtsException("unknown algorithm");
}


boost::shared_ptr<CAlgorithm> CAlgorithmFactory::createAlgorithmTemplate(const boost::property_tree::ptree &objPTree) {

    unsigned int iAlgoID = objPTree.get<unsigned int>("algorithm.algoid");
    std::string strName = objPTree.get<std::string>("algorithm.name");
    unsigned int iBufferSize = objPTree.get<unsigned int>("algorithm.buffersize");
    unsigned int iThrottleMS = objPTree.get<unsigned int>("algorithm.throttlemsec");
    unsigned int iExecAlgoID = objPTree.get<unsigned int>("algorithm.params.execalgoid");

    boost::shared_ptr<Mts::Algorithm::CAlgorithm> ptrAlgo(
            new Mts::Algorithm::CAlgorithmTemplate(iAlgoID, strName, iBufferSize, iThrottleMS, iExecAlgoID));

    // trade schedule
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
                  objPTree.get_child("algorithm.params.tradingschedule")) {
                    const boost::property_tree::ptree &objChild = v.second;

                    const char *pszHHMM = objChild.data().c_str();
                    unsigned int iHH = static_cast<unsigned int>(Mts::Math::CMath::atoi_2(pszHHMM));
                    unsigned int iMM = static_cast<unsigned int>(Mts::Math::CMath::atoi_2(pszHHMM + 2));
                    unsigned int iTimeInMin = iHH * 60 + iMM;

                    (boost::static_pointer_cast<Mts::Algorithm::CAlgorithmTemplate>(ptrAlgo))->setTradeTime(iTimeInMin);
                }

    // traded instruments
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("algorithm.symbol")) {
                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = objChild.data();

                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);
                    ptrAlgo->addTradedInstrument(objSymbol);
                }

    // sub-models
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("algorithm.portfolio")) {
                    const boost::property_tree::ptree &objChild = v.second;

                    std::string strModelName = objChild.get_child("name").data();
                    std::string strModelTag = objChild.get_child("modeltag").data();
                    std::string strConfigTag = objChild.get_child("configtag").data();

                    boost::shared_ptr<Mts::Model::CModel> ptrModel = Mts::Model::CModelFactory::getInstance().createModel(
                            objPTree, strModelName, strModelTag, "algorithm." + strConfigTag);
                    (boost::static_pointer_cast<Mts::Algorithm::CAlgorithmTemplate>(ptrAlgo))->addModel(ptrModel);
                }

    loadRiskLimits(objPTree, ptrAlgo);

    return ptrAlgo;
}


void CAlgorithmFactory::loadRiskLimits(const boost::property_tree::ptree &objPTree,
        boost::shared_ptr<Mts::Algorithm::CAlgorithm> ptrAlgo) {

    // position limits for each symbol
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
                  objPTree.get_child("algorithm.limits.maxpositionsize")) {

                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = v.first;
                    unsigned int iMaxPositionSize = boost::lexical_cast<unsigned int>(objChild.data());

                    if (Mts::Core::CSymbol::isSymbol(strSymbol) == false)
                        continue;

                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);
                    ptrAlgo->setMaxPosition(objSymbol, iMaxPositionSize);
                }


    // order size limits for each symbol
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
                  objPTree.get_child("algorithm.limits.maxordersize")) {

                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = v.first;
                    unsigned int iMaxOrderSize = boost::lexical_cast<unsigned int>(objChild.data());

                    if (Mts::Core::CSymbol::isSymbol(strSymbol) == false)
                        continue;

                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);
                    ptrAlgo->setMaxOrderSize(objSymbol, iMaxOrderSize);
                }


    // min time interval between trades in a given pair
    int iMinTradeIntervalSecs = boost::lexical_cast<int>(
            objPTree.get_child("algorithm.limits.tradeintervalinsec").data());
    ptrAlgo->setMinTradeIntervalSecs(iMinTradeIntervalSecs);
}


boost::shared_ptr<CAlgorithm> CAlgorithmFactory::createAlgorithmFIXTest(const boost::property_tree::ptree &objPTree) {

    unsigned int iAlgoID = objPTree.get<unsigned int>("algorithm.algoid");
    std::string strName = objPTree.get<std::string>("algorithm.name");
    unsigned int iBufferSize = objPTree.get<unsigned int>("algorithm.buffersize");
    unsigned int iThrottleMS = objPTree.get<unsigned int>("algorithm.throttlemsec");
    unsigned int iTestID = objPTree.get<unsigned int>("algorithm.params.testid");

    boost::shared_ptr<Mts::Algorithm::CAlgorithm> ptrAlgo(
            new Mts::Algorithm::CAlgorithmFIXTest(iAlgoID, strName, iBufferSize, iThrottleMS, iTestID));

    // traded instruments
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("algorithm.symbol")) {
                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = objChild.data();

                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);
                    ptrAlgo->addTradedInstrument(objSymbol);
                }

    loadRiskLimits(objPTree, ptrAlgo);

    return ptrAlgo;
}


