#include <boost/foreach.hpp>
#include "CModelFactory.h"
#include "CXMLParser.h"
#include "CMtsException.h"
#include "CModelPython.h"
#include "CModelCpp.h"


using namespace Mts::Model;


CModelFactory &CModelFactory::getInstance() {

    static CModelFactory objInstance;

    return objInstance;
}


boost::shared_ptr<CModel> CModelFactory::createModel(const boost::property_tree::ptree &objPTree,
        const std::string &strModelName,
        const std::string &strModelTag,
        const std::string &strTagPrefix) {

    if (strModelTag == "CModelPython")
        return createModelPython(objPTree, strModelName, strTagPrefix);

    if (strModelTag == "CModelCpp")
        return createModelCpp(objPTree, strModelName, strTagPrefix);

    throw Mts::Exception::CMtsException("CModelFactory: Unknown model");
}


boost::shared_ptr<CModel> CModelFactory::createModelPython(const boost::property_tree::ptree &objPTree,
        const std::string &strModelName,
        const std::string &strTagPrefix) {

    std::string strPythonModule = objPTree.get<std::string>(strTagPrefix + ".module");
    std::string strPythonFunction = objPTree.get<std::string>(strTagPrefix + ".function");
    unsigned int iBarSizeMin = objPTree.get<unsigned int>(strTagPrefix + ".barsizemin");
    unsigned int iFormationPeriod = objPTree.get<unsigned int>(strTagPrefix + ".formationperiod");

    boost::shared_ptr<Mts::Model::CModelPython> ptrModel(
            new Mts::Model::CModelPython(strModelName, strPythonModule, strPythonFunction, iBarSizeMin,
                                         iFormationPeriod));

    // instruments
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child(strTagPrefix + ".instruments")) {

                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = objChild.data();

                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);

                    ptrModel->addTradedInstrument(objSymbol);
                }

    bool bRet = ptrModel->initialize();

    if (bRet == false) {

        throw Mts::Exception::CMtsException("CModelFactory: Failed to initialize model python");
    }

    return ptrModel;
}


boost::shared_ptr<CModel> CModelFactory::createModelCpp(const boost::property_tree::ptree &objPTree,
        const std::string &strModelName,
        const std::string &strTagPrefix) {

    unsigned int iBarSizeMin = objPTree.get<unsigned int>(strTagPrefix + ".barsizemin");
    unsigned int iFormationPeriod = objPTree.get<unsigned int>(strTagPrefix + ".formationperiod");
    unsigned int iClipSize = objPTree.get<unsigned int>(strTagPrefix + ".clipsize");

    boost::shared_ptr<Mts::Model::CModelCpp> ptrModel(
            new Mts::Model::CModelCpp(strModelName, iBarSizeMin, iFormationPeriod, iClipSize));

    // instruments
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child(strTagPrefix + ".instruments")) {

                    const boost::property_tree::ptree &objChild = v.second;
                    std::string strSymbol = objChild.data();

                    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);

                    ptrModel->addTradedInstrument(objSymbol);
                }

    bool bRet = ptrModel->initialize();

    if (bRet == false) {

        throw Mts::Exception::CMtsException("CModelFactory: Failed to initialize model cpp");
    }

    return ptrModel;
}

