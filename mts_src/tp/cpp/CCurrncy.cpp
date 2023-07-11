#include <fstream>
#include <iterator>
#include <sstream>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "CCurrncy.h"
#include "CTokens.h"
#include "CXMLParser.h"


using namespace Mts::Core;


boost::unordered_map<std::string, boost::shared_ptr<CCurrncy> >        CCurrncy::m_CcysByName;
boost::unordered_map<unsigned int, boost::shared_ptr<CCurrncy> >    CCurrncy::m_CcysByID;


bool CCurrncy::load(const std::string &strXML) {

    Mts::XML::CXMLParser objParser;
    boost::property_tree::ptree objPTree = objParser.parseXMLFile(strXML);

    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("currencies")) {
                    const boost::property_tree::ptree &objChild = v.second;

                    unsigned int iCcyID = boost::lexical_cast<unsigned int>(objChild.get_child("id").data());
                    std::string strCcy = objChild.get_child("ccy").data();
                    double dCcy2USDExcRate = boost::lexical_cast<double>(objChild.get_child("usdrate").data());

                    boost::shared_ptr<CCurrncy> ptrCcy(new CCurrncy(iCcyID, strCcy, dCcy2USDExcRate));
                    m_CcysByName.insert(std::pair<std::string, boost::shared_ptr<CCurrncy> >(strCcy, ptrCcy));
                    m_CcysByID.insert(std::pair<unsigned int, boost::shared_ptr<CCurrncy> >(iCcyID, ptrCcy));
                }

    return true;
}


const CCurrncy &CCurrncy::getCcy(const std::string &strCcy) {

    return *m_CcysByName[strCcy];
}


const CCurrncy &CCurrncy::getCcy(unsigned int iCcyID) {

    return *m_CcysByID[iCcyID];
}


bool CCurrncy::isCcy(const std::string &strCcy) {

    return m_CcysByName.find(strCcy) != m_CcysByName.end();
}


bool CCurrncy::isCcy(unsigned int iCcyID) {

    return m_CcysByID.find(iCcyID) != m_CcysByID.end();
}


std::vector<boost::shared_ptr<const CCurrncy> > CCurrncy::getCcys() {

    std::vector<boost::shared_ptr<const CCurrncy> > ccys;
    CcyMapByName::const_iterator iter = m_CcysByName.begin();

    for (; iter != m_CcysByName.end(); ++iter) {

        ccys.push_back(iter->second);
    }

    return ccys;
}


CCurrncy::CCurrncy()
        : m_iCcyID(0),
          m_strCcy(""),
          m_dCcy2USDExcRate(0) {

}


CCurrncy::CCurrncy(unsigned int iCcyID,
        const std::string &strCcy,
        double dCcy2USDExcRate)
        : m_iCcyID(iCcyID),
          m_strCcy(strCcy),
          m_dCcy2USDExcRate(dCcy2USDExcRate) {

}


unsigned int CCurrncy::getCcyID() const {

    return m_iCcyID;
}


std::string CCurrncy::getCcy() const {

    return m_strCcy;
}


double CCurrncy::getCcy2USDExcRate() const {

    return m_dCcy2USDExcRate;
}



