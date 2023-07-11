#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "CXMLParser.h"
#include "CProvider.h"
#include "CTokens.h"
#include "CConfig.h"
#include "CMtsException.h"
#include "CStringTokenizer.h"
#include "CFileUtil.h"


using namespace Mts::Core;


boost::unordered_map<std::string, boost::shared_ptr<CProvider> >    CProvider::m_Providers;
boost::unordered_map<unsigned int, boost::shared_ptr<CProvider> > CProvider::m_ProvidersByID;


bool CProvider::load(const std::string &strXML) {

    Mts::XML::CXMLParser objParser;
    boost::property_tree::ptree objPTree = objParser.parseXMLFile(strXML);

    BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, objPTree.get_child("providers")) {
                    const boost::property_tree::ptree &objChild = v.second;

                    unsigned int iProviderID = boost::lexical_cast<unsigned int>(objChild.get_child("id").data());
                    std::string strName = objChild.get_child("name").data();
                    std::string strShortName = objChild.get_child("shortcode").data();
                    double dRTTMSec = boost::lexical_cast<double>(objChild.get_child("rttms").data());
                    double dTicketFeeUSD = boost::lexical_cast<double>(objChild.get_child("ticketcostusd").data());

                    if (iProviderID >= CConfig::MAX_NUM_PROVIDERS)
                        throw Mts::Exception::CMtsException("maximum number of providers exceeded");

                    boost::shared_ptr<CProvider> ptrProvider(
                            new CProvider(iProviderID, strName, strShortName, dRTTMSec, dTicketFeeUSD));
                    m_Providers.insert(std::pair<std::string, boost::shared_ptr<CProvider> >(strName, ptrProvider));
                    m_ProvidersByID.insert(
                            std::pair<unsigned int, boost::shared_ptr<CProvider> >(iProviderID, ptrProvider));
                }

    return true;
}


const CProvider &CProvider::getProvider(const std::string &strName) {

    return *m_Providers.at(strName);
}


const CProvider &CProvider::getProvider(unsigned int iProviderID) {

    return *m_ProvidersByID.at(iProviderID);
}


std::vector<boost::shared_ptr<const CProvider> > CProvider::getProviders() {

    std::vector<boost::shared_ptr<const CProvider> > providers;
    for (auto &x: m_Providers) {
        providers.push_back(x.second);
    }
    //std::transform(m_Providers.begin(), m_Providers.end(), std::back_inserter(providers), [](auto x){ return x.second; });

    return providers;
}


bool CProvider::isProvider(unsigned int iProviderID) {

    return m_ProvidersByID.find(iProviderID) != m_ProvidersByID.end();
}


bool CProvider::isProvider(std::string strName) {

    return m_Providers.find(strName) != m_Providers.end();
}


CProvider::CProvider(unsigned int iProviderID,
        const std::string &strName,
        const std::string &strShortName,
        double dRTTMSec,
        double dTicketFeeUSD)
        : m_iProviderID(iProviderID),
          m_strName(strName),
          m_strShortName(strShortName),
          m_dRTTMSec(dRTTMSec),
          m_dTicketFeeUSD(dTicketFeeUSD) {

    m_dRTTDayFrac = dRTTMSec / (24.0 * 3600.0 * 1000.0);
}

