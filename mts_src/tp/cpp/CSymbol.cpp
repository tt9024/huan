#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include "CSymbol.h"
#include "CSymbolGreater.h"
//#include "CTokens.h"
//#include "CStringTokenizer.h"
#include "symbol_map.h"
#include "plcc/PLCC.hpp"

using namespace Mts::Core;


// key by the "symbol" in symbol definition
std::unordered_map<std::string, boost::shared_ptr<CSymbol> >  CSymbol::m_SymbolsByName;
std::unordered_map<unsigned int, boost::shared_ptr<CSymbol> > CSymbol::m_SymbolsByID;
std::unordered_map<std::string, boost::shared_ptr<CSymbol> >  CSymbol::m_SymbolsByExchContract;

bool CSymbol::load(const std::string &strXML) {
    std::string cfg = plcc_getString("VenueMap");
    const utils::ConfigureReader cr(cfg.c_str());

    unsigned int iSymbolID = 0;
    const auto &all_tradable(utils::SymbolMapReader::get().listAllTradable());
    for (const auto *tinfo: all_tradable) {
        ++iSymbolID;
        std::string strSymbol = tinfo->_mts_symbol;
        std::string strBaseCcy = tinfo->_currency;
        double dPointValue = tinfo->_point_value;
        double dTickSize = tinfo->_tick_size;
        //std::string     strExchSymbol       = tinfo->_symbol;
        std::string strExchSymbol = tinfo->_exch_symbol;
        std::string strExchange = tinfo->_venue;
        std::string strTTExchange = cr.get<std::string>(strExchange.c_str(), nullptr, strExchange);
        std::string strSecurityType = tinfo->_type;
        std::string strTTSymbol = tinfo->_exch_symbol;
        std::string strContractExchSym = tinfo->_tradable;
        double dTTScaleMultiplier = tinfo->_px_multiplier;
        std::string strContractExchSymbol = tinfo->_tradable;
        std::string strContractMonth = tinfo->_contract_month;
        std::string strTTSecID = tinfo->_tt_security_id;

        // not used for now
        std::string strTDSymbol = "";
        double dTDScaleMultiplier = 0.0;
        std::string strRefCcy = strBaseCcy;
        unsigned int iAssetClass = 1;
        unsigned int iTransmitToGUI = 0;

        unsigned int iLotsPerMin = 10;
        unsigned int iTimeSliceSecs = 30;

        unsigned int iDiscVal = 10;
        unsigned int iVariance = 10;
        unsigned int iLimitTicksAway = 1;

        // get base ccy
        const CCurrncy &objBaseCcy = CCurrncy::getCcy(strBaseCcy);

        if (objBaseCcy.getCcyID() == 0)
            throw Mts::Exception::CMtsException("missing currency definition");

        // get ref ccy
        const CCurrncy &objRefCcy = CCurrncy::getCcy(strRefCcy);

        if (objRefCcy.getCcyID() == 0)
            throw Mts::Exception::CMtsException("missing currency definition");

        boost::shared_ptr<CSymbol> ptrSymbol(new CSymbol(
                iSymbolID, strSymbol, objBaseCcy, objRefCcy,
                dPointValue, dTickSize, static_cast<Mts::Core::CSymbol::AssetClass>(iAssetClass),
                iTransmitToGUI == 1, strExchSymbol, strExchange, strTTExchange, strSecurityType, strTTSymbol,
                dTTScaleMultiplier, strTDSymbol, dTDScaleMultiplier, iLotsPerMin,
                iTimeSliceSecs, iDiscVal, iVariance, iLimitTicksAway,
                strContractExchSymbol, strContractMonth, strTTSecID));
        m_SymbolsByName.insert(std::pair<std::string, boost::shared_ptr<CSymbol> >(strSymbol, ptrSymbol));
        m_SymbolsByID.insert(std::pair<unsigned int, boost::shared_ptr<CSymbol> >(iSymbolID, ptrSymbol));

    }

    return true;
}

const CSymbol &CSymbol::getSymbol(const std::string &strSymbol) {
    if (!isSymbol(strSymbol)) {
        return getSymbolFromExchSymbol(strSymbol);
    }
    return *m_SymbolsByName.at(strSymbol);
}


const CSymbol &CSymbol::getSymbol(unsigned int iSymbolID) {

    return *m_SymbolsByID.at(iSymbolID);
}


bool CSymbol::isSymbol(const std::string &strSymbol) {

    return m_SymbolsByName.find(strSymbol) != m_SymbolsByName.end();
}


bool CSymbol::isSymbol(unsigned int iSymbolID) {

    return m_SymbolsByID.find(iSymbolID) != m_SymbolsByID.end();
}


bool CSymbol::transmitToGUI() const {

    return m_bTransmitToGUI;
}


std::vector<boost::shared_ptr<const CSymbol> > CSymbol::getGUISymbols() {

    std::vector<boost::shared_ptr<const CSymbol> > symbols;
    SymbolMapByName::const_iterator iter = m_SymbolsByName.begin();

    for (; iter != m_SymbolsByName.end(); ++iter) {
        if (iter->second->transmitToGUI() == true)
            symbols.push_back(iter->second);
    }

    sort(symbols.begin(), symbols.end(), CSymbolGreater());

    return symbols;
}


std::vector<boost::shared_ptr<const CSymbol> > CSymbol::getSymbols(AssetClass iAssetClass) {

    std::vector<boost::shared_ptr<const CSymbol> > symbols;
    SymbolMapByName::const_iterator iter = m_SymbolsByName.begin();

    for (; iter != m_SymbolsByName.end(); ++iter) {
        if (iter->second->getAssetClass() == iAssetClass)
            symbols.push_back(iter->second);
    }

    sort(symbols.begin(), symbols.end(), CSymbolGreater());

    return symbols;
}


std::vector<boost::shared_ptr<const CSymbol> > CSymbol::getSymbols() {

    std::vector<boost::shared_ptr<const CSymbol> > symbols;

    for (auto &x: m_SymbolsByName) {
        symbols.push_back(x.second);
    }
    sort(symbols.begin(), symbols.end(), CSymbolGreater());

    return symbols;
}


CSymbol::CSymbol() {

}

CSymbol::CSymbol(unsigned int iSymbolID,
        const std::string &strSymbol,
        const CCurrncy &objBaseCcy,
        const CCurrncy &objRefCcy,
        double dPointValue,
        double dTickSize,
        AssetClass iAssetClass,
        bool bTransmitToGUI,
        const std::string &strExchSymbol,
        const std::string &strExchange,
        const std::string &strTTExchange,
        const std::string &strSecurityType,
        const std::string &strTTSymbol,
        double dTTScaleMultiplier,
        const std::string &strTDSymbol,
        double dTDScaleMultiplier,
        unsigned int iLotsPerMin,
        unsigned int iTimeSliceSecs,
        unsigned int iDiscVal,
        unsigned int iVariance,
        unsigned int iLimitTicksAway,
        const std::string &strContractExchSymbol,
        const std::string &strContractMonth,
        const std::string &strTTSecID)
        : m_iSymbolID(iSymbolID),
          m_strSymbol(strSymbol),
          m_objBaseCcy(objBaseCcy),
          m_objRefCcy(objRefCcy),
          m_dPointValue(dPointValue),
          m_dTickSize(dTickSize),
          m_iAssetClass(iAssetClass),
          m_bTransmitToGUI(bTransmitToGUI),
          m_strExchSymbol(strExchSymbol),
          m_strExchange(strExchange),
          m_strTTExchange(strTTExchange),
          m_strSecurityType(strSecurityType),
          m_strTTSymbol(strTTSymbol),
          m_dTTScaleMultiplier(dTTScaleMultiplier),
          m_strTTSecID(strTTSecID),
          m_strContractExchSymbol(strContractExchSymbol),
          m_strContractTicker(strContractMonth),
          m_strContractMonth(strContractMonth),
          m_strTDSymbol(strTDSymbol),
          m_dTDScaleMultiplier(dTDScaleMultiplier),
          m_iLotsPerMin(iLotsPerMin),
          m_iTimeSliceSecs(iTimeSliceSecs),
          m_iDiscVal(iDiscVal),
          m_iVariance(iVariance),
          m_iLimitTicksAway(iLimitTicksAway) {
}

// This doesn't include the contract month
const CSymbol &CSymbol::getSymbolFromExchSymbol(const std::string &strExchSymbol) {

    SymbolMapByName::const_iterator iter = std::find_if(m_SymbolsByName.begin(), m_SymbolsByName.end(),
                                                        [strExchSymbol](SymbolMapByName::value_type &x) {
                                                            return x.second->getExchSymbol().compare(strExchSymbol) ==
                                                                   0;
                                                        });

    if (iter != m_SymbolsByName.end())
        return *iter->second;

    // give it another try
    return getSymbolFromExchContract(strExchSymbol);
}

const CSymbol &CSymbol::getSymbolFromExchContract(const std::string &strExchContract) {
    auto iter = m_SymbolsByExchContract.find(strExchContract);
    if (iter != m_SymbolsByExchContract.end()) {
        return *iter->second;
    };

    fprintf(stderr, "cannot find ExchContract %s\n", strExchContract.c_str());
    throw Mts::Exception::CMtsException("missing exchage symbol");
}

const std::string CSymbol::getExchContractByMTSSymbol(const std::string &sym) {
    return getSymbol(sym).getContractExchSymbol();
}

const std::string CSymbol::getMTSSymbolByExchContract(const std::string &exch_contract) {
    return getSymbolFromExchContract(exch_contract).getSymbol();
}

const CSymbol &CSymbol::getSymbolFromTTSecID(const std::string &strTTSecID) {

    SymbolMapByName::const_iterator iter = std::find_if(m_SymbolsByName.begin(), m_SymbolsByName.end(),
                                                        [strTTSecID](SymbolMapByName::value_type &x) {
                                                            return x.second->getTTSecID().compare(strTTSecID) == 0;
                                                        });

    if (iter != m_SymbolsByName.end())
        return *iter->second;

    throw Mts::Exception::CMtsException("Missing TTSecID");
}


void CSymbol::clear() {

    m_SymbolsByName.clear();
    m_SymbolsByID.clear();
}

bool CSymbol::isHomogenous(Mts::Core::CSymbol::AssetClass iAssetClass) {
    bool bRet = std::all_of(m_SymbolsByName.begin(), m_SymbolsByName.end(),
                            [iAssetClass](SymbolMapByName::value_type &x) {
                                return x.second->getAssetClass() == iAssetClass;
                            });
    return bRet;
}


const CSymbol &CSymbol::getSymbolFromTDSymbol(const std::string &strTDSymbol) {
    auto iter = std::find_if(m_SymbolsByName.begin(), m_SymbolsByName.end(),
                             [strTDSymbol](SymbolMapByName::value_type &x) {
                                 return x.second->getTDSymbol().compare(strTDSymbol) == 0;
                             });

    if (iter == m_SymbolsByName.end())
        throw Mts::Exception::CMtsException("missing td symbol definition");

    return *iter->second;
}


bool CSymbol::isWithinTradingWindow(const Mts::Core::CDateTime &dtNow) const {
    // TODO - not used, to be removed
    return false;
}


bool CSymbol::isWithinSettWindow(const Mts::Core::CDateTime &dtNow) const {
    return false;
}


unsigned int CSymbol::getDiscVal() const {

    return m_iDiscVal;
}


unsigned int CSymbol::getVariance() const {

    return m_iVariance;
}


unsigned int CSymbol::getLimitTicksAway() const {

    return m_iLimitTicksAway;
}


void CSymbol::mapTTSecID(unsigned int iSymbolID,
        const std::string &strTTSecID) {

    m_SymbolsByID.find(iSymbolID)->second->setTTSecID(strTTSecID);
}

