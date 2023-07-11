#include "CModel.h"
#include "CMtsException.h"


using namespace Mts::Model;


CModel::CModel(const std::string &strModelName,
        unsigned int iBarSizeMin,
        unsigned int iFormationPeriod)
        : m_strModelName(strModelName),
          m_iBarSizeMin(iBarSizeMin),
          m_iFormationPeriod(iFormationPeriod),
          m_PositionManager(strModelName),
          m_bOHLCHistComplete(false) {

    memset(m_iDesiredPos, 0, sizeof(int) * Mts::Core::CConfig::MAX_NUM_SYMBOLS);
}


bool CModel::initialize() {

    return true;
}


void CModel::onEvent(const Mts::Order::COrderFill &objOrderFill) {

    m_PositionManager.updatePosition(objOrderFill);
}


void CModel::onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) {

    m_PositionManager.updatePnL(objBidAsk);

    auto iter = m_Symbol2OHLCBars.find(objBidAsk.getSymbolID());

    if (iter == m_Symbol2OHLCBars.end())
        return;

    iter->second.update(objBidAsk);
}


void CModel::onEvent(const Mts::OrderBook::CKeyValue &objKeyValue) {

}


const std::string &CModel::getModelName() const {

    return m_strModelName;
}


int CModel::getCurrPosition(unsigned int iSymbolID) const {

    return m_PositionManager.getPosition(iSymbolID).getPosition();
}


int CModel::getDesiredPosition(unsigned int iSymbolID) const {

    return m_iDesiredPos[iSymbolID];
}


void CModel::setDesiredPosition(unsigned int iSymbolID,
        int iDesiredPos) {

    m_iDesiredPos[iSymbolID] = iDesiredPos;
}


void CModel::addTradedInstrument(const Mts::Core::CSymbol &objSymbol) {

    m_TradedInstruments.push_back(objSymbol);

    m_Symbol2OHLCBars.insert(std::pair<unsigned int, Mts::Indicator::COHLCBarHist>(objSymbol.getSymbolID(),
                                                                                   Mts::Indicator::COHLCBarHist(
                                                                                           m_iBarSizeMin,
                                                                                           m_iFormationPeriod)));
}


const Mts::Indicator::COHLCBarHist &CModel::getOHLCBars(unsigned int iSymbolID) {

    auto iter = m_Symbol2OHLCBars.find(iSymbolID);

    if (iter == m_Symbol2OHLCBars.end())
        throw Mts::Exception::CMtsException("Error: Model does not traded requested instrument");

    return iter->second;
}


double CModel::getPnLUSD() const {

    return m_PositionManager.getTotalUSDPnL();
}


unsigned int CModel::getNumFills() const {

    unsigned int iCount = 0;

    for (auto &objSymbol: m_TradedInstruments) {

        iCount += m_PositionManager.getNumFills(objSymbol.getSymbolID());
    }

    return iCount;
}


unsigned int CModel::getNumTradedInstruments() const {

    return m_TradedInstruments.size();
}


const Mts::Core::CSymbol &CModel::getTradedInstrument(unsigned int iIndex) const {

    return m_TradedInstruments[iIndex];
}


bool CModel::isOHLCHistComplete() {

    if (m_bOHLCHistComplete == true)
        return true;

    for (auto objInst: m_TradedInstruments) {

        if (m_Symbol2OHLCBars.find(objInst.getSymbolID())->second.isFull() == false)
            return false;
    }

    m_bOHLCHistComplete = true;

    return true;
}


