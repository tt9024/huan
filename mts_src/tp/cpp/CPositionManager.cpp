#include "CPositionManager.h"
#include "CSymbol.h"
#include <iostream>


using namespace Mts::Accounting;


CPositionManager::CPositionManager()
        : m_dtLastUpdate(0),
          m_strName("") {

    initialize();
}


CPositionManager::CPositionManager(const std::string &strName)
        : m_dtLastUpdate(0),
          m_strName(strName) {

    initialize();
}


const CPosition &CPositionManager::getPosition(unsigned int iSymbolID) const {

    auto iter = m_SymbolPositions.find(iSymbolID);

    return iter->second;
}


const CPosition &CPositionManager::getPosition(const Mts::Core::CSymbol &objSymbol) const {

    auto iter = m_SymbolPositions.find(objSymbol.getSymbolID());

    return iter->second;
}


void CPositionManager::setPosition(const Mts::Core::CSymbol &objSymbol,
        const CPosition &objPosition) {

    //m_SymbolPositions.insert(std::pair<unsigned int, Mts::Accounting::CPosition>(objSymbol.getSymbolID(), objPosition));
    //m_SymbolPositions[objSymbol.getSymbolID()] = objPosition;

    auto iter = m_SymbolPositions.find(objSymbol.getSymbolID());

    if (iter == m_SymbolPositions.end())
        return;

    long position = objPosition.getPosition();
    Mts::Order::COrder::BuySell direction =
            position > 0 ? Mts::Order::COrder::BuySell::BUY : Mts::Order::COrder::BuySell::SELL;
    iter->second.updatePosition(direction, abs(position), objPosition.getWAP());
}


void CPositionManager::updatePosition(const Mts::Core::CSymbol &objSymbol,
        const Mts::Order::COrder::BuySell iDirection,
        unsigned int iQuantity,
        double dPrice) {

    auto iter = m_SymbolPositions.find(objSymbol.getSymbolID());

    if (iter == m_SymbolPositions.end())
        return;

    iter->second.updatePosition(iDirection, iQuantity, dPrice);
}


void CPositionManager::updatePosition(const Mts::Order::COrderFill &objFill) {

    auto iterPos = m_SymbolPositions.find(objFill.getOrigOrder().getSymbolID());

    if (iterPos == m_SymbolPositions.end())
        return;

    iterPos->second.updatePosition(objFill);

    auto iterFill = m_SymbolFills.find(objFill.getOrigOrder().getSymbolID());

    if (iterFill == m_SymbolFills.end())
        return;

    iterFill->second.push_back(objFill);

    if (objFill.getOrigOrder().getDirection() == Mts::Order::COrder::BUY)
        m_iNumBuys += objFill.getFillQuantity();
    else
        m_iNumSells += objFill.getFillQuantity();
}


void CPositionManager::initialize() {

    const std::vector<boost::shared_ptr<const Mts::Core::CSymbol> > symbols = Mts::Core::CSymbol::getSymbols();

    for (int i = 0; i != symbols.size(); ++i) {

        m_SymbolPositions.insert(
                std::pair<unsigned int, Mts::Accounting::CPosition>(symbols[i]->getSymbolID(), CPosition(*symbols[i])));
        m_SymbolFills.insert(std::pair<unsigned int, std::vector<Mts::Order::COrderFill>>(symbols[i]->getSymbolID(),
                                                                                          std::vector<Mts::Order::COrderFill>()));
    }

    m_iNumBuys = 0;
    m_iNumSells = 0;
}


void CPositionManager::updatePnL(const Mts::OrderBook::CBidAsk &objBidAsk) {

    auto iter = m_SymbolPositions.find(objBidAsk.getSymbolID());

    if (iter == m_SymbolPositions.end())
        return;

    // if a new trading day, log the prior day closing PnL, reset trade count
    if (m_dtLastUpdate.getValue() > 0 && m_dtLastUpdate.equalDay(objBidAsk.getTimestamp()) == false) {

        double dPriorEODPnL = getTotalUSDPnL();
        m_dtLastUpdate.removeTime();

        m_DailyPnL.push_back(std::pair<Mts::Core::CDateTime, double>(m_dtLastUpdate, dPriorEODPnL));
        m_DailyBuyCount.push_back(std::pair<Mts::Core::CDateTime, unsigned int>(m_dtLastUpdate, m_iNumBuys));
        m_DailySellCount.push_back(std::pair<Mts::Core::CDateTime, unsigned int>(m_dtLastUpdate, m_iNumSells));

        m_iNumBuys = 0;
        m_iNumSells = 0;
    }

    iter->second.calcPnL(objBidAsk.getMidPx());

    m_dtLastUpdate = objBidAsk.getTimestamp();
}


double CPositionManager::getTotalUSDPnL() const {

    double dTotalPnL = 0.0;

    //std::for_each(m_SymbolPositions.begin(), m_SymbolPositions.end(), [&dTotalPnL](auto x){ dTotalPnL += x.second.getPnL();});

    return dTotalPnL;
}


void CPositionManager::printCurrentPositions() {

    auto iter = m_SymbolPositions.begin();
    for (; iter != m_SymbolPositions.end(); ++iter) {
        std::cout << iter->second.getSymbolID() << " Position: " << iter->second.getPosition() << std::endl;
    }
}


unsigned int CPositionManager::getNumFills(unsigned int iSymbolID) const {

    auto iterFill = m_SymbolFills.find(iSymbolID);

    if (iterFill == m_SymbolFills.end())
        return 0;
    else
        return static_cast<unsigned int>(iterFill->second.size());
}


std::vector<std::pair<Mts::Core::CDateTime, double>> CPositionManager::getDailyPnL() const {

    return m_DailyPnL;
}


std::vector<std::pair<Mts::Core::CDateTime, unsigned int>> CPositionManager::getDailyBuyCount() const {

    return m_DailyBuyCount;
}


std::vector<std::pair<Mts::Core::CDateTime, unsigned int>> CPositionManager::getDailySellCount() const {

    return m_DailySellCount;
}


const std::string &CPositionManager::getName() const {

    return m_strName;
}


std::vector<unsigned int> CPositionManager::getTradedInstruments() const {

    std::vector<unsigned int> objTradedInst;

    //std::for_each(m_SymbolPositions.begin(), m_SymbolPositions.end(), [&objTradedInst](auto x) { if (x.second.getGrossPosition() > 0) objTradedInst.push_back(x.first); });

    return objTradedInst;
}




