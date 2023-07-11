#include "CConsolidatedPositionManager.h"


using namespace Mts::Accounting;


CConsolidatedPositionManager::CConsolidatedPositionManager() {

    memset(m_CcyPosition, 0, Mts::Core::CConfig::MAX_NUM_CCYS * sizeof(double));
}


const CPosition &CConsolidatedPositionManager::getPosition(unsigned int iAlgoID,
        unsigned int iSymbolID) {

    return m_PositionManagers[iAlgoID].getPosition(Mts::Core::CSymbol::getSymbol(iSymbolID));
}


const CPosition &CConsolidatedPositionManager::getPosition(boost::shared_ptr<CAlgorithm> ptrAlgorithm,
        const Mts::Core::CSymbol &objSymbol) {

    return m_PositionManagers[ptrAlgorithm->getAlgoID()].getPosition(objSymbol);
}


double CConsolidatedPositionManager::getCcyPosition(const Mts::Core::CCurrncy &objCcy) const {

    return m_CcyPosition[objCcy.getCcyID()];
}


void CConsolidatedPositionManager::updatePosition(const Mts::Order::COrderFill &objFill) {

    m_PositionManagers[objFill.getOrigOrder().getOriginatorAlgoID()].updatePosition(objFill);

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objFill.getOrigOrder().getSymbolID());

    double dTradeSign = objFill.getBuySell() == Mts::Order::COrder::BUY ? 1 : -1;
    double dBaseCcyQty = dTradeSign * static_cast<double>(objFill.getFillQuantity());
    double dRefCcyQty = -dTradeSign * static_cast<double>(objFill.getFillQuantity()) * objFill.getFillPrice();

    int iBaseCcyID = objSymbol.getBaseCcy().getCcyID();
    int iRefCcyID = objSymbol.getRefCcy().getCcyID();

    m_CcyPosition[iBaseCcyID] += dBaseCcyQty;
    m_CcyPosition[iRefCcyID] += dRefCcyQty;
}


