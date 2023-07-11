#include <boost/lexical_cast.hpp>
#include "COrderFill.h"


using namespace Mts::Order;


COrderFill::COrderFill()
        : CEvent(Mts::Event::CEvent::FILL) {

}


COrderFill::COrderFill(const COrder &objOrder,
        const Mts::Core::CDateTime &dtFillTimestamp,
        unsigned int uiFillQuantity,
        double dFillPrice,
        const Mts::Core::CDateTime &dtSettDate)
        : CEvent(Mts::Event::CEvent::FILL),
          m_OrigOrder(objOrder),
          m_dtFillTimestamp(dtFillTimestamp),
          m_uiFillQuantity(uiFillQuantity),
          m_dFillPrice(dFillPrice),
          m_dtSettDate(dtSettDate),
          m_bRecoveryFill(false) {

}


const COrder &COrderFill::getOrigOrder() const {

    return m_OrigOrder;
}


void COrderFill::setOrigOrder(const COrder &objOrigOrder) {

    m_OrigOrder = objOrigOrder;
}


Mts::Core::CDateTime COrderFill::getFillTimestamp() const {

    return m_dtFillTimestamp;
}


void COrderFill::setFillTimestamp(Mts::Core::CDateTime &dtFillTimestamp) {

    m_dtFillTimestamp = dtFillTimestamp;
}


Mts::Order::COrder::BuySell COrderFill::getBuySell() const {

    return m_OrigOrder.getDirection();
}


unsigned int COrderFill::getFillQuantity() const {

    return m_uiFillQuantity;
}


void COrderFill::setFillQuantity(unsigned int uiFillQuantity) {

    m_uiFillQuantity = uiFillQuantity;
}


double COrderFill::getFillPrice() const {

    return m_dFillPrice;
}


void COrderFill::setFillPrice(double dFillPrice) {

    m_dFillPrice = dFillPrice;
}


Mts::Core::CDateTime COrderFill::getSettDate() const {

    return m_dtSettDate;
}


void COrderFill::setSettDate(Mts::Core::CDateTime &dtSettDate) {

    m_dtSettDate = dtSettDate;
}


std::string COrderFill::toString() const {

    char szBuffer[512];
    const char *pszBuySell = m_OrigOrder.getDirection() == COrder::BUY ? "B" : "S";
    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(m_OrigOrder.getSymbolID());
    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider(m_OrigOrder.getProviderID());

    snprintf(szBuffer, sizeof(szBuffer), "%llu ORDER FILLED: %s %s %s %d @ %.5f %s %s %s\n",
             m_dtFillTimestamp.getCMTime(), m_OrigOrder.getMtsOrderID(), objSymbol.getSymbol().c_str(), pszBuySell,
             m_uiFillQuantity, m_dFillPrice, objProvider.getName().c_str(), m_dtSettDate.toString().c_str(),
             m_OrigOrder.getExecBrokerCode());

    return szBuffer;
}


void COrderFill::createBroadcastMessage(char *pszBuffer) const {

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(getOrigOrder().getSymbolID());

    sprintf(pszBuffer, "%s,%s,%d,%0.5f,%llu,%llu,%s,%d", objSymbol.getSymbol().c_str(),
            getOrigOrder().getDirectionString().c_str(),
            getFillQuantity(),
            getFillPrice(),
            getOrigOrder().getCreateTimestamp().getCMTime(),
            getFillTimestamp().getCMTime(),
            getOrigOrder().getMtsOrderID(),
            getOrigOrder().getOriginatorAlgoID());
}


bool COrderFill::isRecoveryFill() const {
    return m_bRecoveryFill;
}


void COrderFill::setRecoveryFill(bool bRecoveryFill) {
    m_bRecoveryFill = bRecoveryFill;
}

