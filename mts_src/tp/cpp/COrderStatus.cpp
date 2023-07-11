#include "COrderStatus.h"


using namespace Mts::Order;


COrderStatus::COrderStatus()
        : CEvent(Mts::Event::CEvent::ORDER_STATUS) {

}


COrderStatus::COrderStatus(const COrder &objOrder,
        const Mts::Core::CDateTime &dtCreateTime)
        : CEvent(Mts::Event::CEvent::ORDER_STATUS),
          m_OrigOrder(objOrder),
          m_dtCreateTimestamp(dtCreateTime) {

}


const COrder &COrderStatus::getOrigOrder() const {

    return m_OrigOrder;
}


void COrderStatus::setOrigOrder(const COrder &objOrigOrder) {

    m_OrigOrder = objOrigOrder;
}


Mts::Core::CDateTime COrderStatus::getCreateTimestamp() const {

    return m_dtCreateTimestamp;
}


void COrderStatus::setCreateTimestamp(const Mts::Core::CDateTime &dtCreateTimestamp) {

    m_dtCreateTimestamp = dtCreateTimestamp;
}


COrder::OrderState COrderStatus::getStatus() const {

    return m_OrigOrder.getOrderState();
}


void COrderStatus::setStatus(Mts::Order::COrder::OrderState iOrderState) {

    m_OrigOrder.setOrderState(iOrderState);
}


std::string COrderStatus::toString() const {

    char szBuffer[255];
    std::string strDescription = m_OrigOrder.getOrderStateDescription();

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(m_OrigOrder.getSymbolID());
    sprintf(szBuffer, "%llu Status: %s orgOrdId=%s provider=%d\n", m_dtCreateTimestamp.getCMTime(),
            strDescription.c_str(), m_OrigOrder.getMtsOrderID(), m_OrigOrder.getProviderID());

    return szBuffer;
}



