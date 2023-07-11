#include "COrderCancelRequest.h"


using namespace Mts::Order;


COrderCancelRequest::COrderCancelRequest()
        : m_iOriginatorAlgoID(0),
          m_dtCreateTimestamp(0) {

    strcpy(m_szMtsCancelReqID, "");
}


COrderCancelRequest::COrderCancelRequest(unsigned int iOriginatorAlgoID,
        const char *pszMtsCancelReqID,
        const Mts::Core::CDateTime &dtCreateTimestamp,
        const COrder &objOrigOrder)
        : m_iOriginatorAlgoID(iOriginatorAlgoID),
          m_dtCreateTimestamp(dtCreateTimestamp),
          m_OrigOrder(objOrigOrder) {

    strcpy(m_szMtsCancelReqID, pszMtsCancelReqID);
}


unsigned int COrderCancelRequest::getOriginatorAlgoID() const {

    return m_iOriginatorAlgoID;
}


void COrderCancelRequest::setOriginatorAlgoID(unsigned int iOriginatorAlgoID) {

    m_iOriginatorAlgoID = iOriginatorAlgoID;
}


const char *COrderCancelRequest::getMtsCancelReqID() const {

    return m_szMtsCancelReqID;
}


void COrderCancelRequest::setMtsCancelReqID(const char *pszMtsCancelReqID) {

    strcpy(m_szMtsCancelReqID, pszMtsCancelReqID);
}


const Mts::Core::CDateTime &COrderCancelRequest::getCreateTimestamp() const {

    return m_dtCreateTimestamp;
}


void COrderCancelRequest::setCreateTimestamp(const Mts::Core::CDateTime &dtCreateTimestamp) {

    m_dtCreateTimestamp = dtCreateTimestamp;
}


const COrder &COrderCancelRequest::getOrigOrder() const {

    return m_OrigOrder;
}


void COrderCancelRequest::setOrigOrder(const Mts::Order::COrder &objOrigOrder) {

    m_OrigOrder = objOrigOrder;
}


void COrderCancelRequest::cancelAccepted() {

    m_OrigOrder.setOrderState(COrder::CANCELLED);
}


void COrderCancelRequest::cancelRejected() {

    m_OrigOrder.setOrderState(COrder::REJECTED);
}


std::string COrderCancelRequest::getDescription() const {

    char szBuffer[255];
    const char *pszBuySell = getOrigOrder().getDirection() == COrder::BUY ? "B" : "S";
    const char *pszOrderType = getOrigOrder().getOrderType() == COrder::IOC ? "IOC" : "GTC";
    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(getOrigOrder().getSymbolID());
    unsigned long uiQuantity = getOrigOrder().getQuantity();
    double dPrice = getOrigOrder().getPrice();

    sprintf(szBuffer, "%llu CANCEL REQUEST %s %s %s %d @ %.5f %s %s", m_dtCreateTimestamp.getCMTime(),
            getMtsCancelReqID(), objSymbol.getSymbol().c_str(), pszBuySell, uiQuantity, dPrice, pszOrderType,
            getOrigOrder().getMtsOrderID());

    return szBuffer;
}


std::string COrderCancelRequest::toString() const {

    return getDescription();
}


