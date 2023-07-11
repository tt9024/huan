#include "COrder.h"


using namespace Mts::Order;


COrder::COrder()
        : m_iOriginatorAlgoID(0),
          m_dtCreateTimestamp(0),
          m_dtMtsTimestamp(0),
          m_dtExcTimestamp(0),
          m_iSymbolID(0),
          m_iProviderID(0),
          m_iDirection(BUY),
          m_uiQuantity(0),
          m_iOrderType(IOC),
          m_dPrice(0),
          m_iOrderState(NEW),
          m_dtLastFillTimestamp(0),
          m_uiTotalFilledQty(0) {

    strcpy(m_szMtsOrderID, "");
    strcpy(m_szExcOrderID, "");
    strcpy(m_szExcExecID, "");
    strcpy(m_szOrderTag, "");
    strcpy(m_szExecBrokerCode, "");
}


COrder::COrder(unsigned int iOriginatorAlgoID,
        const char *pszMtsOrderID,
        const Mts::Core::CDateTime &dtCreateTimestamp,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        BuySell iDirection,
        unsigned int uiQuantity,
        OrderType iOrderType,
        double dPrice,
        const char *pszExecBrokerCode)
        : m_iOriginatorAlgoID(iOriginatorAlgoID),
          m_dtCreateTimestamp(dtCreateTimestamp),
          m_dtMtsTimestamp(0),
          m_dtExcTimestamp(0),
          m_iSymbolID(iSymbolID),
          m_iProviderID(iProviderID),
          m_iDirection(iDirection),
          m_uiQuantity(uiQuantity),
          m_iOrderType(iOrderType),
          m_dPrice(dPrice),
          m_iOrderState(NEW),
          m_dtLastFillTimestamp(0),
          m_uiTotalFilledQty(0) {

    strcpy(m_szMtsOrderID, pszMtsOrderID);
    strcpy(m_szExcOrderID, "");
    strcpy(m_szExcExecID, "");
    strcpy(m_szOrderTag, "");
    strcpy(m_szExecBrokerCode, pszExecBrokerCode);
}


std::string COrder::getOrderStateDescription() const {

    std::string strDescription;

    switch (m_iOrderState) {
        case NEW:
            strDescription = "New";
            break;

        case PARTIALLY_FILLED:
            strDescription = "Partial Fill";
            break;

        case FILLED:
            strDescription = "Filled";
            break;

        case CANCEL_PENDING:
            strDescription = "Cancel Pending";
            break;

        case CANCELLED:
            strDescription = "Cancelled";
            break;

        case REPLACE_PENDING:
            strDescription = "Replace Pending";
            break;

        case REPLACED:
            strDescription = "Replaced";
            break;

        case REJECTED:
            strDescription = "Rejected";
            break;

        case LIMIT_VIOLATION:
            strDescription = "Rejected Internal";
            break;
    }

    return strDescription;
}


std::string COrder::getDescription() const {

    char szBuffer[255];
    const char *pszBuySell = m_iDirection == COrder::BUY ? "B" : "S";
    const char *pszOrderType = m_iOrderType == COrder::IOC ? "IOC" : "GTC";
    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(m_iSymbolID);

    sprintf(szBuffer, "%s %d @ %.5f %s", pszBuySell, m_uiQuantity, m_dPrice, pszOrderType);

    return szBuffer;
}


std::string COrder::toString() const {

    char szBuffer[512];
    const char *pszBuySell = m_iDirection == COrder::BUY ? "B" : "S";
    const char *pszOrderType = m_iOrderType == COrder::IOC ? "IOC" : "GTC";
    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(m_iSymbolID);
    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider(m_iProviderID);

    snprintf(szBuffer, sizeof(szBuffer),
             "%llu ORDER CREATED: STATE=%s {%s %s %s %d @ %.5f %s provider=%s tag=%s}",
             m_dtCreateTimestamp.getCMTime(),
             orderstate2String(m_iOrderState).c_str(),
             m_szMtsOrderID,
             objSymbol.getSymbol().c_str(),
             pszBuySell,
             m_uiQuantity,
             m_dPrice,
             pszOrderType,
             objProvider.getName().c_str(),
             m_szOrderTag);

    return szBuffer;
}


std::string COrder::orderstate2String(OrderState iOrdState) {

    switch (iOrdState) {
        case NEW:
            return "New";

        case PARTIALLY_FILLED:
            return "Partial Fill";

        case FILLED:
            return "Filled";

        case CANCEL_PENDING:
            return "Cancel Pending";

        case CANCELLED:
            return "Cancelled";

        case REPLACE_PENDING:
            return "Replace Pending";

        case REPLACED:
            return "Replaced";

        case REJECTED:
            return "Rejected";

        case LIMIT_VIOLATION:
            return "Rejected Internal";
    }

    return "Unknown order state";
}


