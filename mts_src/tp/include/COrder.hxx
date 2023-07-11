using namespace Mts::Order;


inline unsigned int COrder::getOriginatorAlgoID() const {

    return m_iOriginatorAlgoID;
}


inline void COrder::setOriginatorAlgoID(unsigned int iOriginatorAlgoID) {

    m_iOriginatorAlgoID = iOriginatorAlgoID;
}


inline const char *COrder::getMtsOrderID() const {

    return m_szMtsOrderID;
}


inline void COrder::setMtsOrderID(const char *pszMtsOrderID) {

    strcpy(m_szMtsOrderID, pszMtsOrderID);
}


inline const char *COrder::getExcOrderID() const {

    return m_szExcOrderID;
}


inline void COrder::setExcOrderID(const char *pszExcOrderID) {

    strcpy(m_szExcOrderID, pszExcOrderID);
}


inline const char *COrder::getExcExecID() const {

    return m_szExcExecID;
}


inline void COrder::setExcExecID(const char *pszExcExecID) {

    strcpy(m_szExcExecID, pszExcExecID);
}


inline const char *COrder::getOrderTag() const {

    return m_szOrderTag;
}


inline void COrder::setOrderTag(const char *pszOrderTag) {

    strcpy(m_szOrderTag, pszOrderTag);
}


inline Mts::Core::CDateTime COrder::getCreateTimestamp() const {

    return m_dtCreateTimestamp;
}


inline void COrder::setCreateTimestamp(const Mts::Core::CDateTime &dtCreateTimestamp) {

    m_dtCreateTimestamp = dtCreateTimestamp;
}


inline Mts::Core::CDateTime COrder::getMtsTimestamp() const {

    return m_dtMtsTimestamp;
}


inline void COrder::setMtsTimestamp(const Mts::Core::CDateTime &dtMtsTimestamp) {

    m_dtMtsTimestamp = dtMtsTimestamp;
}


inline Mts::Core::CDateTime COrder::getExcTimestamp() const {

    return m_dtExcTimestamp;
}


inline void COrder::setExcTimestamp(const Mts::Core::CDateTime &dtExcTimestamp) {

    m_dtExcTimestamp = dtExcTimestamp;
}


inline unsigned int COrder::getSymbolID() const {

    return m_iSymbolID;
}


inline void COrder::setSymbolID(unsigned int iSymbolID) {

    m_iSymbolID = iSymbolID;
}


inline unsigned int COrder::getProviderID() const {

    return m_iProviderID;
}


inline void COrder::setProviderID(unsigned int iProviderID) {

    m_iProviderID = iProviderID;
}


inline Mts::Order::COrder::BuySell COrder::getDirection() const {

    return m_iDirection;
}


inline std::string COrder::getDirectionString() const {

    return m_iDirection == Mts::Order::COrder::BUY ? "B" : "S";
}


inline void COrder::setDirection(Mts::Order::COrder::BuySell iDirection) {

    m_iDirection = iDirection;
}


inline unsigned int COrder::getQuantity() const {

    return m_uiQuantity;
}


inline void COrder::setQuantity(unsigned int uiQuantity) {

    m_uiQuantity = uiQuantity;
}


inline double COrder::getPrice() const {

    return m_dPrice;
}


inline void COrder::setPrice(double dPrice) {

    m_dPrice = dPrice;
}


inline COrder::OrderType COrder::getOrderType() const {

    return m_iOrderType;
}


inline std::string COrder::getOrderTypeString() const {

    return "TWAP";
}


inline void COrder::setOrderType(OrderType iOrderType) {

    m_iOrderType = iOrderType;
}


inline COrder::OrderState COrder::getOrderState() const {

    return m_iOrderState;
}


inline void COrder::setOrderState(OrderState iOrderState) {

    m_iOrderState = iOrderState;
}


inline const char *COrder::getExecBrokerCode() const {

    return m_szExecBrokerCode;
}


inline void COrder::setExecBrokerCode(const char *pszExecBrokerCode) {

    strcpy(m_szExecBrokerCode, pszExecBrokerCode);
}


inline unsigned int COrder::getTotalFilledQty() const {

    return m_uiTotalFilledQty;
}


inline void COrder::updateTotalFilledQty(unsigned int iFillQty) {

    m_uiTotalFilledQty += iFillQty;
}


inline Mts::Core::CDateTime COrder::getLastFillTimestamp() const {

    return m_dtLastFillTimestamp;
}


inline void COrder::setLastFillTimestamp(const Mts::Core::CDateTime &dtLastFillTimestamp) {

    m_dtLastFillTimestamp = dtLastFillTimestamp;
}



