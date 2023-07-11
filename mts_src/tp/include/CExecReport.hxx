#include <cstring>


using namespace Mts::Order;


inline unsigned int CExecReport::getOriginatorAlgoID() const {

    return m_iOriginatorAlgoID;
}


inline const char *CExecReport::getParentOrderID() const {

    return m_szParentOrderID;
}


inline const char *CExecReport::getTicker() const {

    return m_szTicker;
}


inline const char *CExecReport::getBuySell() const {

    return m_szBuySell;
}


inline unsigned int CExecReport::getOrigQty() const {

    return m_iOrigQty;
}


inline const Mts::Core::CDateTime &CExecReport::getCreateTime() const {

    return m_dtCreateTime;
}


inline const char *CExecReport::getAlgorithm() const {

    return m_szAlgorithm;
}


inline const char *CExecReport::getParameters() const {

    return m_szParameters;
}


inline unsigned int CExecReport::getLastFillQty() const {

    return m_iLastFillQty;
}


inline double CExecReport::getLastFillPx() const {

    return m_dLastFillPx;
}


inline const char *CExecReport::getLastMkt() const {

    return m_szLastMkt;
}


inline unsigned int CExecReport::getTotalFillQty() const {

    return m_iTotalFillQty;
}


inline double CExecReport::getAvgFillPx() const {

    return m_dAvgFillPx;
}


inline unsigned int CExecReport::getRemainingQty() const {

    return m_iRemainingQty;
}


inline COrder::OrderState CExecReport::getMtsOrderState() const {

    return m_iMtsOrderState;
}


inline const char *CExecReport::getStatus() const {

    return m_szStatus;
}


inline const Mts::Core::CDateTime &CExecReport::getUpdateTime() const {

    return m_dtUpdateTime;
}


inline void CExecReport::setOriginatorAlgoID(unsigned int iOriginatorAlgoID) {

    m_iOriginatorAlgoID = iOriginatorAlgoID;
}


inline void CExecReport::setParentOrderID(const char *pszParentOrderID) {

    strcpy(m_szParentOrderID, pszParentOrderID);
}


inline void CExecReport::setTicker(const char *pszTicker) {

    strcpy(m_szTicker, pszTicker);
}


inline void CExecReport::setBuySell(const char *pszBuySell) {

    strcpy(m_szBuySell, pszBuySell);
}


inline void CExecReport::setOrigQty(unsigned int iOrigQty) {

    m_iOrigQty = iOrigQty;
}


inline void CExecReport::setCreateTime(const Mts::Core::CDateTime &dtCreateTime) {

    m_dtCreateTime = dtCreateTime;
}


inline void CExecReport::setAlgorithm(const char *pszAlgorithm) {

    strcpy(m_szAlgorithm, pszAlgorithm);
}


inline void CExecReport::setParameters(const char *pszParameters) {

    strcpy(m_szParameters, pszParameters);
}


inline void CExecReport::setLastFillQty(unsigned int iLastFillQty) {

    m_iLastFillQty = iLastFillQty;
}


inline void CExecReport::setLastFillPx(double dLastFillPx) {

    m_dLastFillPx = dLastFillPx;
}


inline void CExecReport::setLastMkt(const char *pszLastMkt) {

    strcpy(m_szLastMkt, pszLastMkt);
}


inline void CExecReport::setTotalFillQty(unsigned int iTotalFillQty) {

    m_iTotalFillQty = iTotalFillQty;
}


inline void CExecReport::setAvgFillPx(double dAvgFillPx) {

    m_dAvgFillPx = dAvgFillPx;
}


inline void CExecReport::setRemainingQty(unsigned int iRemainingQty) {

    m_iRemainingQty = iRemainingQty;
}


inline void CExecReport::setMtsOrderState(COrder::OrderState iMtsOrderState) {

    m_iMtsOrderState = iMtsOrderState;
}


inline void CExecReport::setStatus(const char *pszStatus) {

    strcpy(m_szStatus, pszStatus);
}


inline void CExecReport::setUpdateTime(const Mts::Core::CDateTime &dtUpdateTime) {

    m_dtUpdateTime = dtUpdateTime;
}



