#include "CHeartbeat.h"


using namespace Mts::LifeCycle;


CHeartbeat::CHeartbeat()
        : m_iProviderID(0),
          m_bFeedFlag(true),
          m_dJulianDateTime(0),
          m_bActiveFlag(false) {

}


CHeartbeat::CHeartbeat(unsigned int iProviderID,
        bool bFeedFlag,
        double dJulianDateTime)
        : m_iProviderID(iProviderID),
          m_bFeedFlag(bFeedFlag),
          m_dJulianDateTime(dJulianDateTime),
          m_bActiveFlag(false) {

}


unsigned int CHeartbeat::getProviderID() const {

    return m_iProviderID;
}


void CHeartbeat::setProviderID(unsigned int iProviderID) {
    m_iProviderID = iProviderID;
}


bool CHeartbeat::isFeed() const {

    return m_bFeedFlag;
}


void CHeartbeat::setIsFeed(bool bFeedFlag) {

    m_bFeedFlag = bFeedFlag;
}


double CHeartbeat::getJulianDateTime() const {

    return m_dJulianDateTime;
}


void CHeartbeat::setJulianDateTime(double dJulianDateTime) {

    m_dJulianDateTime = dJulianDateTime;
}


bool CHeartbeat::isActive() const {

    return m_bActiveFlag;
}


void CHeartbeat::setActiveFlag(bool bActiveFlag) {

    m_bActiveFlag = bActiveFlag;
}

