using namespace Mts::Core;


inline unsigned int CDateTime::getHour() const {

    return m_iHour;
}


inline unsigned int CDateTime::getMin() const {

    return m_iMin;
}


inline unsigned int CDateTime::getSec() const {

    return m_iSec;
}


inline unsigned int CDateTime::getMSec() const {

    return m_iMSec;
}


inline double CDateTime::getValue() const {
    return m_dJulianDate;
}


inline double CDateTime::getTimeValue() const {
    return m_dJulianDate - floor(m_dJulianDate);
}


inline unsigned long long CDateTime::getCMTime() const {
    return m_ullCMTime;
}


inline unsigned int CDateTime::getJulianDay() const {

    return static_cast<unsigned int>(floor(m_dJulianDate));
}


inline unsigned int CDateTime::getDay() const {

    return m_iDay;
}


inline unsigned int CDateTime::getMonth() const {

    return m_iMonth;
}


inline unsigned int CDateTime::getYear() const {

    return m_iYear;
}


inline unsigned int CDateTime::getDayOfWeek() const {

    return m_iDayOfWeek;
}


// 1=sun, 2=mon etc.
inline unsigned int CDateTime::calcDayOfWeek() const {

    unsigned int iDay = m_iDay;
    unsigned int iMonth = m_iMonth;
    unsigned int iYear = m_iYear;

    int iRes = (iDay += iMonth < 3 ? iYear-- : iYear - 2, 23 * iMonth / 9 + iDay + 4 + iYear / 4 - iYear / 100 +
                                                          iYear / 400) % 7;

    // bump to sync with matlab weekday output
    ++iRes;

    return iRes;
}


inline double CDateTime::getMatlabDateNum() const {

    return m_dJulianDate + MATLABBasisAdjustment;
}


inline unsigned int CDateTime::getTimeInMin() const {

    return m_iHour * 60 + m_iMin;
}


inline unsigned int CDateTime::getTimeInSec() const {

    return m_iHour * 3600 + m_iMin * 60 + m_iSec;
}

