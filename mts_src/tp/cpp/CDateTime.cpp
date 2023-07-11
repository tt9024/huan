#include <time.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time/local_time_adjustor.hpp"
#include "boost/date_time/c_local_time_adjustor.hpp"
#include "CDateTime.h"
#include "CMath.h"
#include "CMtsException.h"


using namespace Mts::Core;


CDateTime::CDateTime()
        : m_iDay(0),
          m_iMonth(0),
          m_iYear(0),
          m_iHour(0),
          m_iMin(0),
          m_iSec(0),
          m_iMSec(0),
          m_dJulianDate(0),
          m_ullCMTime(0) {

}


CDateTime::CDateTime(unsigned int iYear,
        unsigned int iMonth,
        unsigned int iDay,
        unsigned int iHH,
        unsigned int iMM,
        unsigned int iSS,
        unsigned int iMS)
        : m_iHour(iHH),
          m_iMin(iMM),
          m_iSec(iSS),
          m_iMSec(iMS),
          m_iDay(iDay),
          m_iMonth(iMonth),
          m_iYear(iYear) {

    m_dJulianDate = calendar2Julian(iYear, iMonth, iDay);
    double dTS = time2DayFrac(iHH, iMM, iSS, iMS);

    m_dJulianDate += dTS;
    m_ullCMTime = time2CMTime(iYear, iMonth, iDay, iHH, iMM, iSS, iMS);
    m_iDayOfWeek = calcDayOfWeek();
}


CDateTime::CDateTime(const std::string &strCMTime) {

    const char *p = strCMTime.c_str();

    m_iYear = Mts::Math::CMath::atoi_4(p);
    m_iMonth = Mts::Math::CMath::atoi_2(p + 4);
    m_iDay = Mts::Math::CMath::atoi_2(p + 6);

    m_dJulianDate = calendar2Julian(m_iYear, m_iMonth, m_iDay);

    m_iHour = Mts::Math::CMath::atoi_2(p + 8);
    m_iMin = Mts::Math::CMath::atoi_2(p + 10);
    m_iSec = Mts::Math::CMath::atoi_2(p + 12);
    m_iMSec = Mts::Math::CMath::atoi_3(p + 14);

    double dTS = time2DayFrac(m_iHour, m_iMin, m_iSec, m_iMSec);

    m_dJulianDate += dTS;
    m_ullCMTime = time2CMTime(m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec, m_iMSec);
    m_iDayOfWeek = calcDayOfWeek();
}


// the expected date format is yyyy-mm-dd
// the expected time format is hh:mm:ss.000
CDateTime::CDateTime(const std::string &strDate,
        const std::string &strTime) {

    const char *p = strDate.c_str();
    m_iYear = Mts::Math::CMath::atoi_4(p);
    m_iMonth = Mts::Math::CMath::atoi_2(p + 5);
    m_iDay = Mts::Math::CMath::atoi_2(p + 8);

    m_dJulianDate = calendar2Julian(m_iYear, m_iMonth, m_iDay);

    p = strTime.c_str();
    m_iHour = Mts::Math::CMath::atoi_2(p);
    m_iMin = Mts::Math::CMath::atoi_2(p + 3);
    m_iSec = Mts::Math::CMath::atoi_2(p + 6);
    m_iMSec = Mts::Math::CMath::atoi_3(p + 9);

    double dTS = time2DayFrac(m_iHour, m_iMin, m_iSec, m_iMSec);

    m_dJulianDate += dTS;
    m_ullCMTime = time2CMTime(m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec, m_iMSec);
    m_iDayOfWeek = calcDayOfWeek();
}


// the expected format is yyyy-mm-dd hh:mm or hh:mm:ss
CDateTime::CDateTime(const std::string &strDateTime,
        unsigned int iDummy) {

    const char *p = strDateTime.c_str();
    m_iYear = Mts::Math::CMath::atoi_4(p);
    m_iMonth = Mts::Math::CMath::atoi_2(p + 5);
    m_iDay = Mts::Math::CMath::atoi_2(p + 8);

    m_dJulianDate = calendar2Julian(m_iYear, m_iMonth, m_iDay);

    m_iHour = Mts::Math::CMath::atoi_2(p + 11);
    m_iMin = Mts::Math::CMath::atoi_2(p + 14);
    m_iSec = iDummy == 0 ? 0 : Mts::Math::CMath::atoi_2(p + 17);
    m_iMSec = iDummy == 0 ? 0 : Mts::Math::CMath::atoi_2(p + 20);

    double dTS = time2DayFrac(m_iHour, m_iMin, m_iSec, m_iMSec);

    m_dJulianDate += dTS;
    m_ullCMTime = time2CMTime(m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec, m_iMSec);
    m_iDayOfWeek = calcDayOfWeek();
}


CDateTime::CDateTime(double dJulianDate)
        : m_dJulianDate(dJulianDate) {

    int iJD = static_cast<int>(floor(dJulianDate));
    double dDayFrac = dJulianDate - iJD;

    julian2Calendar(iJD, m_iYear, m_iMonth, m_iDay);
    dayFrac2Time(dDayFrac, m_iHour, m_iMin, m_iSec, m_iMSec);

    // special case, midnight so roll day
    if (m_iHour == 24 && m_iMin == 0 && m_iSec == 0 && m_iMSec == 0) {

        ++iJD;
        julian2Calendar(iJD, m_iYear, m_iMonth, m_iDay);

        m_dJulianDate = iJD;
        m_iHour = 0;
    }

    m_ullCMTime = time2CMTime(m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec, m_iMSec);
    m_iDayOfWeek = calcDayOfWeek();
}


void CDateTime::removeTime() {

    m_dJulianDate = calendar2Julian(m_iYear, m_iMonth, m_iDay);

    m_iHour = 0;
    m_iMin = 0;
    m_iSec = 0;
    m_iMSec = 0;

    m_ullCMTime = time2CMTime(m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec, m_iMSec);
}


void CDateTime::getYMD(unsigned int iYear,
        unsigned int iMonth,
        unsigned int iDay) const {

    julian2Calendar(static_cast<int>(floor(m_dJulianDate)), iYear, iMonth, iDay);
}


std::string CDateTime::toString() const {

    char szBuffer[255];
    sprintf(szBuffer, "%4d%02d%02d", m_iYear, m_iMonth, m_iDay);

    return std::string(szBuffer);
}


std::string CDateTime::toStringMDYYYY() const {

    char szBuffer[255];
    sprintf(szBuffer, "%02d/%02d/%4d", m_iMonth, m_iDay, m_iYear);

    return std::string(szBuffer);
}


std::string CDateTime::toStringMDY() const {

    char szBuffer[255];
    sprintf(szBuffer, "%02d%02d%4d", m_iMonth, m_iDay, m_iYear);

    return std::string(szBuffer);
}


std::string CDateTime::toStringYMD() const {

    char szBuffer[255];
    sprintf(szBuffer, "%4d%02d%02d", m_iYear, m_iMonth, m_iDay);

    return std::string(szBuffer);
}


std::string CDateTime::toStringFIX() const {

    char szBuffer[255];
    sprintf(szBuffer, "%4d%02d%02d-%02d:%02d:%02d", m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec);

    return std::string(szBuffer);
}


std::string CDateTime::toStringFull() const {

    char szBuffer[255];

    // format changed on 8/15/18 to enable compatibility with mariadb (this function is mostly called from CFXDB.cpp)
    //sprintf(szBuffer, "%4d%02d%02d %02d:%02d:%02d.%03d", m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec, m_iMSec);
    sprintf(szBuffer, "%4d-%02d-%02d %02d:%02d:%02d.%03d", m_iYear, m_iMonth, m_iDay, m_iHour, m_iMin, m_iSec, m_iMSec);

    return std::string(szBuffer);
}

size_t CDateTime::toStringLogger(char *buf, size_t bufSize) const {
    return snprintf(buf, bufSize, "%02d%02d%02d.%03d", m_iHour, m_iMin, m_iSec, m_iMSec);
}

std::string CDateTime::toStringSQL() const {

    char szBuffer[255];
    sprintf(szBuffer, "%02d/%02d/%04d %02d:%02d:%02d.%03d", m_iMonth, m_iDay, m_iYear, m_iHour, m_iMin, m_iSec,
            m_iMSec);

    return std::string(szBuffer);
}


std::string CDateTime::toStringDate() const {

    char szBuffer[255];
    sprintf(szBuffer, "%04d-%02d-%02d", m_iYear, m_iMonth, m_iDay);

    return std::string(szBuffer);
}


std::string CDateTime::toStringTime() const {

    char szBuffer[255];
    sprintf(szBuffer, "%02d:%02d:%02d.%03d", m_iHour, m_iMin, m_iSec, m_iMSec);

    return std::string(szBuffer);
}


unsigned long long CDateTime::nowEx() {

    boost::posix_time::ptime dtTime = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration duration(dtTime.time_of_day());

    return static_cast<unsigned long long>(duration.total_milliseconds());
}


CDateTime CDateTime::now() {

    static double dMillisPerDay = 24.0 * 3600.0 * 1000.0;

    boost::posix_time::ptime dtTime = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration duration(dtTime.time_of_day());
    double dDayFrac = static_cast<double>(duration.total_milliseconds()) / dMillisPerDay;

    unsigned int iHour = 0;
    unsigned int iMin = 0;
    unsigned int iSec = 0;
    unsigned int iMSec = 0;

    dayFrac2Time(dDayFrac, iHour, iMin, iSec, iMSec);

    time_t t = time(NULL);
    struct tm *pt = localtime(&t);

    unsigned int iYY = pt->tm_year + 1900;
    unsigned int iMM = pt->tm_mon + 1;
    unsigned int iDD = pt->tm_mday;

    return CDateTime(iYY, iMM, iDD, iHour, iMin, iSec, iMSec);
}


CDateTime CDateTime::nowUTC() {

    boost::gregorian::date dt = boost::gregorian::day_clock::universal_day();
    int iHr = static_cast<unsigned int>(boost::posix_time::second_clock::universal_time().time_of_day().hours());
    int iMn = static_cast<unsigned int>(boost::posix_time::second_clock::universal_time().time_of_day().minutes());
    int iSe = static_cast<unsigned int>(boost::posix_time::second_clock::universal_time().time_of_day().seconds());
    int iMs = static_cast<unsigned int>(
            boost::posix_time::second_clock::universal_time().time_of_day().total_milliseconds() % 1000);

    return CDateTime(dt.year(), dt.month(), dt.day(), iHr, iMn, iSe, iMs);
}


CDateTime CDateTime::startOfDay() {

    CDateTime dtNow = now();

    unsigned int iYY = 0;
    unsigned int iMM = 0;
    unsigned int iDD = 0;
    unsigned int iHour = 0;
    unsigned int iMin = 0;
    unsigned int iSec = 0;
    unsigned int iMSec = 0;

    dtNow.decompose(iYY, iMM, iDD, iHour, iMin, iSec, iMSec);

    return CDateTime(iYY, iMM, iDD, 0, 0, 0, 0);
}


bool CDateTime::isLeapYear(unsigned int iYYYY) {

    return (iYYYY % 4 == 0 && iYYYY % 100 != 0) || iYYYY % 400 == 0;
}


void CDateTime::julian2Calendar(unsigned int iJD,
        unsigned int &iYear,
        unsigned int &iMonth,
        unsigned int &iDay) {

    iJD += ExcelBasisAdjustment;

    unsigned int iL = iJD + 68569;
    unsigned int iN = 4 * iL / 146097;
    iL = iL - (146097 * iN + 3) / 4;
    unsigned int iI = 4000 * (iL + 1) / 1461001;
    iL = iL - 1461 * iI / 4 + 31;
    unsigned int iJ = 80 * iL / 2447;
    unsigned int iK = iL - 2447 * iJ / 80;
    iL = iJ / 11;
    iJ = iJ + 2 - 12 * iL;
    iI = 100 * (iN - 49) + iI + iL;

    iYear = iI;
    iMonth = iJ;
    iDay = iK;
}


// the conversion formula is sourced from http://aa.usno.navy.mil/faq/docs/JD_Formula.php
// the constant ExcelBasisAdjustment must be used for the formula to be accurate

unsigned int CDateTime::calendar2Julian(unsigned int iYear,
        unsigned int iMonth,
        unsigned int iDay) {

    int iJD = static_cast<int>(iDay) - 32075 +
              1461 * (static_cast<int>(iYear) + 4800 + (static_cast<int>(iMonth) - 14) / 12) / 4 +
              367 * (static_cast<int>(iMonth) - 2 - (static_cast<int>(iMonth) - 14) / 12 * 12) / 12 -
              3 * ((static_cast<int>(iYear) + 4900 + (static_cast<int>(iMonth) - 14) / 12) / 100) / 4;

    return iJD - ExcelBasisAdjustment;
}


void CDateTime::dayFrac2Time(double dDayFrac,
        unsigned int &iHour,
        unsigned int &iMin,
        unsigned int &iSec,
        unsigned int &iMSec) {

    // round to nearest millisecond (4th dp)
    double dNumSecs = dDayFrac * 24.0 * 3600.0;
    dNumSecs = floor(dNumSecs * 10000 + 0.5) / 10000;

    double dHour = floor(dNumSecs / 3600.0);
    dNumSecs -= dHour * 3600.0;

    double dMin = floor(dNumSecs / 60.0);
    dNumSecs -= dMin * 60.0;

    double dSec = floor(dNumSecs);

    double dMSec = dNumSecs - dSec;
    dMSec = 1000 * dMSec;
    dMSec = floor(dMSec * 1000 + 0.5) / 1000;

    iHour = static_cast<unsigned int>(dHour);
    iMin = static_cast<unsigned int>(dMin);
    iSec = static_cast<unsigned int>(dSec);
    iMSec = static_cast<unsigned int>(dMSec);
}


double CDateTime::time2DayFrac(unsigned int iHour,
        unsigned int iMin,
        unsigned int iSec,
        unsigned int iMSec) {

    static double dSecsPerDay = 24.0 * 3600;
    double dNumSecs = static_cast<double>(iHour) * 3600 + static_cast<double>(iMin) * 60 + static_cast<double>(iSec) +
                      static_cast<double>(iMSec) / 1000.0;
    return dNumSecs / dSecsPerDay;
}


unsigned long long CDateTime::time2CMTime(unsigned int iYear,
        unsigned int iMon,
        unsigned int iDay,
        unsigned int iHH,
        unsigned int iMM,
        unsigned int iSS,
        unsigned int iMS) {

    int iYYYYMMDD = iDay + iMon * 100L + iYear * 10000L;
    return iMS + iSS * 1000LL + iMM * 100000LL + iHH * 10000000LL + iYYYYMMDD * 1000000000LL;
}


unsigned long long CDateTime::julian2CMTime(double dJulian) {

    int iJD = static_cast<int>(floor(dJulian));
    double dDayFrac = dJulian - iJD;

    unsigned int iYY = 0, iMM = 0, iDD = 0, iHour = 0, iMin = 0, iSec = 0, iMSec = 0;
    julian2Calendar(iJD, iYY, iMM, iDD);
    dayFrac2Time(dDayFrac, iHour, iMin, iSec, iMSec);

    return time2CMTime(iYY, iMM, iDD, iHour, iMin, iSec, iMSec);
}


bool CDateTime::equalDay(const CDateTime &objRhs) const {

    return fabs(floor(m_dJulianDate) - floor(objRhs.getValue())) < 0.01;
}


void CDateTime::decompose(unsigned int &iYY,
        unsigned int &iMM,
        unsigned int &iDD,
        unsigned int &iHour,
        unsigned int &iMin,
        unsigned int &iSec,
        unsigned int &iMSec) const {

    //julian2Calendar(floor(m_dJulianDate), iYY, iMM, iDD);

    iYY = m_iYear;
    iMM = m_iMonth;
    iDD = m_iDay;
    iHour = m_iHour;
    iMin = m_iMin;
    iSec = m_iSec;
    iMSec = m_iMSec;
}


CDateTime &CDateTime::operator=(const Mts::Core::CDateTime &rhs) {

    m_dJulianDate = rhs.getValue();
    m_iDay = rhs.getDay();
    m_iMonth = rhs.getMonth();
    m_iYear = rhs.getYear();
    m_iHour = rhs.getHour();
    m_iMin = rhs.getMin();
    m_iSec = rhs.getSec();
    m_iMSec = rhs.getMSec();
    m_ullCMTime = rhs.getCMTime();
    m_iDayOfWeek = rhs.getDayOfWeek();

    return *this;
}


void CDateTime::addMinutes(unsigned int iMins) {

    double dDayFrac = static_cast<double>(iMins) / (24.0 * 60.0);

    Mts::Core::CDateTime dNewDateTime = CDateTime(m_dJulianDate += dDayFrac);

    *this = dNewDateTime;
}


// assumes "this" datetime is UTC
CDateTime CDateTime::toLocal() const {

    using namespace boost::posix_time;
    using namespace boost::gregorian;

    //eastern timezone is utc-5
    typedef boost::date_time::local_adjustor<ptime, -5, us_dst> us_eastern;

    ptime dtUTC(date(m_iYear, m_iMonth, m_iDay),
                hours(m_iHour) + minutes(m_iMin) + seconds(m_iSec) + milliseconds(m_iMSec));
    ptime dtLocal = us_eastern::utc_to_local(dtUTC);//back should be the same

    return CDateTime(dtLocal.date().year(), dtLocal.date().month(), dtLocal.date().day(), dtLocal.time_of_day().hours(),
                     dtLocal.time_of_day().minutes(), dtLocal.time_of_day().seconds(),
                     dtLocal.time_of_day().total_milliseconds() % 1000);
}


// assumes "this" datetime is local
CDateTime CDateTime::toUTC() const {

    using namespace boost::posix_time;
    using namespace boost::gregorian;

    //eastern timezone is utc-5
    typedef boost::date_time::local_adjustor<ptime, -5, us_dst> us_eastern;

    ptime dtLocal(date(m_iYear, m_iMonth, m_iDay),
                  hours(m_iHour) + minutes(m_iMin) + seconds(m_iSec) + milliseconds(m_iMSec));
    ptime dtUTC = us_eastern::local_to_utc(dtLocal);

    return CDateTime(dtUTC.date().year(), dtUTC.date().month(), dtUTC.date().day(), dtUTC.time_of_day().hours(),
                     dtUTC.time_of_day().minutes(), dtUTC.time_of_day().seconds(),
                     dtUTC.time_of_day().total_milliseconds() % 1000);
}


CDateTime CDateTime::createDateMDY(const std::string &strDate) {

    const char *p = strDate.c_str();

    unsigned int iYear = 0;
    unsigned int iMonth = 0;
    unsigned int iDay = 0;

    if (p[1] == '/') {

        iMonth = Mts::Math::CMath::atoi_1(p);
        p += 2;
    } else {

        iMonth = Mts::Math::CMath::atoi_2(p);
        p += 3;
    }

    if (p[1] == '/') {

        iDay = Mts::Math::CMath::atoi_1(p);
        p += 2;
    } else {

        iDay = Mts::Math::CMath::atoi_2(p);
        p += 3;
    }

    iYear = Mts::Math::CMath::atoi_4(p);

    return CDateTime(iYear, iMonth, iDay, 0, 0, 0, 0);
}


CDateTime CDateTime::epochInNanos2DateTime(long long iEpochInNanos) {

    // convert from linux epoch time in nanos to internal timestamp
    long long iMilliFrEpoch = iEpochInNanos / 1000000;

    static const boost::posix_time::ptime dtEpoch(boost::gregorian::date(1970, 1, 1));

    boost::posix_time::ptime dtTimestampUTC = dtEpoch + boost::posix_time::milliseconds(iMilliFrEpoch);

    boost::posix_time::ptime dtTimestampLocal = boost::date_time::c_local_adjustor<boost::posix_time::ptime>::utc_to_local(
            dtTimestampUTC);

    int iYY = dtTimestampLocal.date().year();
    int iMM = dtTimestampLocal.date().month();
    int iDD = dtTimestampLocal.date().day();
    int iHH = dtTimestampLocal.time_of_day().hours();
    int iMN = dtTimestampLocal.time_of_day().minutes();
    int iSS = dtTimestampLocal.time_of_day().seconds();
    int iMS = (int) (dtTimestampLocal.time_of_day().fractional_seconds() / 1000LL);
    int iTimeInMin = iHH * 60 + iMN;

    return CDateTime(iYY, iMM, iDD, iHH, iMN, iSS, iMS);
}



