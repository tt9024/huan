#ifndef CDATETIME_HEADER

#define CDATETIME_HEADER

#include <string>
#include "math.h"

namespace Mts {
    namespace Core {
        class CDateTime
        {
        public:
            CDateTime();

            CDateTime(unsigned int iYear,
                    unsigned int iMonth,
                    unsigned int iDay,
                    unsigned int iHH,
                    unsigned int iMM,
                    unsigned int iSS,
                    unsigned int iMS);

            CDateTime(const std::string &strDate,
                    const std::string &strTime);

            CDateTime(const std::string &strCMTime);

            CDateTime(const std::string &strDateTime,
                    unsigned int iDummy);

            CDateTime(double dJulian);

            void removeTime();

            double getValue() const;

            double getTimeValue() const;

            unsigned long long getCMTime() const;

            std::string toString() const;

            std::string toStringFull() const;

            std::string toStringSQL() const;

            std::string toStringDate() const;

            std::string toStringTime() const;

            std::string toStringMDYYYY() const;

            std::string toStringMDY() const;

            std::string toStringYMD() const;

            std::string toStringFIX() const;

            size_t toStringLogger(char *buf, size_t bufSize) const;

            unsigned int getDay() const;

            unsigned int getMonth() const;

            unsigned int getYear() const;

            unsigned int getHour() const;

            unsigned int getMin() const;

            unsigned int getSec() const;

            unsigned int getMSec() const;

            unsigned int getTimeInMin() const;

            unsigned int getTimeInSec() const;

            unsigned int getJulianDay() const;

            double getMatlabDateNum() const;

            CDateTime &operator=(const Mts::Core::CDateTime &rhs);

            void addMinutes(unsigned int iMins);

            void getYMD(unsigned int iYear,
                    unsigned int iMonth,
                    unsigned int iDay) const;

            bool equalDay(const CDateTime &objRhs) const;

            void decompose(unsigned int &iYY,
                    unsigned int &iMM,
                    unsigned int &iDD,
                    unsigned int &iHour,
                    unsigned int &iMin,
                    unsigned int &iSec,
                    unsigned int &iMSec) const;

            unsigned int getDayOfWeek() const;

            unsigned int calcDayOfWeek() const;

            CDateTime toUTC() const;

            CDateTime toLocal() const;

            // static utility methods
            static unsigned long long nowEx();

            static CDateTime now();

            static CDateTime nowUTC();

            static CDateTime startOfDay();

            static CDateTime createDateMDY(const std::string &strDate);

            static bool isLeapYear(unsigned int iYYYY);

            static void julian2Calendar(unsigned int iJD,
                    unsigned int &iYear,
                    unsigned int &iMonth,
                    unsigned int &iDay);

            static unsigned int calendar2Julian(unsigned int iYear,
                    unsigned int iMonth,
                    unsigned int iDay);

            static void dayFrac2Time(double dDayFrac,
                    unsigned int &iHour,
                    unsigned int &iMin,
                    unsigned int &iSec,
                    unsigned int &iMSec);

            static double time2DayFrac(unsigned int iHour,
                    unsigned int iMin,
                    unsigned int iSec,
                    unsigned int iMSec);

            static unsigned long long time2CMTime(unsigned int iYear,
                    unsigned int iMon,
                    unsigned int iDay,
                    unsigned int iHH,
                    unsigned int iMM,
                    unsigned int iSS,
                    unsigned int iMS);

            static unsigned long long julian2CMTime(double dJulian);

            static CDateTime epochInNanos2DateTime(long long iEpochInNanos);

        private:
            enum Offset
            {
                ExcelBasisAdjustment = 2415019, MATLABBasisAdjustment = 693960
            };

            double m_dJulianDate;
            unsigned long long m_ullCMTime;
            unsigned int m_iHour;
            unsigned int m_iMin;
            unsigned int m_iSec;
            unsigned int m_iMSec;

            unsigned int m_iYear;
            unsigned int m_iMonth;
            unsigned int m_iDay;

            unsigned int m_iDayOfWeek;
        };
    }
}

#include "CDateTime.hxx"

#endif

