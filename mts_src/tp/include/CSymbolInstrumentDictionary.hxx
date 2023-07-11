#include <exception>
#include "CMtsException.h"


using namespace Mts::Core;


inline const InstrumentCalendar &CSymbolInstrumentDictionary::getCalendarFromMtsSymbol(unsigned int iSymbolID) const {

    return getCalendarFromMtsSymbol(iSymbolID, Mts::Core::CDateTime::now());
}


inline const InstrumentCalendar &CSymbolInstrumentDictionary::getCalendarFromMtsSymbol(unsigned int iSymbolID,
        const Mts::Core::CDateTime &dtQuery) const {

    return getCalendar(iSymbolID, dtQuery);
}


inline bool CSymbolInstrumentDictionary::hasCalendar(unsigned int iSymbolID,
        const Mts::Core::CDateTime &dtQuery) const {

    auto iter1 = m_MtsSymbolID2CalendarMap.find(iSymbolID);

    if (iter1 == m_MtsSymbolID2CalendarMap.end())
        return false;

    auto iter2 = iter1->second.find(dtQuery.getJulianDay());

    if (iter2 == iter1->second.end())
        return false;

    return true;
}


inline const InstrumentCalendar &CSymbolInstrumentDictionary::getCalendar(unsigned int iSymbolID,
        const Mts::Core::CDateTime &dtQuery) const {

    auto iter1 = m_MtsSymbolID2CalendarMap.find(iSymbolID);
    auto iter2 = iter1->second.find(dtQuery.getJulianDay());
    return *(iter2->second);
}

