#ifndef CSYMBOLINSTRUMENTDICTIONARY_HEADER

#define CSYMBOLINSTRUMENTDICTIONARY_HEADER

#include <boost/unordered_map.hpp>
#include "CConfig.h"
#include "CDateTime.h"

namespace Mts {
    namespace Core {
        struct InstrumentCalendar
        {
            std::string MtsSymbol;
            std::string ContractTicker;
            std::string ContractExchSymbol;
            double ContractPointValue;
            std::string ContractTTSecID;
            unsigned int TradingStartMin;
            unsigned int TradingEndMin;
            unsigned int SettStartSec;
            unsigned int SettEndSec;
            Mts::Core::CDateTime ExpirationDate;
            Mts::Core::CDateTime TradingStartTime;
            Mts::Core::CDateTime TradingEndTime;
            Mts::Core::CDateTime SettStartTime;
            Mts::Core::CDateTime SettEndTime;
        };

        class CSymbolInstrumentDictionary
        {
        public:
            CSymbolInstrumentDictionary();

            bool load(const std::string &strXML);

            const InstrumentCalendar &getCalendarFromMtsSymbol(unsigned int iSymbolID) const;

            const InstrumentCalendar &getCalendarFromMtsSymbol(unsigned int iSymbolID,
                    const Mts::Core::CDateTime &dtQuery) const;

            bool hasCalendar(unsigned int iSymbolID,
                    const Mts::Core::CDateTime &dtQuery) const;

            const InstrumentCalendar &getCalendar(unsigned int iSymbolID,
                    const Mts::Core::CDateTime &dtQuery) const;

        private:
            typedef boost::unordered_map<unsigned int, boost::shared_ptr<InstrumentCalendar>> JulianDate2CalendarMap;
            typedef boost::unordered_map<unsigned int, JulianDate2CalendarMap> Symbol2CalendarMap;

            // Mts symbol ID -> instrument calendar
            Symbol2CalendarMap m_MtsSymbolID2CalendarMap;
        };
    }
}

#include "CSymbolInstrumentDictionary.hxx"

#endif


