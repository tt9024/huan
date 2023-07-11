#include <exception>
#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include "CXMLParser.h"
#include "CTokens.h"
#include "CSymbolInstrumentDictionary.h"
#include "CSymbol.h"
#include "CMtsException.h"
#include "CStringTokenizer.h"
#include "CApplicationLog.h"
#include <cctype>

using namespace Mts::Core;


CSymbolInstrumentDictionary::CSymbolInstrumentDictionary() {

}


bool CSymbolInstrumentDictionary::load(const std::string & strXML) {

    FILE * ptrInfile = NULL;
	try {

		char szBuffer[512];

		ptrInfile = fopen(strXML.c_str(), "r");
        if (!ptrInfile) {
            throw Mts::Exception::CMtsException( std::string("Calendar file not found : ") + strXML);
        }

		Mts::Core::CStringTokenizer objTokenizer;

		bool bHeader = true;

		while (fgets(szBuffer, 512, ptrInfile) != NULL) {

			if (bHeader == true) {

				bHeader = false;
				continue;
			}

			std::vector<std::string> tokens = objTokenizer.splitString(szBuffer, ",");

			if (tokens.size() != 12)
				continue;

			for (size_t i = 0; i != tokens.size(); ++i) {

				boost::algorithm::trim(tokens[i]);
			}

			// extract select fields of interest
			std::string strMtsSymbol = tokens[0];
            strMtsSymbol.erase( 
                    std::remove_if( 
                        strMtsSymbol.begin(), 
                        strMtsSymbol.end(),
                        [](char c){
                            return std::isspace(static_cast<unsigned char>(c));
                        }
                    ), 
                    strMtsSymbol.end()
                );

            if (Mts::Core::CSymbol::isSymbol(strMtsSymbol) == false)
                throw Mts::Exception::CMtsException(
                    std::string("unknown symbol (") + strMtsSymbol + std::string(") in calendar csv ") + strXML);

			const Mts::Core::CSymbol & objSymbol	= Mts::Core::CSymbol::getSymbol(strMtsSymbol);
			const std::string &		   strCalDate	= tokens[1];
			const std::string &		   strDataType	= tokens[2];

			if (strDataType == "X")
				continue;
            
            // contract
			const std::string & strInstrument = tokens[3];
			const std::string & strContractExchSymbol = tokens[4];
			const std::string & strContractExpiry = tokens[5]; // expected format yyyy-mm-dd
			double dContractPointValue = atof(tokens[6].c_str());
			const std::string & strContractTTSecID = tokens[7];

			// expected format yyyy-mm-dd hh:mm
			const std::string & strTradingStart = tokens[8];
			const std::string & strTradingEnd = tokens[9];

			// expected format yyyy-mm-dd hh:mm:ss.000
			const std::string & strSettStart = tokens[10];
			const std::string & strSettEnd = tokens[11];

			Mts::Core::CDateTime dtCalDate(strCalDate, "00:00:00.000");
			Mts::Core::CDateTime dtExpirationDate(strContractExpiry, "00:00:00.000");
			Mts::Core::CDateTime dtTradingStart(strTradingStart, 0);
			Mts::Core::CDateTime dtTradingEnd(strTradingEnd, 0);
			Mts::Core::CDateTime dtSettStart(strSettStart, 1);
			Mts::Core::CDateTime dtSettEnd(strSettEnd, 1);

			boost::shared_ptr<InstrumentCalendar> ptrCalendar(new InstrumentCalendar);
			ptrCalendar->MtsSymbol = strMtsSymbol;
			ptrCalendar->ContractTicker = strInstrument;
			ptrCalendar->ContractExchSymbol = strContractExchSymbol;
			ptrCalendar->ContractPointValue = dContractPointValue;
			ptrCalendar->ContractTTSecID = strContractTTSecID;
			ptrCalendar->TradingStartMin = dtTradingStart.getHour() * 60 + dtTradingStart.getMin();
			ptrCalendar->TradingEndMin = dtTradingEnd.getHour() * 60 + dtTradingEnd.getMin();
			ptrCalendar->SettStartSec = dtSettStart.getHour() * 3600 + dtSettStart.getMin() * 60 + dtSettStart.getSec();
			ptrCalendar->SettEndSec = dtSettEnd.getHour() * 3600 + dtSettEnd.getMin() * 60 + dtSettEnd.getSec();
			ptrCalendar->ExpirationDate = dtExpirationDate;
			ptrCalendar->TradingStartTime = dtTradingStart;
			ptrCalendar->TradingEndTime = dtTradingEnd;
			ptrCalendar->SettStartTime = dtSettStart;
			ptrCalendar->SettEndTime = dtSettEnd;

			m_MtsSymbolID2CalendarMap[objSymbol.getSymbolID()][dtCalDate.getJulianDay()] = ptrCalendar;
		}

	}
	catch (const std::exception & e) {
        if (ptrInfile) {
            fclose(ptrInfile);
        }
        throw std::runtime_error(e.what());
	}

    fclose(ptrInfile);
    return true;
}

