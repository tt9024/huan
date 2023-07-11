#include "CTrade.h"


using namespace Mts::OrderBook;


CTrade::CTrade()
: CEvent(Mts::Event::CEvent::TRADE),
	m_iSymbolID(0),
	m_iProviderID(0),
  m_dtMtsTimestamp(0),
  m_dtExcTimestamp(0),
  m_dPrice(0),
  m_iSize(0) {

	strcpy(m_szExch, "");
	strcpy(m_szExDest, "");
}


CTrade::CTrade(unsigned int									iSymbolID,
							 unsigned int									iProviderID,
							 const Mts::Core::CDateTime & dtMtsTimestamp, 
							 const Mts::Core::CDateTime & dtExcTimestamp, 
							 double												dPrice, 
							 unsigned int									iSize,
							 const std::string &					strExch,
							 const std::string &					strExDest)
: CEvent(Mts::Event::CEvent::TRADE),
	m_iSymbolID(iSymbolID),
	m_iProviderID(iProviderID),
  m_dtMtsTimestamp(dtMtsTimestamp),
  m_dtExcTimestamp(dtExcTimestamp),
  m_dPrice(dPrice),
  m_iSize(iSize) {

	strcpy(m_szExch, strExch.c_str());
	strcpy(m_szExDest, strExDest.c_str());
}


std::string CTrade::toString() const {

	const Mts::Core::CSymbol &		objSymbol				= Mts::Core::CSymbol::getSymbol(m_iSymbolID);
	
	char szBuffer[255];
	sprintf(szBuffer, "%llu,%s,%s,%s,%.7f,%d", m_dtMtsTimestamp.getCMTime(), m_szExch, m_szExDest, objSymbol.getSymbol().c_str(), m_dPrice, m_iSize);
	
	return szBuffer;
}

