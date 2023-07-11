#include "CQuote.h"


using namespace Mts::OrderBook;


CQuote::CQuote()
        : CEvent(Mts::Event::CEvent::QUOTE),
          m_iSymbolID(0),
          m_iProviderID(0),
          m_dtMtsTimestamp(0),
          m_dtExcTimestamp(0),
          m_iSide(BID),
          m_dPrice(0),
          m_iSize(0),
          m_iValueDateJulian(0) {

    strcpy(m_szExch, "");
    strcpy(m_szExDest, "");
}


CQuote::CQuote(unsigned int iSymbolID,
        unsigned int iProviderID,
        const Mts::Core::CDateTime &dtMtsTimestamp,
        const Mts::Core::CDateTime &dtExcTimestamp,
        Side iSide,
        double dPrice,
        unsigned int iSize,
        const std::string &strExch,
        const std::string &strExDest)
        : CEvent(Mts::Event::CEvent::QUOTE),
          m_iSymbolID(iSymbolID),
          m_iProviderID(iProviderID),
          m_dtMtsTimestamp(dtMtsTimestamp),
          m_dtExcTimestamp(dtExcTimestamp),
          m_iSide(iSide),
          m_dPrice(dPrice),
          m_iSize(iSize),
          m_iValueDateJulian(0) {

    strcpy(m_szExch, strExch.c_str());
    strcpy(m_szExDest, strExDest.c_str());
}


std::string CQuote::toString() const {

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(m_iSymbolID);
    std::string strSide = m_iSide == BID ? "Bid" : "Ask";

    char szBuffer[255];
    sprintf(szBuffer, "%llu,%s,%s,%s,%.7f,%d,%s", m_dtMtsTimestamp.getCMTime(), m_szExch, m_szExDest,
            objSymbol.getSymbol().c_str(), m_dPrice, m_iSize, strSide.c_str());

    return szBuffer;
}


unsigned int CQuote::getValueDateJulian() const {

    return m_iValueDateJulian;
}


void CQuote::setValueDateJulian(unsigned int iValueDateJulian) {

    m_iValueDateJulian = iValueDateJulian;
}

