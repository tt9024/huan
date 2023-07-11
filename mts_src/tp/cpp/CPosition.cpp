#include "CPosition.h"
#include "CMath.h"
#include "CApplicationLog.h"
#include <string>
#include <sstream>

using namespace Mts::Accounting;

CPosition::CPosition()
        : m_uiSymbolID(0),
          m_ulLongQty(0),
          m_dLongWAP(0),
          m_ulShortQty(0),
          m_dShortWAP(0),
          m_dPointValue(1) {
    m_dLastMktPrice = 0.0;
    m_dLastPnL = 0.0;
}


CPosition::CPosition(const Mts::Core::CSymbol &objSymbol)
        : m_uiSymbolID(objSymbol.getSymbolID()),
          m_ulLongQty(0),
          m_dLongWAP(0),
          m_ulShortQty(0),
          m_dShortWAP(0),
          m_dPointValue(objSymbol.getPointValue()) {
    m_dLastMktPrice = 0.0;
    m_dLastPnL = 0.0;
}

CPosition::CPosition(
        const Mts::Core::CSymbol &objSymbol,
        unsigned long ulLongQty,
        double dLongWAP,
        unsigned long ulShortQty,
        double dShortWAP
)
        : m_uiSymbolID(objSymbol.getSymbolID()),
          m_ulLongQty(ulLongQty),
          m_dLongWAP(dLongWAP),
          m_ulShortQty(ulShortQty),
          m_dShortWAP(dShortWAP),
          m_dPointValue(objSymbol.getPointValue()) {
    m_dLastMktPrice = 0.0;
    m_dLastPnL = 0.0;
}


CPosition::CPosition(
        const Mts::Core::CSymbol &objSymbol,
        long iPosition,
        double dWAP
)
        : m_uiSymbolID(objSymbol.getSymbolID()),
          m_dPointValue(objSymbol.getPointValue()) {
    if (iPosition > 0) {
        m_ulLongQty = static_cast<unsigned long>(iPosition);
        m_dLongWAP = dWAP;
        m_ulShortQty = 0;
        m_dShortWAP = 0;
    } else {
        m_ulLongQty = 0;
        m_dLongWAP = 0;
        m_ulShortQty = static_cast<unsigned long>(abs(iPosition));
        m_dShortWAP = dWAP;
    }

    m_dLastMktPrice = 0.0;
    m_dLastPnL = 0.0;
}

long CPosition::getPosition() const {
    return m_ulLongQty - m_ulShortQty;
}

// TODO - do we need this function?
long CPosition::getGrossPosition() const {
    return m_ulLongQty + m_ulShortQty;
}

double CPosition::getWAP() const {
    if (m_ulLongQty == m_ulShortQty)
        return 0.0;
    if (m_ulLongQty > m_ulShortQty)
        return m_dLongWAP;
    else
        return m_dShortWAP;
}

unsigned int CPosition::getSymbolID() const {
    return m_uiSymbolID;
}

double CPosition::calcPnL(double dMktPrice) {
    double dRealizedPnL = Mts::Math::CMath::min(m_ulLongQty, m_ulShortQty) * (m_dShortWAP - m_dLongWAP);
    long iNetPosition = m_ulLongQty - m_ulShortQty;
    double dUnRealizedPnL = iNetPosition * (iNetPosition > 0 ? dMktPrice - m_dLongWAP : dMktPrice - m_dShortWAP);

    m_dLastMktPrice = dMktPrice;
    m_dLastPnL = (dRealizedPnL + dUnRealizedPnL) * m_dPointValue;

    return m_dLastPnL;
}


double CPosition::getPnL() const {
    return m_dLastPnL;
}

void CPosition::updatePosition(
        const Mts::Order::COrder::BuySell iDirection,
        unsigned int iQuantity,
        double dPrice) {
    if (iDirection == Mts::Order::COrder::BUY) {
        m_dLongWAP = ((dPrice * iQuantity) + (m_ulLongQty * m_dLongWAP)) / static_cast<double>(m_ulLongQty + iQuantity);
        m_ulLongQty += iQuantity;
    } else {
        m_dShortWAP =
                ((dPrice * iQuantity) + (m_ulShortQty * m_dShortWAP)) / static_cast<double>(m_ulShortQty + iQuantity);
        m_ulShortQty += iQuantity;
    }
}

void CPosition::updatePosition(const Mts::Order::COrderFill &objFill) {
    updatePosition(objFill.getBuySell(), objFill.getFillQuantity(), objFill.getFillPrice());
}

// Format of a position string
// Symbol, LongQty, LongWap, ShortQty, ShortWap, LastPx, LastPnl

std::string CPosition::toString() const {
    char szBuffer[255];
    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(m_uiSymbolID);
    sprintf(szBuffer, "%s,%d,%.7lf,%d,%.7lf,%.7lf,%.0f",
            objSymbol.getSymbol().c_str(), m_ulLongQty, m_dLongWAP, m_ulShortQty, m_dShortWAP, m_dLastMktPrice,
            m_dLastPnL);
    return szBuffer;
}

bool CPosition::fromString(const std::string &line) {
    try {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> strFields;
        while (std::getline(iss, token, ',')) {
            strFields.push_back(token);
        }
        if (strFields.size() < 5) {
            AppError("Failed to parse the position string %s", line.c_str());
            return false;
        }
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strFields[0]);
        m_uiSymbolID = objSymbol.getSymbolID();
        m_dPointValue = objSymbol.getPointValue();
        m_ulLongQty = atol(strFields[1].c_str());
        m_dLongWAP = atof(strFields[2].c_str());
        m_ulShortQty = atol(strFields[3].c_str());
        m_dShortWAP = atof(strFields[5].c_str());
        m_dLastMktPrice = 0.0;
        m_dLastPnL = 0.0;
    }
    catch (const std::exception &e) {
        AppError("Exception during parsing position string %s\n%s", line.c_str(), e.what());
        return false;
    }
}

std::tuple<long, double, long, double, double> CPosition::getSummary() const {

    return std::make_tuple(m_ulLongQty, m_dLongWAP, m_ulShortQty, m_dShortWAP, m_dLastPnL);
}



