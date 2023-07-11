#include "COHLCBarHist.h"


using namespace Mts::Indicator;


COHLCBarHist::COHLCBarHist(unsigned int iRollMinute,
        unsigned int iNumBars)
        : m_iRollMinute(iRollMinute),
          m_iNumBars(iNumBars),
          m_dtLastRoll(0),
          m_BarHist(iNumBars, 0) {

}


void COHLCBarHist::update(const Mts::OrderBook::CBidAsk &objBidAsk) {

    if (objBidAsk.getTimestamp().getMin() % m_iRollMinute == 0 &&
        objBidAsk.getTimestamp().getMin() != m_dtLastRoll.getMin()) {

        m_dtLastRoll = objBidAsk.getTimestamp();

        roll();
    }

    m_CurrBar.update(objBidAsk);
}


void COHLCBarHist::update(const Mts::Core::CDateTime &dtTimestamp) {

    if (dtTimestamp.getMin() % m_iRollMinute != 0 || dtTimestamp.getMin() == m_dtLastRoll.getMin())
        return;

    m_dtLastRoll = dtTimestamp;

    roll();
}


void COHLCBarHist::roll() {

    m_BarHist.append(m_CurrBar.getTimestampClose(), m_CurrBar);
    m_CurrBar.roll();
}


std::tuple<Mts::Core::CDateTime, double, double, double, double> COHLCBarHist::getOHLCBar(unsigned int iIndex) const {

    Mts::Indicator::COHLCBar objBar = m_BarHist.getItemOffsetFromHead(iIndex);

    return std::make_tuple(objBar.getTimestampClose(), objBar.getOpen(), objBar.getHigh(), objBar.getLow(),
                           objBar.getClose());
}


Mts::Core::CDateTime COHLCBarHist::getBarTimestampClose(unsigned int iIndex) const {

    return m_BarHist.getItemOffsetFromHead(iIndex).getTimestampClose();
}


double COHLCBarHist::getBarOpen(unsigned int iIndex) const {

    return m_BarHist.getItemOffsetFromHead(iIndex).getOpen();
}


double COHLCBarHist::getBarHigh(unsigned int iIndex) const {

    return m_BarHist.getItemOffsetFromHead(iIndex).getHigh();
}


double COHLCBarHist::getBarLow(unsigned int iIndex) const {

    return m_BarHist.getItemOffsetFromHead(iIndex).getLow();
}


double COHLCBarHist::getBarClose(unsigned int iIndex) const {

    return m_BarHist.getItemOffsetFromHead(iIndex).getClose();
}


bool COHLCBarHist::isFull() const {

    return m_BarHist.isFull();
}


unsigned int COHLCBarHist::getNumBars() const {

    return m_BarHist.getSize();
}

