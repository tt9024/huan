#include "COHLCBar.h"


using namespace Mts::Indicator;


COHLCBar::COHLCBar()
        : m_dtTimestampOpen(0),
          m_dtTimestampClose(0),
          m_dOpen(0.0),
          m_dHigh(0.0),
          m_dLow(0.0),
          m_dClose(0.0),
          m_bNewBar(true) {

}


const Mts::Core::CDateTime &COHLCBar::getTimestampOpen() const {

    return m_dtTimestampOpen;
}


const Mts::Core::CDateTime &COHLCBar::getTimestampClose() const {

    return m_dtTimestampClose;
}


double COHLCBar::getOpen() const {

    return m_dOpen;
}


double COHLCBar::getHigh() const {

    return m_dHigh;
}


double COHLCBar::getLow() const {

    return m_dLow;
}


double COHLCBar::getClose() const {

    return m_dClose;
}


void COHLCBar::update(const Mts::OrderBook::CBidAsk &objBidAsk) {

    if (m_bNewBar) {

        m_dOpen = objBidAsk.getMidPx();
        m_dHigh = m_dOpen;
        m_dLow = m_dOpen;
        m_dClose = m_dOpen;

        m_dtTimestampOpen = objBidAsk.getTimestamp();
        m_dtTimestampClose = m_dtTimestampOpen;

        m_bNewBar = false;
    } else {

        m_dClose = objBidAsk.getMidPx();
        m_dHigh = std::max(m_dHigh, m_dClose);
        m_dLow = std::min(m_dLow, m_dClose);

        m_dtTimestampClose = objBidAsk.getTimestamp();
    }
}


void COHLCBar::update(const Mts::Core::CDateTime &dtTimestamp) {

    m_dtTimestampClose = dtTimestamp;
}


void COHLCBar::roll() {

    m_bNewBar = true;
}


