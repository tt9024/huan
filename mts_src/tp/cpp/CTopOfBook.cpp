#include "CTopOfBook.h"
#include "CDoubleEqualTo.h"


using namespace Mts::OrderBook;


CTopOfBook::CTopOfBook()
: m_dBestBidPx(0),
	m_iBestBidQty(0),
	m_dBestAskPx(0),
	m_iBestAskQty(0) {

}


void CTopOfBook::update(const CQuote & objBestBid,
												const CQuote & objBestAsk) {

	Mts::Math::CDoubleEqualTo objDoubleEqualTo;

	m_bChanged = (objBestBid.getSize() != m_iBestBidQty || 
							  objBestAsk.getSize() != m_iBestAskQty || 
							  objDoubleEqualTo(objBestBid.getPrice(), m_dBestBidPx) == false || 
							  objDoubleEqualTo(objBestAsk.getPrice(), m_dBestAskPx) == false);

	if (m_bChanged == true) {

		m_dBestBidPx	= objBestBid.getPrice();
		m_iBestBidQty	= objBestBid.getSize();
		m_dBestAskPx	= objBestAsk.getPrice();
		m_iBestAskQty	= objBestAsk.getSize();
	}
}


void CTopOfBook::update(const CBidAsk & objBidAsk) {

	update(objBidAsk.getBid(), objBidAsk.getAsk());
}


bool CTopOfBook::isChanged() const {

	return m_bChanged;
}


double CTopOfBook::getMidPx() const {

	return 0.5 * m_dBestBidPx + 0.5 * m_dBestAskPx;
}




