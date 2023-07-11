#include "CBidAsk.h"


using namespace Mts::OrderBook;


CBidAsk::CBidAsk() {


}


CBidAsk::CBidAsk(unsigned int iSymbolID,
        const Mts::Core::CDateTime &dtTimestamp,
        const CQuote &objBid,
        const CQuote &objAsk)
        : m_iSymbolID(iSymbolID),
          m_dtTimestamp(dtTimestamp),
          CEvent(BID_ASK),
          m_Bid(objBid),
          m_Ask(objAsk) {

}


const Mts::Core::CDateTime &CBidAsk::getTimestamp() const {

    return m_dtTimestamp;
}


unsigned int CBidAsk::getSymbolID() const {

    return m_iSymbolID;
}


const CQuote &CBidAsk::getBid() const {

    return m_Bid;
}


const CQuote &CBidAsk::getAsk() const {

    return m_Ask;
}


std::string CBidAsk::toString() const {
    return "Bid:[" + m_Bid.toString() + "] Ask:[" + m_Ask.toString() + "]";
};


double CBidAsk::getMidPx() const {

    return 0.5 * m_Bid.getPrice() + 0.5 * m_Ask.getPrice();
}


