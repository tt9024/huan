#include "CFeed.h"


using namespace Mts::Feed;

CFeed::CFeed()
        : m_ptrThread(),
          m_bFIXProtocol(false) {

}


CFeed::CFeed(bool bFIXProtocol)
        : m_ptrThread(),
          m_bFIXProtocol(bFIXProtocol) {

}


void CFeed::run() {

    m_ptrThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::ref(*this)));
}


void CFeed::onStop() {

}


void CFeed::publishTrade(const Mts::OrderBook::CTrade &objTrade) {

    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();

    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedTrade(objTrade);
}


void CFeed::publishQuoteSim(const Mts::OrderBook::CBidAsk &objQuoteBidAsk) {

    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();

    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedQuoteSim(objQuoteBidAsk);
}


void CFeed::publishQuote(const Mts::OrderBook::CBidAsk &objQuoteBidAsk) {
    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();
    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedQuoteBidAsk(objQuoteBidAsk);
}

void CFeed::publishHalfQuote(const Mts::OrderBook::CQuote &objQuote) {
    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();
    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedQuote(objQuote);
}

void CFeed::publishHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat) {

    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();

    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedHeartbeat(objHeartbeat);
}


void CFeed::publishLogon(unsigned int iProviderID) {

    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();

    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedLogon(iProviderID);
}


void CFeed::publishLogout(unsigned int iProviderID) {

    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();

    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedLogout(iProviderID);
}


void CFeed::publishNoMoreData() {

    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();

    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedNoMoreData();
}


void CFeed::publishKeyValue(const Mts::OrderBook::CKeyValue &objKeyValue) {

    FeedSubscriberArray::iterator iter = m_FeedSubscribers.begin();

    for (; iter != m_FeedSubscribers.end(); ++iter)
        (*iter)->onFeedKeyValue(objKeyValue);
}


void CFeed::addSubscriber(IFeedSubscriber *ptrSubscriber) {

    m_FeedSubscribers.push_back(ptrSubscriber);
}


bool CFeed::isFIXProtocol() const {

    return m_bFIXProtocol;
}


void CFeed::addSubscriber(const std::string &strProvider) {

}


void CFeed::removeSubscriber(const std::string &strProvider) {

}


const Mts::Core::CDateTime &CFeed::getHeartbeat() const {

    return m_dtHeartbeat;
}


void CFeed::setHeartbeat(const Mts::Core::CDateTime &dtNow) {

    m_dtHeartbeat = dtNow;
}

