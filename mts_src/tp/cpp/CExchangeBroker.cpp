#include "CExchangeBroker.h"
#include "CApplicationLog.h"


using namespace Mts::Exchange;


CExchangeBroker &CExchangeBroker::getInstance() {

    static CExchangeBroker objTheInstance;
    return objTheInstance;
}


void CExchangeBroker::addExchange(int iProviderID,
        boost::shared_ptr<CExchange> ptrExchange) {

    m_Exchanges[iProviderID] = ptrExchange;
}


void CExchangeBroker::connectToAllExchanges() {

    ExchangeMap::iterator iter = m_Exchanges.begin();

    for (; iter != m_Exchanges.end(); ++iter)
        iter->second->connect();
}


void CExchangeBroker::disconnectFromAllExchanges() {

    ExchangeMap::iterator iter = m_Exchanges.begin();

    for (; iter != m_Exchanges.end(); ++iter)
        iter->second->disconnect();
}


void CExchangeBroker::connectToExchange(int iProviderID) {

    boost::optional<std::reference_wrapper<CExchange> > objExchRef = getExchange(iProviderID);

    if (objExchRef != boost::none)
        objExchRef.value().get().connect();

    //getExchange(iProviderID).connect();
}


void CExchangeBroker::disconnectFromExchange(int iProviderID) {

    boost::optional<std::reference_wrapper<CExchange> > objExchRef = getExchange(iProviderID);

    if (objExchRef != boost::none)
        objExchRef.value().get().disconnect();

    //getExchange(iProviderID).disconnect();
}


void CExchangeBroker::initiateAllExchanges() {

    ExchangeMap::iterator iter = m_Exchanges.begin();

    for (; iter != m_Exchanges.end(); ++iter) {

        char szBuffer[255];
        sprintf(szBuffer, "Initializing exchange for provider %u", iter->second->getProviderID());
        AppLog(szBuffer);

        iter->second->run();
    }
}


void CExchangeBroker::feedTestExchanges(const Mts::OrderBook::CBidAsk &objBidAsk) {

    ExchangeMap::iterator iter = m_Exchanges.begin();

    for (; iter != m_Exchanges.end(); ++iter)
        iter->second->onEvent(objBidAsk);
}


/*
CExchange & CExchangeBroker::getExchange(int iProviderID) {

	return *m_Exchanges.at(iProviderID);
}
*/


boost::optional<std::reference_wrapper<CExchange> > CExchangeBroker::getExchange(int iProviderID) {

    return boost::optional<std::reference_wrapper<CExchange> >(*m_Exchanges.at(iProviderID));
}


void CExchangeBroker::dump() {

    ExchangeMap::iterator iter = m_Exchanges.begin();

    for (; iter != m_Exchanges.end(); ++iter) {

        char szBuffer[255];
        sprintf(szBuffer, "Dumping exchange for provider %u %u", iter->first, iter->second->getProviderID());
        AppLog(szBuffer);
    }
}


