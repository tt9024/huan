#include "CExchange.h"
#include "CApplicationLog.h"

using namespace Mts::Exchange;


CExchange::CExchange()
        : m_bFIXProtocol(false) {

}


CExchange::CExchange(bool bFIXProtocol)
        : m_bFIXProtocol(bFIXProtocol) {

}


void CExchange::run() {

    m_ptrThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::ref(*this)));
}


bool CExchange::submitCustomOrder(const Mts::Core::CDateTime &dtNow,
        const std::string &strOrderSpec) {

    return true;
}


bool CExchange::cancelCustomOrder(const Mts::Core::CDateTime &dtNow,
        const std::string &strCancelSpec) {

    return true;
}


bool CExchange::submitLmtOrder(const Mts::Order::COrder &objOrder,
        Mts::Exchange::CExchange::TimeInForce iTIF) {

    return approveOrder(objOrder);
}


bool CExchange::submitMktOrder(const Mts::Order::COrder &objOrder) {

    return approveOrder(objOrder);
}


bool CExchange::submitTWAPOrder(const Mts::Order::COrder &objOrder) {

    return approveOrder(objOrder);
}


bool CExchange::submitIcebergOrder(const Mts::Order::COrder &objOrder) {

    return approveOrder(objOrder);
}


bool CExchange::approveOrder(const Mts::Order::COrder &objOrder) {
    /*
	Mts::Risk::CRiskManager::RiskManagerDecision objApproval = Mts::Risk::CRiskManager::getInstance().approveOrder(objOrder);

	if (objApproval != Mts::Risk::CRiskManager::ORDER_APPROVED) {

		Mts::Order::COrder objRejectedOrder(objOrder);
		objRejectedOrder.setOrderState(Mts::Order::COrder::LIMIT_VIOLATION);
		COrderStatus objOrderStatus(objRejectedOrder, Mts::Core::CDateTime::now());
		publishRiskViolation(objOrderStatus);

		return false;
	}
    */
    return true;
}


void CExchange::publishRiskViolation(const Mts::Order::COrderStatus &objOrderStatus) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeRiskViolation(objOrderStatus);
}


void CExchange::publishOrderNew(const Mts::Order::COrder &objOrder) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeOrderNew(objOrder);
}


void CExchange::publishOrderStatus(const Mts::Order::COrderStatus &objOrderStatus) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeOrderStatus(objOrderStatus);
}


void CExchange::publishOrderFill(const Mts::Order::COrderFill &objOrderFill) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeOrderFill(objOrderFill);
}


void CExchange::publishExecReport(const Mts::Order::CExecReport &objExecReport) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeExecReport(objExecReport);
}

void CExchange::publishHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeHeartbeat(objHeartbeat);
}


void CExchange::publishLogon(unsigned int iProviderID) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeLogon(iProviderID);
}


void CExchange::publishLogout(unsigned int iProviderID) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeLogout(iProviderID);
}


void CExchange::addSubscriber(IExchangeSubscriber *ptrSubscriber) {

    m_ExchangeSubscribers.push_back(ptrSubscriber);
}


void CExchange::onEvent(const Mts::OrderBook::CBidAsk &) {

}


bool CExchange::isFIXProtocol() const {

    return m_bFIXProtocol;
}


void CExchange::publishExchangeMessage(const std::string &strMsg) {

    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();

    for (; iter != m_ExchangeSubscribers.end(); ++iter)
        (*iter)->onExchangeMessage(strMsg);
}


void CExchange::publishExecutionReport(const pm::ExecutionReport &er) {
    ExchangeSubscriberArray::iterator iter = m_ExchangeSubscribers.begin();
    for (; iter != m_ExchangeSubscribers.end(); ++iter) {
        // send it directly without assembling/desembling
        // COrders to construct an ExecutionReport
        (*iter)->sendExecutionReport(er);
    }
}
