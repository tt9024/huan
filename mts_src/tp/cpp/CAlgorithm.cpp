#include <exception>
#include <iostream>
#include "CAlgorithm.h"
#include "CEvent.h"
#include "CQuote.h"
#include "COrderStatus.h"
#include "COrderFill.h"
#include "CConfig.h"
#include "CMtsException.h"


using namespace Mts::Algorithm;


CAlgorithm::CAlgorithm(unsigned int					uiAlgoID,
											 const std::string &	strAlgoName,
											 unsigned int					uiEventBufferSizeBytes,
											 unsigned int					uiUpdateMSec)
: m_uiAlgoID(uiAlgoID),
	m_strAlgoName(strAlgoName),
	m_EventQ(uiEventBufferSizeBytes),
	m_ptrThread(),
	m_iOperationalMode(PASSIVE),
	m_iUpdateMSec(uiUpdateMSec),
	m_iUpdateMSecPnL(5000),
	m_dMinTradeIntervalDayFrac(0),
	m_bEventQEnabled(true),
	m_dDailyMaxLoss(0.0),
	m_dCurrPnLUSD(0.0),
	m_dMaxPnLUSD(0.0),
	m_dStartOfDayPnLUSD(0.0),
	m_PositionManager(strAlgoName),
    m_Logger(Mts::Log::CApplicationLog::algoLog(strAlgoName)),
	m_bRecoveryDone(false) {

	m_dUpdateIntervalDayFrac					= static_cast<double>(m_iUpdateMSec) / (24.0 * 60.0 * 60.0 * 1000.0);
	m_dDefaultUpdateIntervalDayFrac		= m_dUpdateIntervalDayFrac;
	m_dThrottledUpdateIntervalDayFrac	= static_cast<double>(10.0) / (24.0 * 60.0 * 60.0 * 1000.0);

	memset(m_TradedInstrumentMap, 0, Mts::Core::CConfig::MAX_NUM_SYMBOLS * sizeof(unsigned int));
	memset(m_iMaxPosition, 0, Mts::Core::CConfig::MAX_NUM_SYMBOLS * sizeof(int));
	memset(m_iMaxOrderSize, 0, Mts::Core::CConfig::MAX_NUM_SYMBOLS * sizeof(int));
}


void CAlgorithm::run() {

	m_ptrThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::ref(*this)));
}


void CAlgorithm::operator()() {

	LogInfo(m_Logger, "Algorithm started");

	try {

		while (isStopped() == false) {

			m_EventQ.lock();

			while (m_EventQ.isReadReady() == false) {
				m_EventQ.wait();
			}

			Mts::Event::CEvent::EventID iEventID = m_EventQ.getNextEventID();

			switch (iEventID) {

				case Mts::Event::CEvent::UNDEFINED:
					m_EventQ.unlock();
					throw Mts::Exception::CMtsException("undefined event read from queue");

				case Mts::Event::CEvent::BID_ASK:
					handleEvent<Mts::OrderBook::CBidAsk>(iEventID);
					break;

				case Mts::Event::CEvent::TRADE:
					handleEvent<Mts::OrderBook::CTrade>(iEventID);
					break;

				case Mts::Event::CEvent::ORDER_STATUS:
					handleEvent<Mts::Order::COrderStatus>(iEventID);
					break;

				case Mts::Event::CEvent::FILL:
					handleEvent<Mts::Order::COrderFill>(iEventID);
					break;

				case Mts::Event::CEvent::EXEC_REPORT:
					handleEvent<Mts::Order::CExecReport>(iEventID);
					break;

				case Mts::Event::CEvent::DATETIME:
					handleEvent<Mts::Core::CDateTimeEvent>(iEventID);
					break;

				case Mts::Event::CEvent::KEY_VALUE:
					handleEvent<Mts::OrderBook::CKeyValue>(iEventID);
					break;
                
                case Mts::Event::CEvent::MANUAL_COMMAND:
                    handleEvent<Mts::OrderBook::CManualCommand>(iEventID);
                    break;

			}
		}
	}
	catch(std::exception & e) {
        LogError(m_Logger,e.what());
	}
}


bool CAlgorithm::onMktBidAsk(const Mts::OrderBook::CBidAsk & objBidAsk) {

	if (m_bEventQEnabled == false)
		return true;

	if (objBidAsk.getTimestamp().getValue() - m_dtLastUpdate[objBidAsk.getSymbolID()].getValue() < m_dUpdateIntervalDayFrac) {

		return true;
	}

	m_dtLastUpdate[objBidAsk.getSymbolID()] = objBidAsk.getTimestamp();

	bool bRet = m_EventQ.push(objBidAsk);

	if (bRet == false) {

		m_dUpdateIntervalDayFrac = m_dThrottledUpdateIntervalDayFrac;

		LogInfo(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push bid-ask");
	}
	else {

		m_dUpdateIntervalDayFrac = m_dDefaultUpdateIntervalDayFrac;
	}

	return bRet;
}


bool CAlgorithm::onTrade(const Mts::OrderBook::CTrade & objTrade) {

	if (m_bEventQEnabled == false)
		return true;

	if (objTrade.getMtsTimestamp().getValue() - m_dtLastUpdate[objTrade.getSymbolID()].getValue() < m_dUpdateIntervalDayFrac) {

		return true;
	}

	m_dtLastUpdate[objTrade.getSymbolID()] = objTrade.getMtsTimestamp();

	bool bRet = m_EventQ.push(objTrade);

	if (bRet == false) {

		m_dUpdateIntervalDayFrac = m_dThrottledUpdateIntervalDayFrac;

		LogInfo(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push trade");
	}
	else {

		m_dUpdateIntervalDayFrac = m_dDefaultUpdateIntervalDayFrac;
	}

	return bRet;
}


bool CAlgorithm::onKeyValue(const Mts::OrderBook::CKeyValue & objKeyValue) {

	bool bRet = m_EventQ.push(objKeyValue);

	if (bRet == false)
		LogError(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push key value");

	return bRet;
}

bool CAlgorithm::onCommand(const Mts::OrderBook::CManualCommand& objMC) {

    bool bRet = m_EventQ.push(objMC);

    if (bRet == false)
        LogError(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push command");

    return bRet;
}



bool CAlgorithm::onOrderStatus(const Mts::Order::COrderStatus & objOrderStatus) {

	bool bRet = m_EventQ.push(objOrderStatus);

	if (bRet == false)
		LogError(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push order status");

	return bRet;
}


bool CAlgorithm::onOrderFill(const Mts::Order::COrderFill & objOrderFill) {

	bool bRet = m_EventQ.push(objOrderFill);

	if (bRet == false)
		LogError(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push order fill");

	return bRet;
}


bool CAlgorithm::onExecReport(const Mts::Order::CExecReport & objExecReport) {

	bool bRet = m_EventQ.push(objExecReport);

	if (bRet == false)
		LogError(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push exec report");

	return bRet;
}


bool CAlgorithm::onDateTime(const Mts::Core::CDateTimeEvent & objDateTime) {

	bool bRet = m_EventQ.push(objDateTime);

	if (bRet == false)
		LogError(m_Logger, m_strAlgoName + " - Algorithm event queue is full, can't push time event");

	return bRet;
}


unsigned int CAlgorithm::getAlgoID() const {

	return m_uiAlgoID;
}


const std::string & CAlgorithm::getAlgoName() const {

	return m_strAlgoName;
}


bool CAlgorithm::isStopped() const {

	return m_bStopped;
}


const Mts::Accounting::CPosition & CAlgorithm::getPosition(const Mts::Core::CSymbol & objSymbol) const {

	return m_PositionManager.getPosition(objSymbol);
}


void CAlgorithm::setPosition(const Mts::Core::CSymbol & objSymbol,
														 const Mts::Accounting::CPosition & objPosition) {

	char szBuffer[512];
	sprintf(szBuffer, "CAlgorithm::setPosition, algoid %d symbol %s position %d", m_uiAlgoID, objSymbol.getSymbol().c_str(), objPosition.getPosition());
	LogInfo(m_Logger, szBuffer);
	m_PositionManager.setPosition(objSymbol, objPosition);
}


bool CAlgorithm::isTradedInstrument(unsigned int iSymbolID) {

	return m_TradedInstrumentMap[iSymbolID] == 1;
}


void CAlgorithm::addTradedInstrument(const Mts::Core::CSymbol & objSymbol) {

	m_TradedInstruments.push_back(objSymbol);
	m_TradedInstrumentMap[objSymbol.getSymbolID()] = 1;
}


void CAlgorithm::addAlgorithmSubscriber(IAlgorithmSubscriber * ptrAlgoritmSubscriber) {

	m_AlgoSubscribers.push_back(ptrAlgoritmSubscriber);
}


void CAlgorithm::broadcastMessage(const std::string & strMsg) const {

	AlgorithmSubscriberArray::const_iterator iter = m_AlgoSubscribers.begin();

	for (; iter != m_AlgoSubscribers.end(); ++iter)
		(*iter)->onAlgorithmMessage(strMsg);
}


void CAlgorithm::broadcastRiskBreach(unsigned int iAlgoID) const {

	AlgorithmSubscriberArray::const_iterator iter = m_AlgoSubscribers.begin();

	for (; iter != m_AlgoSubscribers.end(); ++iter)
		(*iter)->onAlgorithmRiskBreach(iAlgoID);
}


void CAlgorithm::broadcastInternalError(const std::string & strMsg) const {

	AlgorithmSubscriberArray::const_iterator iter = m_AlgoSubscribers.begin();

	for (; iter != m_AlgoSubscribers.end(); ++iter)
		(*iter)->onAlgorithmInternalError(m_uiAlgoID, strMsg);
}


void CAlgorithm::onCreate() {
	LogInfo(m_Logger, "CAlgorithm::onCreate()\n");
}


void CAlgorithm::onEvent(const Mts::Order::COrderFill & objOrderFill) {

	m_PositionManager.updatePosition(objOrderFill);

	const Mts::Core::CSymbol &					objSymbol		= Mts::Core::CSymbol::getSymbol(objOrderFill.getOrigOrder().getSymbolID());
	const Mts::Accounting::CPosition &	objPosition = m_PositionManager.getPosition(objSymbol);

	AlgorithmSubscriberArray::const_iterator iter = m_AlgoSubscribers.begin();

	for (; iter != m_AlgoSubscribers.end(); ++iter)
		(*iter)->onAlgorithmPositionUpdate(m_uiAlgoID, objPosition);

	// post execution risk checks
	enforceMaxPositionLimit(objPosition, objSymbol);
	enforceMaxOrderSizeLimit(objOrderFill, objSymbol);
	enforceMinTradeInterval(objOrderFill, objSymbol);
}


void CAlgorithm::onDestroy() {

	try {
		LogInfo(m_Logger, "CAlgorithm::onDestroy()\n");
	}
	catch(std::exception & e) {
		LogError(m_Logger,e.what());
	}
}


void CAlgorithm::onStart() {

	m_bStopped = false;
}


void CAlgorithm::onStartOfDay(Mts::Core::CDateTime) {

}


void CAlgorithm::onEngineStart(const Mts::TickData::CPriceMatrix &) {

}


void CAlgorithm::onEngineStart(const Mts::TickData::CPriceMatrix &,
															 const Mts::TickData::CPriceMatrix &) {

}


// this method should be called at the end of any algo specific override due to the latency overhead
void CAlgorithm::onEvent(const Mts::OrderBook::CBidAsk & objBidAsk) {

	const Mts::Core::CSymbol &	objSymbol = Mts::Core::CSymbol::getSymbol(objBidAsk.getSymbolID());
	const Mts::Core::CCurrncy & objRefCcy = objSymbol.getRefCcy();
	double											dBestMid	= objBidAsk.getMidPx();

	if (dBestMid <= 0 || objBidAsk.getBid().getPrice() <= 0 || objBidAsk.getAsk().getPrice() <= 0)
		return;

	static double UPDATE_INTERVAL_DAYFRAC = static_cast<double>(m_iUpdateMSecPnL) / (24.0 * 60.0 * 60.0 * 1000.0);

	if (objBidAsk.getTimestamp().getValue() - m_dtLastUpdatePnL[objBidAsk.getSymbolID()].getValue() < UPDATE_INTERVAL_DAYFRAC) {

		return;
	}

	bool bNewDay = objBidAsk.getTimestamp().equalDay(m_dtLastUpdatePnL[objBidAsk.getSymbolID()]) == false;

	m_dtLastUpdatePnL[objBidAsk.getSymbolID()] = objBidAsk.getTimestamp();

	m_PositionManager.updatePnL(objBidAsk);

	// track max pnl for stop
	m_dCurrPnLUSD = m_PositionManager.getTotalUSDPnL();
	m_dMaxPnLUSD	= std::max(m_dMaxPnLUSD, m_dCurrPnLUSD);

	if (bNewDay == true) {

		m_dMaxPnLUSD				= m_dCurrPnLUSD;
		m_dStartOfDayPnLUSD	= m_dCurrPnLUSD;
	}

	broadcastPnL();
}


void CAlgorithm::onEvent(const Mts::OrderBook::CTrade & objTrade) {

}


void CAlgorithm::onEvent(const Mts::Order::COrderStatus &) {

}


void CAlgorithm::onEvent(const Mts::Order::CExecReport &) {

}



void CAlgorithm::onEvent(const Mts::OrderBook::CManualCommand & objMC) {

}


void CAlgorithm::onEvent(const Mts::Core::CDateTimeEvent &) {

}


void CAlgorithm::onEndOfDay(Mts::Core::CDateTime) {

}


void CAlgorithm::onEvent(const Mts::OrderBook::CKeyValue &) {

}

void CAlgorithm::onProviderActive(unsigned int iProviderID) {

	if (m_ProviderStatus.find(iProviderID) == m_ProviderStatus.end()) {
		m_ProviderStatus.insert(std::pair<unsigned int, bool>(iProviderID, true));
	}
	else {
		m_ProviderStatus[iProviderID] = true;
	}
}


void CAlgorithm::onProviderInactive(unsigned int iProviderID) {

	if (m_ProviderStatus.find(iProviderID) == m_ProviderStatus.end()) {
		m_ProviderStatus.insert(std::pair<unsigned int, bool>(iProviderID, false));
	}
	else {
		m_ProviderStatus[iProviderID] = false;
	}
}


bool CAlgorithm::isExchangeAvailable(unsigned int iProviderID) const {

	ProviderAvailabilityMap::const_iterator iter = m_ProviderStatus.find(iProviderID);

	if (iter == m_ProviderStatus.end()) {
		return false;
	}
	else {
		return iter->second;
	}
}


void CAlgorithm::onStop() {

	m_bStopped = true;
}


std::vector<std::string> CAlgorithm::getMessageHistory() const {

	std::vector<std::string> objMessages;
	return objMessages;
}


std::vector<std::string> CAlgorithm::getState() const {

	std::vector<std::string> objMessages;

	for (size_t i = 0; i != m_TradedInstruments.size(); ++i) {

		objMessages.push_back(buildPositionMessage(m_TradedInstruments[i]));
	}

	objMessages.push_back(buildPnLUSDMessage());

	return objMessages;
}


void CAlgorithm::broadcastPosition(const Mts::Core::CSymbol & objSymbol) const {

	broadcastMessage(buildPositionMessage(objSymbol));
}


void CAlgorithm::broadcastPnL() const {

	broadcastMessage(buildPnLUSDMessage());
}


double CAlgorithm::getPnLUSD() const {

	return m_PositionManager.getTotalUSDPnL();
}


std::string CAlgorithm::buildPositionMessage(const Mts::Core::CSymbol & objSymbol) const {

	double dPosition = getPosition(objSymbol).getPosition();

	char szBuffer[512];
	sprintf(szBuffer, "ALGO_POSITION,%d,%s,%d,%s,%d,%s,%d", getAlgoID(), objSymbol.getSymbol().c_str(), static_cast<int>(dPosition), objSymbol.getBaseCcy().getCcy().c_str(), static_cast<int>(dPosition), objSymbol.getRefCcy().getCcy().c_str(), static_cast<int>(dPosition));
	
	return szBuffer;
}


std::string CAlgorithm::buildPnLUSDMessage() const {

	double dPnLUSD = m_PositionManager.getTotalUSDPnL();

	char szBuffer[512];
	sprintf(szBuffer, "ALGO_PNL,%d,%d", getAlgoID(), static_cast<int>(dPnLUSD));

	return szBuffer;
}


double CAlgorithm::getUsdPosition() const {

	static Mts::Core::CCurrncy objUSD = Mts::Core::CCurrncy::getCcy("USD");

	return 0.0;
}


double CAlgorithm::getCcyUsdPosition(const Mts::Core::CCurrncy & objBaseCcy) const {

	return 0.0;
}


std::string CAlgorithm::getAlgoInitString() const {

	return "";
}


void CAlgorithm::setMaxPosition(const Mts::Core::CSymbol & objSymbol, 
																int												iMaxPosition) {

	m_iMaxPosition[objSymbol.getSymbolID()] = iMaxPosition;
}


void CAlgorithm::setMaxOrderSize(const Mts::Core::CSymbol & objSymbol, 
																 int												iMaxOrderSize) {

	m_iMaxOrderSize[objSymbol.getSymbolID()] = iMaxOrderSize;
}


void CAlgorithm::setMinTradeIntervalSecs(int iMinTradeIntervalInSecs) {

	m_dMinTradeIntervalDayFrac = static_cast<double>(iMinTradeIntervalInSecs) / (24.0 * 60.0 * 60.0);
}


// after the event check which will shut down the algo if limit breached. backup to higher level pre-event checks.
void CAlgorithm::enforceMaxPositionLimit(const Mts::Accounting::CPosition &	objPosition,
																				 const Mts::Core::CSymbol &					objSymbol) {

	if (abs(static_cast<int>(objPosition.getPosition())) > m_iMaxPosition[objSymbol.getSymbolID()]) {

		setOperationalMode(PASSIVE);
		LogError(m_Logger, m_strAlgoName + " - Algorithm has breached position limit for " + objSymbol.getSymbol() + " and has been set to PASSIVE");
		broadcastRiskBreach(m_uiAlgoID);
	}
}


// after the event check which will shut down the algo if limit breached. backup to higher level pre-event checks.
void CAlgorithm::enforceMaxOrderSizeLimit(const Mts::Order::COrderFill &	objOrderFill,
																					const Mts::Core::CSymbol &			objSymbol) {

	if (abs(static_cast<int>(objOrderFill.getOrigOrder().getQuantity())) > m_iMaxOrderSize[objSymbol.getSymbolID()]) {

		setOperationalMode(PASSIVE);
		LogError(m_Logger, m_strAlgoName + " - Algorithm has breached order size limit for " + objSymbol.getSymbol() + " and has been set to PASSIVE");
		broadcastRiskBreach(m_uiAlgoID);
	}
}


// after the event check which will shut down the algo if trade throttle breached. backup to higher level pre-event checks.
void CAlgorithm::enforceMinTradeInterval(const Mts::Order::COrderFill &	objOrderFill,
																				 const Mts::Core::CSymbol &			objSymbol) {

	if (objOrderFill.getFillTimestamp().getValue() - m_dtLastFillTime[objSymbol.getSymbolID()].getValue() < m_dMinTradeIntervalDayFrac) {

		setOperationalMode(PASSIVE);
		LogError(m_Logger, m_strAlgoName + " - Algorithm has breached order throttle for " + objSymbol.getSymbol() + " and has been set to PASSIVE");
		broadcastRiskBreach(m_uiAlgoID);
	}

	m_dtLastFillTime[objSymbol.getSymbolID()] = objOrderFill.getFillTimestamp();
}


// pre-order submission check, must be called explicitly by derived class
bool CAlgorithm::isOrderCompliant(const Mts::Order::COrder & objOrder) {

	const Mts::Core::CSymbol & objSymbol = Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID());

	long iCurrPos				= getPosition(objSymbol).getPosition();

	bool bLimitBreach		= static_cast<int>(objOrder.getQuantity()) > m_iMaxOrderSize[objSymbol.getSymbolID()] ||
												iCurrPos + (objOrder.getDirection() == Mts::Order::COrder::BUY ? 1 : -1) * static_cast<int>(objOrder.getQuantity()) > m_iMaxPosition[objSymbol.getSymbolID()];

	bool bMinSizeBreach	=	objOrder.getQuantity() == 0;

	bool bPnLBreach			= isPnLStopHit();

	if (bLimitBreach == true || bMinSizeBreach || bPnLBreach) {

		if (bPnLBreach) {
			setOperationalMode(LIQUIDATE);
			LogError(m_Logger, m_strAlgoName + " - Algorithm PnL stop loss hit for " + objSymbol.getSymbol() + " and has been set to LIQUIDATE");
		}
		else {
			setOperationalMode(PASSIVE);
			LogError(m_Logger, m_strAlgoName + " - Algorithm has attempted to breach position or order size limit for " + objSymbol.getSymbol() + " and has been set to PASSIVE");
			broadcastRiskBreach(m_uiAlgoID);
		}
	}

	return bLimitBreach == false;
}


void CAlgorithm::enableEventQueue() {

	m_bEventQEnabled = true;
}


void CAlgorithm::disableEventQueue() {

	m_bEventQEnabled = false;

	m_EventQ.reset();
}


bool CAlgorithm::isPnLStopHit() const {

	return m_dDailyMaxLoss < 0 && m_dCurrPnLUSD <= m_dDailyMaxLoss;
}


void CAlgorithm::resetStop() {

	m_dMaxPnLUSD = m_dCurrPnLUSD;
}


double CAlgorithm::getDailyDD() const {

	return m_dCurrPnLUSD - m_dMaxPnLUSD;
}


double CAlgorithm::getDailyPnL() const {

	return m_dCurrPnLUSD - m_dStartOfDayPnLUSD;
}


bool CAlgorithm::isConcentrationLimitBreached(const Mts::Core::CSymbol &	objSymbol,
																							Mts::Order::COrder::BuySell	iDirection,
																							unsigned int								iQuantityUSD,
																							unsigned int								iCcyPosLimitUSD) const {

	double dCurrPosBaseCcyUSD = CAlgorithm::getCcyUsdPosition(objSymbol.getBaseCcy());
	double dCurrPosRefCcyUSD	= CAlgorithm::getCcyUsdPosition(objSymbol.getRefCcy());

	double dNewPosBaseCcyUSD  = dCurrPosBaseCcyUSD + (iDirection == Mts::Order::COrder::BUY ? 1 : -1) * static_cast<double>(iQuantityUSD);
	double dNewPosRefCcyUSD		= dCurrPosRefCcyUSD - (iDirection == Mts::Order::COrder::BUY ? 1 : -1) * static_cast<double>(iQuantityUSD);

	return fabs(dNewPosBaseCcyUSD) > iCcyPosLimitUSD || fabs(dNewPosRefCcyUSD) > iCcyPosLimitUSD;
}


void CAlgorithm::applyRiskMultiplier(double dMultiplier) {

	for (int i = 0; i != Mts::Core::CConfig::MAX_NUM_SYMBOLS; ++i) {

		m_iMaxPosition[i]	= static_cast<int>(static_cast<double>(m_iMaxPosition[i]) * dMultiplier);
		m_iMaxOrderSize[i] = static_cast<int>(static_cast<double>(m_iMaxOrderSize[i]) * dMultiplier);
	}
}


Mts::Accounting::CPositionManager	CAlgorithm::getPositionManager() const {

	return m_PositionManager;
}


void CAlgorithm::onRecoveryDone() {

	m_bRecoveryDone = true;
}


bool CAlgorithm::isRecoveryDone() const {

	return m_bRecoveryDone;
}
