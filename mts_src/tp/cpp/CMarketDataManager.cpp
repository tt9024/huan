#include "CMarketDataManager.h"


using namespace Mts::TickData;


CMarketDataManager::CMarketDataManager(unsigned int uiEventBufferSizeBytes,
        unsigned int uiUpdateMSec,
        bool bUseSQL)
        : CAlgorithm(0, "Market Data", uiEventBufferSizeBytes, uiUpdateMSec),
          m_bUseSQL(bUseSQL) {

    memset(m_dIntervalEndTime, 0, sizeof(double) * Mts::Core::CConfig::MAX_NUM_SYMBOLS);
}


CMarketDataManager::~CMarketDataManager() {

}


void CMarketDataManager::onCreate() {

    CAlgorithm::onCreate();
}


void CMarketDataManager::onStart() {

    CAlgorithm::onStart();

}


void CMarketDataManager::onStartOfDay(Mts::Core::CDateTime dtTimestamp) {

}


void CMarketDataManager::onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) {

    // used for building 5 min bars (300 seconds)
    static double dWriteIntervalDayFrac = (static_cast<double>(300) / (24.0 * 3600.0));

    // used for building 1 min bars (60 seconds)
    static double dWriteIntervalDayFracHF = (static_cast<double>(60) / (24.0 * 3600.0));

    /*
	if (m_bUseSQL == false)
		return;


	if (objBidAsk.getBid().getSize() == 0 || objBidAsk.getBid().getPrice() == 0.0 ||
			objBidAsk.getAsk().getSize() == 0 || objBidAsk.getAsk().getPrice() == 0.0 )
		return;


	const Mts::Core::CDateTime	dtTimestamp = objBidAsk.getTimestamp();
	unsigned int								iSymbolID		= objBidAsk.getSymbolID();
	double											dBestMid		= objBidAsk.getMidPx();
	double											dSpreadBps	= 10000.0 * (objBidAsk.getAsk().getPrice() - objBidAsk.getBid().getPrice()) / dBestMid;


	Mts::OrderBook::CToBSnapshot objToB(dtTimestamp,
																			iSymbolID,
																			objBidAsk.getBid().getPrice(),
																			objBidAsk.getAsk().getPrice(),
																			objBidAsk.getBid().getSize(),
																			objBidAsk.getAsk().getSize(),
																			0,
																			0,
																			dSpreadBps);


	// compute the initial interval end time (multiple of the write interval) on the first event
	if (m_dIntervalEndTime[iSymbolID] <= 0) {
		m_dIntervalEndTime[iSymbolID] = computeIntervalEndTime(dtTimestamp, dWriteIntervalDayFrac);
	}


	// fill gap since last interval
	if (dtTimestamp.getValue() > m_dIntervalEndTime[iSymbolID]) {

		while (m_dIntervalEndTime[iSymbolID] <= dtTimestamp.getValue()) {
			long long iCMTimeIntervalEnd = Mts::Core::CDateTime::julian2CMTime(m_dIntervalEndTime[iSymbolID]);

			m_TopOfBookSnapshot[iSymbolID].setTimestamp(Mts::Core::CDateTime(m_dIntervalEndTime[iSymbolID]));
			m_DBConn.write(m_TopOfBookSnapshot[iSymbolID], m_LastTrade[iSymbolID], 5);

			m_dIntervalEndTime[iSymbolID] += dWriteIntervalDayFrac;
		}

		m_dIntervalEndTime[iSymbolID] = computeIntervalEndTime(dtTimestamp, dWriteIntervalDayFrac);
	}


	// compute the initial interval end time (multiple of the write interval) on the first event
	if (m_dIntervalEndTimeHF[iSymbolID] <= 0) {
		m_dIntervalEndTimeHF[iSymbolID] = computeIntervalEndTime(dtTimestamp, dWriteIntervalDayFracHF);
	}


	// fill gap since last interval
	if (dtTimestamp.getValue() > m_dIntervalEndTimeHF[iSymbolID]) {

		while (m_dIntervalEndTimeHF[iSymbolID] <= dtTimestamp.getValue()) {
			long long iCMTimeIntervalEnd = Mts::Core::CDateTime::julian2CMTime(m_dIntervalEndTimeHF[iSymbolID]);

			m_TopOfBookSnapshot[iSymbolID].setTimestamp(Mts::Core::CDateTime(m_dIntervalEndTimeHF[iSymbolID]));
			m_DBConn.write(m_TopOfBookSnapshot[iSymbolID], m_LastTrade[iSymbolID], 1);

			m_dIntervalEndTimeHF[iSymbolID] += dWriteIntervalDayFracHF;
		}

		m_dIntervalEndTimeHF[iSymbolID] = computeIntervalEndTime(dtTimestamp, dWriteIntervalDayFracHF);
	}


	m_TopOfBookSnapshot[iSymbolID] = objToB;
    */
}


void CMarketDataManager::onEvent(const Mts::OrderBook::CTrade &objTrade) {

    m_LastTrade[objTrade.getSymbolID()] = objTrade;
}


void CMarketDataManager::onEvent(const Mts::Order::COrderStatus &objOrderStatus) {

}


void CMarketDataManager::onEvent(const Mts::Order::COrderFill &objOrderFill) {

}


void CMarketDataManager::onEvent(const std::string &strUserCommand) {

}


void CMarketDataManager::onEndOfDay(Mts::Core::CDateTime dtTimestamp) {

}


void CMarketDataManager::onStop() {

    CAlgorithm::onStop();
    /*
	if (m_bUseSQL == true)
		m_DBConn.disconnect();
    */
}


void CMarketDataManager::onDestroy() {

    CAlgorithm::onDestroy();
}


double CMarketDataManager::computeIntervalEndTime(const Mts::Core::CDateTime &dtTimestamp,
        double dWriteIntervalDayFrac) const {

    double dJulianDate = dtTimestamp.getValue();
    double dRem = fmod(dJulianDate, dWriteIntervalDayFrac);
    return dJulianDate - dRem + dWriteIntervalDayFrac;
}





