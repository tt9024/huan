#include <vector>
#include <boost/lexical_cast.hpp>
#include "COrderFactory.h"
#include "CConfig.h"
#include "CMtsException.h"
#include "CDateTime.h"

using namespace Mts::Order;


bool COrderFactory::m_bUseSQL = true;
bool COrderFactory::m_bLoadOrderHistory = true;


COrderFactory &COrderFactory::getInstance() {

    static COrderFactory objTheInstance;
    return objTheInstance;
}


void COrderFactory::setModeSQL(bool bUseSQL) {

    m_bUseSQL = bUseSQL;
}


void COrderFactory::setLoadOrderHistory(bool bLoadOrderHistory) {

    m_bLoadOrderHistory = bLoadOrderHistory;
}


COrderFactory::COrderFactory()
        : m_ulNextOrderID(1000),
          m_ulNextCancellationID(1000) {

    std::string strLogDir = Mts::Core::CConfig::getInstance().getLogFileDirectory();

    // orders may be flat file or DB persisted, cancels still flat file only
    if (m_bUseSQL == false) {

        m_OrderLog.openLog(strLogDir + "/DailyCreateOrderLog.dat");
    }

    m_CancelRequestLog.openLog(strLogDir + "/DailyCreateCancelRequestLog.dat");

    if (m_bLoadOrderHistory == true) {

        loadDailyHistory();
    }

    loadNextOrderID();
}


COrderFactory::~COrderFactory() {

    if (m_bUseSQL == false) {

        m_OrderLog.closeLog();
    }

    m_CancelRequestLog.closeLog();
}


const COrderCancelRequest &COrderFactory::createCancellationRequest(unsigned int iOriginatorAlgoID,
        const COrder &objOrigOrder) {

    Mts::Core::CDateTime dtCreateTimestamp = Mts::Core::CDateTime::now();
    return createCancellationRequest(dtCreateTimestamp, iOriginatorAlgoID, objOrigOrder);
}


const COrderCancelRequest &COrderFactory::createCancellationRequest(
        Mts::Core::CDateTime &dtCreateTime,
        unsigned int iOriginatorAlgoID,
        const COrder &objOrigOrder) {

    boost::mutex::scoped_lock objScopedLock(m_MutexCancellationID);

    unsigned long ulMtsCancellationID = getNextCancellationID();

    std::string strMtsCancellationID = createMtsCancellationID(dtCreateTime, ulMtsCancellationID, iOriginatorAlgoID);

    m_CancelRequest.setOriginatorAlgoID(iOriginatorAlgoID);
    m_CancelRequest.setMtsCancelReqID(strMtsCancellationID.c_str());
    m_CancelRequest.setCreateTimestamp(dtCreateTime);
    m_CancelRequest.setOrigOrder(objOrigOrder);

    m_CancelRequests.insert(std::pair<std::string, COrderCancelRequest>(strMtsCancellationID, m_CancelRequest));
    m_CancelRequestLog.writeToFile(m_CancelRequest);

    return m_CancelRequest;
}


const COrder &COrderFactory::createIOCOrder(unsigned int iOriginatorAlgoID,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        COrder::BuySell iDirection,
        unsigned int uiQuantity,
        double dPrice,
        const std::string &strOrderTag,
        const std::string &strExecBrokerCode) {

    Mts::Core::CDateTime dtCreateTimestamp = Mts::Core::CDateTime::now();
    return createIOCOrder(dtCreateTimestamp, iOriginatorAlgoID, iSymbolID, iProviderID, iDirection, uiQuantity, dPrice,
                          strOrderTag, strExecBrokerCode);
}


const COrder &COrderFactory::createIOCOrder(Mts::Core::CDateTime &dtCreateTime,
        unsigned int iOriginatorAlgoID,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        COrder::BuySell iDirection,
        unsigned int uiQuantity,
        double dPrice,
        const std::string &strOrderTag,
        const std::string &strExecBrokerCode) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);

    unsigned long ulMtsOrderID = getNextOrderID();

    std::string strMtsOrderID = createMtsOrderID(dtCreateTime, ulMtsOrderID, iOriginatorAlgoID);

    m_Order.setMtsOrderID(strMtsOrderID.c_str());
    m_Order.setExcOrderID("");
    m_Order.setExcExecID("");
    m_Order.setCreateTimestamp(dtCreateTime);
    m_Order.setMtsTimestamp(0);
    m_Order.setExcTimestamp(0);
    m_Order.setOriginatorAlgoID(iOriginatorAlgoID);
    m_Order.setSymbolID(iSymbolID);
    m_Order.setProviderID(iProviderID);
    m_Order.setDirection(iDirection);
    m_Order.setQuantity(uiQuantity);
    m_Order.setPrice(dPrice);
    m_Order.setOrderType(COrder::IOC);
    m_Order.setOrderState(COrder::NEW);
    m_Order.setOrderTag(strOrderTag.c_str());
    m_Order.setExecBrokerCode(strExecBrokerCode.c_str());

    m_Orders.insert(std::pair<std::string, COrder>(strMtsOrderID, m_Order));

    if (m_bUseSQL == false) {

#ifndef SIM_MODE
        m_OrderLog.writeToFile(m_Order);
#endif
    }

    return m_Order;
}


void COrderFactory::createIOCOrder(Mts::Core::CDateTime &dtCreateTime,
        unsigned int iOriginatorAlgoID,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        COrder::BuySell iDirection,
        unsigned int uiQuantity,
        double dPrice,
        const std::string &strOrderTag,
        const std::string &strExecBrokerCode,
        COrder &objOrder) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);

    unsigned long ulMtsOrderID = getNextOrderID();

    std::string strMtsOrderID = createMtsOrderID(dtCreateTime, ulMtsOrderID, iOriginatorAlgoID);

    m_Order.setMtsOrderID(strMtsOrderID.c_str());
    m_Order.setExcOrderID("");
    m_Order.setExcExecID("");
    m_Order.setCreateTimestamp(dtCreateTime);
    m_Order.setMtsTimestamp(0);
    m_Order.setExcTimestamp(0);
    m_Order.setOriginatorAlgoID(iOriginatorAlgoID);
    m_Order.setSymbolID(iSymbolID);
    m_Order.setProviderID(iProviderID);
    m_Order.setDirection(iDirection);
    m_Order.setQuantity(uiQuantity);
    m_Order.setPrice(dPrice);
    m_Order.setOrderType(COrder::IOC);
    m_Order.setOrderState(COrder::NEW);
    m_Order.setOrderTag(strOrderTag.c_str());
    m_Order.setExecBrokerCode(strExecBrokerCode.c_str());

    m_Orders.insert(std::pair<std::string, COrder>(strMtsOrderID, m_Order));

    if (m_bUseSQL == false) {

#ifndef SIM_MODE
        m_OrderLog.writeToFile(m_Order);
#endif
    }

    objOrder.setMtsOrderID(strMtsOrderID.c_str());
    objOrder.setExcOrderID("");
    objOrder.setExcExecID("");
    objOrder.setCreateTimestamp(dtCreateTime);
    objOrder.setMtsTimestamp(0);
    objOrder.setExcTimestamp(0);
    objOrder.setOriginatorAlgoID(iOriginatorAlgoID);
    objOrder.setSymbolID(iSymbolID);
    objOrder.setProviderID(iProviderID);
    objOrder.setDirection(iDirection);
    objOrder.setQuantity(uiQuantity);
    objOrder.setPrice(dPrice);
    objOrder.setOrderType(COrder::IOC);
    objOrder.setOrderState(COrder::NEW);
    objOrder.setOrderTag(strOrderTag.c_str());
    objOrder.setExecBrokerCode(strExecBrokerCode.c_str());
}

const COrder &COrderFactory::createGTCOrder(
        const std::string &sMtsOrderID,
        unsigned int iOriginatorAlgoID,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        COrder::BuySell iDirection,
        unsigned int uiQuantity,
        double dLimitPrice,
        const std::string &strOrderTag,
        const std::string &strExecBrokerCode) {

    Mts::Core::CDateTime dtCreateTimestamp = Mts::Core::CDateTime::now();
    return createGTCOrder(dtCreateTimestamp, sMtsOrderID, iOriginatorAlgoID, iSymbolID, iProviderID, iDirection,
                          uiQuantity, dLimitPrice, strOrderTag, strExecBrokerCode);
}


const COrder &COrderFactory::createGTCOrder(
        Mts::Core::CDateTime &dtCreateTime,
        const std::string &strMtsOrderID,
        unsigned int iOriginatorAlgoID,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        COrder::BuySell iDirection,
        unsigned int uiQuantity,
        double dLimitPrice,
        const std::string &strOrderTag,
        const std::string &strExecBrokerCode) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);

    /*
    unsigned long ulMtsOrderID = getNextOrderID();
    std::string strMtsOrderID = createMtsOrderID(dtCreateTime, ulMtsOrderID, iOriginatorAlgoID);
    */

    m_Order.setMtsOrderID(strMtsOrderID.c_str());
    m_Order.setExcOrderID("");
    m_Order.setExcExecID("");
    m_Order.setCreateTimestamp(dtCreateTime);
    m_Order.setMtsTimestamp(0);
    m_Order.setExcTimestamp(0);
    m_Order.setOriginatorAlgoID(iOriginatorAlgoID);
    m_Order.setSymbolID(iSymbolID);
    m_Order.setProviderID(iProviderID);
    m_Order.setDirection(iDirection);
    m_Order.setQuantity(uiQuantity);
    m_Order.setPrice(dLimitPrice);
    m_Order.setOrderType(COrder::GTC);
    m_Order.setOrderState(COrder::NEW);
    m_Order.setOrderTag(strOrderTag.c_str());
    m_Order.setExecBrokerCode(strExecBrokerCode.c_str());

    m_Orders.insert(std::pair<std::string, COrder>(strMtsOrderID, m_Order));
    return m_Order;
}

const COrder &COrderFactory::createIcebergOrder(Mts::Core::CDateTime &dtCreateTime,
        unsigned int iOriginatorAlgoID,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        COrder::BuySell iDirection,
        unsigned int uiQuantity,
        double dLimitPrice,
        const std::string &strOrderTag,
        const std::string &strExecBrokerCode) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);

    unsigned long ulMtsOrderID = getNextOrderID();

    std::string strMtsOrderID = createMtsOrderID(dtCreateTime, ulMtsOrderID, iOriginatorAlgoID);

    m_Order.setMtsOrderID(strMtsOrderID.c_str());
    m_Order.setExcOrderID("");
    m_Order.setExcExecID("");
    m_Order.setCreateTimestamp(dtCreateTime);
    m_Order.setMtsTimestamp(0);
    m_Order.setExcTimestamp(0);
    m_Order.setOriginatorAlgoID(iOriginatorAlgoID);
    m_Order.setSymbolID(iSymbolID);
    m_Order.setProviderID(iProviderID);
    m_Order.setDirection(iDirection);
    m_Order.setQuantity(uiQuantity);
    m_Order.setPrice(dLimitPrice);
    m_Order.setOrderType(COrder::GTC);
    m_Order.setOrderState(COrder::NEW);
    m_Order.setOrderTag(strOrderTag.c_str());
    m_Order.setExecBrokerCode(strExecBrokerCode.c_str());

    m_Orders.insert(std::pair<std::string, COrder>(strMtsOrderID, m_Order));

    if (m_bUseSQL == false) {

#ifndef SIM_MODE
        m_OrderLog.writeToFile(m_Order);
#endif
    }

    return m_Order;
}

const COrder &COrderFactory::createTWAPOrder(Mts::Core::CDateTime &dtCreateTime,
        unsigned int iOriginatorAlgoID,
        unsigned int iSymbolID,
        unsigned int iProviderID,
        COrder::BuySell iDirection,
        unsigned int uiQuantity,
        double dLimitPrice,
        const std::string &strOrderTag,
        const std::string &strExecBrokerCode) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);

    unsigned long ulMtsOrderID = getNextOrderID();

    std::string strMtsOrderID = createMtsOrderID(dtCreateTime, ulMtsOrderID, iOriginatorAlgoID);

    m_Order.setMtsOrderID(strMtsOrderID.c_str());
    m_Order.setExcOrderID("");
    m_Order.setExcExecID("");
    m_Order.setCreateTimestamp(dtCreateTime);
    m_Order.setMtsTimestamp(0);
    m_Order.setExcTimestamp(0);
    m_Order.setOriginatorAlgoID(iOriginatorAlgoID);
    m_Order.setSymbolID(iSymbolID);
    m_Order.setProviderID(iProviderID);
    m_Order.setDirection(iDirection);
    m_Order.setQuantity(uiQuantity);
    m_Order.setPrice(dLimitPrice);
    m_Order.setOrderType(COrder::GTC);
    m_Order.setOrderState(COrder::NEW);
    m_Order.setOrderTag(strOrderTag.c_str());
    m_Order.setExecBrokerCode(strExecBrokerCode.c_str());

    m_Orders.insert(std::pair<std::string, COrder>(strMtsOrderID, m_Order));

    if (m_bUseSQL == false) {

#ifndef SIM_MODE
        m_OrderLog.writeToFile(m_Order);
#endif
    }

    return m_Order;
}


bool COrderFactory::isOrderInCache(const char *pszMtsOrderID) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);

    OrdersMap::iterator iter = m_Orders.find(pszMtsOrderID);

    return iter != m_Orders.end();
}


bool COrderFactory::isCancelRequestInCache(const char *pszMtsCancellationID) {

    boost::mutex::scoped_lock objScopedLock(m_MutexCancellationID);

    CancelRequestMap::iterator iter = m_CancelRequests.find(pszMtsCancellationID);

    return iter != m_CancelRequests.end();
}


COrder &COrderFactory::getOrder(const char *pszMtsOrderID) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);

    OrdersMap::iterator iter = m_Orders.find(pszMtsOrderID);

    if (iter == m_Orders.end())
        throw Mts::Exception::CMtsException("missing order");

    return iter->second;
}


const COrder &COrderFactory::getOrderFromCancellationID(const char *pszMtsCancellationID) {

    boost::mutex::scoped_lock objScopedLock(m_MutexCancellationID);

    CancelRequestMap::iterator iter = m_CancelRequests.find(pszMtsCancellationID);

    if (iter == m_CancelRequests.end())
        throw Mts::Exception::CMtsException("missing cancellation request");

    return iter->second.getOrigOrder();
}


// not to be called once engine has been started, not threadsafe
unsigned long COrderFactory::queryNextOrderID() {

    return m_ulNextOrderID;
}


unsigned long COrderFactory::getNextOrderID() {

    unsigned long ulOrderID = 0;

    ulOrderID = m_ulNextOrderID;
    m_ulNextOrderID += MAX_NUM_ENGINES;

    return ulOrderID;
}


unsigned long COrderFactory::getNextCancellationID() {

    unsigned long ulCancellationID = 0;

    ulCancellationID = m_ulNextCancellationID;
    m_ulNextCancellationID += MAX_NUM_ENGINES;

    return ulCancellationID;
}


void COrderFactory::setGenerateUniqueOrderID(bool bGenerateUniqueOrderID) {

    m_bGenerateUniqueOrderID = bGenerateUniqueOrderID;
}


void COrderFactory::setNextOrderID(unsigned long ulNextOrderID) {

    boost::mutex::scoped_lock objScopedLock(m_MutexOrderID);
    m_ulNextOrderID = ulNextOrderID;
}


void COrderFactory::setNextCancellationID(unsigned long ulNextCancellationID) {

    boost::mutex::scoped_lock objScopedLock(m_MutexCancellationID);
    m_ulNextCancellationID = ulNextCancellationID;
}


std::string COrderFactory::createMtsOrderID(Mts::Core::CDateTime &dtCreateTime,
        unsigned long ulOrderID,
        unsigned int iOriginatorAlgoID) {

    char szBuffer[255];

    if (m_bGenerateUniqueOrderID == true) {

        sprintf(szBuffer, "%05u%02u%02u%02u%03u%04u%02u", dtCreateTime.getJulianDay(), dtCreateTime.getHour(),
                dtCreateTime.getMin(), dtCreateTime.getSec(), dtCreateTime.getMSec(), ulOrderID, iOriginatorAlgoID);
    } else {

        sprintf(szBuffer, "%04u", ulOrderID);
    }

    return szBuffer;
}


std::string COrderFactory::createMtsCancellationID(Mts::Core::CDateTime &dtCreateTime,
        unsigned long ulCancellationID,
        unsigned int iOriginatorAlgoID) {

    char szBuffer[255];
    sprintf(szBuffer, "%lluCX%04u%02u", dtCreateTime.getCMTime(), ulCancellationID, iOriginatorAlgoID);
    return szBuffer;
}


void COrderFactory::loadDailyHistory() {

    if (m_bUseSQL == false) {

        std::vector<Mts::Order::COrder> orders;
        m_OrderLog.readFromFile<Mts::Order::COrder>(orders);

        for (int i = 0; i != orders.size(); ++i) {
            m_Orders.insert(std::pair<std::string, COrder>(orders[i].getMtsOrderID(), orders[i]));
            ++m_ulNextOrderID;
        }
    } else {

        loadDailyHistorySQL();
    }

    // TODO: move cancellations to SQL
    std::vector<Mts::Order::COrderCancelRequest> cancelReqs;
    m_CancelRequestLog.readFromFile<Mts::Order::COrderCancelRequest>(cancelReqs);

    for (int i = 0; i != cancelReqs.size(); ++i) {
        m_CancelRequests.insert(
                std::pair<std::string, COrderCancelRequest>(cancelReqs[i].getMtsCancelReqID(), cancelReqs[i]));
        ++m_ulNextCancellationID;
    }
}


void COrderFactory::loadDailyHistorySQL() {
/*
        Mts::SQL::CMtsDB objDBConn;

        objDBConn.connect();

        std::vector<Mts::Order::COrderStatus> orders = objDBConn.readOrderStatusHistory();

        for (int i = 0; i != orders.size(); ++i) {
                m_Orders.insert(std::pair<std::string, COrder>(orders[i].getOrigOrder().getMtsOrderID(), orders[i].getOrigOrder()));
                ++m_ulNextOrderID;
        }

        objDBConn.disconnect();
*/
}


void COrderFactory::loadNextOrderID() {
    // get millisecond of month as unsigned long (uint32_t)
    // this ensures uniqueness within one month with 1000 orders/second
    auto dtnow = Mts::Core::CDateTime::now();
    m_ulNextOrderID = (dtnow.getDay() * 3600 * 24 + dtnow.getTimeInSec()) * 1000 + (dtnow.getMSec() % 1000);
    AppLog("Starting ClOrdId as %u", m_ulNextOrderID);

    /*
        if (m_bUseSQL == true) {

                Mts::SQL::CMtsDB objDBConn;

                objDBConn.connect();
                m_ulNextOrderID = objDBConn.loadNextID("OrderID");

                // adjust so IDs don't overlap for different engines being run at the same time
                unsigned int iEngineID = Mts::Core::CConfig::getInstance().getEngineID();
                m_ulNextOrderID += iEngineID;

                objDBConn.disconnect();
        } */
}


void COrderFactory::shutDown() {
    /*
        if (m_bUseSQL == true) {

                Mts::SQL::CMtsDB objDBConn;

                objDBConn.connect();
                objDBConn.saveNextID("OrderID", static_cast<unsigned int>(m_ulNextOrderID));
                objDBConn.disconnect();
        }
    */
}


