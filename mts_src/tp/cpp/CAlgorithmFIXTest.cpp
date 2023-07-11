#include <chrono>
#include <iostream>
#include <boost/optional.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/unordered_map.hpp>
#include "CAlgorithmFIXTest.h"
#include "CExchangeBroker.h"
#include "CExchange.h"
#include "COrderFactory.h"
#include "COrder.h"
#include "CMath.h"
#include "CStringTokenizer.h"
#include "CApplicationLog.h"
#include "CUserManager.h"


using namespace Mts::Algorithm;


CAlgorithmFIXTest::CAlgorithmFIXTest(unsigned int uiAlgoID,
        const std::string &strAlgoName,
        unsigned int uiEventBufferSizeBytes,
        unsigned int uiUpdateMSec,
        unsigned int uiTestID)
        : CAlgorithm(uiAlgoID, strAlgoName, uiEventBufferSizeBytes, uiUpdateMSec),
          m_iTestID(uiTestID),
          m_bOrderSent(false),
          m_bCancelSent(false) {

}


CAlgorithmFIXTest::~CAlgorithmFIXTest() {

}


void CAlgorithmFIXTest::onCreate() {

    CAlgorithm::onCreate();
}


void CAlgorithmFIXTest::onStart() {

    CAlgorithm::onStart();
}


void CAlgorithmFIXTest::onStartOfDay(Mts::Core::CDateTime dtTimestamp) {

}


void CAlgorithmFIXTest::onEvent(const Mts::Core::CDateTimeEvent &objDateTime) {

}


void CAlgorithmFIXTest::onEvent(const Mts::OrderBook::CTrade &objTrade) {

}


void CAlgorithmFIXTest::onEvent(const Mts::OrderBook::CKeyValue &objKeyValue) {

}


void CAlgorithmFIXTest::onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) {

    boost::mutex::scoped_lock objScopedLock(m_Mutex);

    //char szBuffer[512];

    try {

        Mts::Core::CDateTime dtNow = objBidAsk.getTimestamp();
        const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider("TT");
        unsigned int iSymbolID = objBidAsk.getSymbolID();
        const Mts::Core::CSymbol &objMtsSymbol = Mts::Core::CSymbol::getSymbol(iSymbolID);
        unsigned int iProviderID = objProvider.getProviderID();

        boost::optional<std::reference_wrapper<Mts::Exchange::CExchange> > objExchange = Mts::Exchange::CExchangeBroker::getInstance().getExchange(
                iProviderID);

        // publish position
        //std::cout << "POS - " << objMtsSymbol.getSymbol() << " " << getPosition(objMtsSymbol).getPosition() << std::endl;

        // abort if trading disabled
        if (getOperationalMode() == PASSIVE) {

            return;
        }


        // test order
        int iOrderQty = 12;
        Mts::Order::COrder::BuySell iDirection = iOrderQty > 0 ? Mts::Order::COrder::BUY : Mts::Order::COrder::SELL;
        unsigned int iQuantity = static_cast<unsigned int>(abs(iOrderQty));
        double dIOCPrice =
                iDirection == Mts::Order::COrder::BUY ? objBidAsk.getAsk().getPrice() : objBidAsk.getBid().getPrice();
        double dGTCPrice =
                iDirection == Mts::Order::COrder::BUY ? objBidAsk.getBid().getPrice() : objBidAsk.getAsk().getPrice();
        const Mts::OrderBook::CQuote &objQuote =
                iDirection == Mts::Order::COrder::BUY ? objBidAsk.getAsk() : objBidAsk.getBid();
        std::string strExecBrokerCode = "TT";


        // IOC
        if (m_iTestID == 0) {

            if (m_bOrderSent == true)
                return;

            if (m_Orders[iSymbolID].getQuantity() != 0)
                return;

            Mts::Order::COrderFactory::getInstance().createIOCOrder(dtNow, getAlgoID(), iSymbolID, iProviderID,
                                                                    iDirection, iQuantity, dIOCPrice, "Algo",
                                                                    strExecBrokerCode, m_Orders[iSymbolID]);

            if (isOrderCompliant(m_Orders[iSymbolID]) == true) {

                objExchange.value().get().submitMktOrder(m_Orders[iSymbolID]);
            }

            m_Logger.log(m_Orders[iSymbolID].toString());

            m_bOrderSent = true;
        }

        // GTC (TIF = day)
        if (m_iTestID == 1) {

            if (m_bOrderSent == true)
                return;

            if (m_Orders[iSymbolID].getQuantity() != 0)
                return;

            m_Orders[iSymbolID] = Mts::Order::COrderFactory::getInstance().createGTCOrder(dtNow, "clOrdId", getAlgoID(),
                                                                                          iSymbolID, iProviderID,
                                                                                          iDirection, iQuantity,
                                                                                          dGTCPrice, "Algo",
                                                                                          strExecBrokerCode);

            if (isOrderCompliant(m_Orders[iSymbolID]) == true) {

                objExchange.value().get().submitLmtOrder(m_Orders[iSymbolID], Mts::Exchange::CExchange::GTC);
            }

            m_Logger.log(m_Orders[iSymbolID].toString());

            m_bOrderSent = true;
        }

        // Iceberg
        if (m_iTestID == 2) {

            if (m_bOrderSent == true)
                return;

            if (m_Orders[iSymbolID].getQuantity() != 0)
                return;

            m_Orders[iSymbolID] = Mts::Order::COrderFactory::getInstance().createIcebergOrder(dtNow, getAlgoID(),
                                                                                              iSymbolID, iProviderID,
                                                                                              iDirection, iQuantity,
                                                                                              dGTCPrice, "Algo",
                                                                                              strExecBrokerCode);

            if (isOrderCompliant(m_Orders[iSymbolID]) == true) {

                objExchange.value().get().submitIcebergOrder(m_Orders[iSymbolID]);
            }

            m_Logger.log(m_Orders[iSymbolID].toString());

            m_bOrderSent = true;
        }

        // TimeSliced
        if (m_iTestID == 3) {

            if (m_bOrderSent == true)
                return;

            if (m_Orders[iSymbolID].getQuantity() != 0)
                return;

            m_Orders[iSymbolID] = Mts::Order::COrderFactory::getInstance().createTWAPOrder(dtNow, getAlgoID(),
                                                                                           iSymbolID, iProviderID,
                                                                                           iDirection, iQuantity,
                                                                                           dGTCPrice, "Algo",
                                                                                           strExecBrokerCode);

            if (isOrderCompliant(m_Orders[iSymbolID]) == true) {

                objExchange.value().get().submitTWAPOrder(m_Orders[iSymbolID]);
            }

            m_Logger.log(m_Orders[iSymbolID].toString());

            m_bOrderSent = true;
        }
    }
    catch (std::exception &e) {

        m_Logger.log(e.what());
    }
}


void CAlgorithmFIXTest::onEvent(const Mts::Order::COrderStatus &objOrderStatus) {

    boost::mutex::scoped_lock objScopedLock(m_Mutex);

    m_Logger.log(objOrderStatus.toString());

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrderStatus.getOrigOrder().getSymbolID());

    if (objOrderStatus.getStatus() == Mts::Order::COrder::FILLED ||
        objOrderStatus.getStatus() == Mts::Order::COrder::CANCELLED ||
        //objOrderStatus.getStatus() == Mts::Order::COrder::REJECTED ||
        objOrderStatus.getStatus() == Mts::Order::COrder::LIMIT_VIOLATION) {

        m_Orders[objSymbol.getSymbolID()].setQuantity(0);
        m_CancelReqs[objSymbol.getSymbolID()].setMtsCancelReqID("");
    }

    if (objOrderStatus.getStatus() == Mts::Order::COrder::REJECTED) {

        m_Logger.log("ORDER CANCEL REJECTED");
    }

    if (objOrderStatus.getStatus() == Mts::Order::COrder::CANCELLED) {

        m_Logger.log("ORDER CANCELLED");
    }
}


void CAlgorithmFIXTest::onEvent(const Mts::Order::COrderFill &objOrderFill) {

    boost::mutex::scoped_lock objScopedLock(m_Mutex);

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrderFill.getOrigOrder().getSymbolID());

    CAlgorithm::onEvent(objOrderFill);
    m_Logger.log(objOrderFill.toString());

    if (objOrderFill.getOrigOrder().getOrderState() == Mts::Order::COrder::FILLED) {

        m_Orders[objSymbol.getSymbolID()].setQuantity(0);
        m_CancelReqs[objSymbol.getSymbolID()].setMtsCancelReqID("");
        m_Logger.log("ORDER COMPLETE");
    }

    // broadcast updated position and P&L
    const Mts::Accounting::CPosition &objPosition = m_PositionManager.getPosition(objSymbol);

    m_Logger.log(objPosition.toString());
    broadcastPosition(objSymbol);
}


void CAlgorithmFIXTest::onEvent(const std::string &strUserCommand) {

}


void CAlgorithmFIXTest::onEndOfDay(Mts::Core::CDateTime dtTimestamp) {

}


void CAlgorithmFIXTest::onStop() {

    CAlgorithm::onStop();
}


void CAlgorithmFIXTest::onDestroy() {

    CAlgorithm::onDestroy();
}


std::vector<std::string> CAlgorithmFIXTest::getMessageHistory() const {

    std::vector<std::string> messageHistory;

    return messageHistory;
}


void CAlgorithmFIXTest::addTradedInstrument(const Mts::Core::CSymbol &objSymbol) {

    CAlgorithm::addTradedInstrument(objSymbol);
}


