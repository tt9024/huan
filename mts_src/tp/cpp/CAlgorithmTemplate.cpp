#include <chrono>
#include <iostream>
#include <boost/optional.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/unordered_map.hpp>
#include "CAlgorithmCommand.hxx"
#include "CAlgorithmTemplate.h"
#include "CExchangeBroker.h"
#include "CExchange.h"
#include "COrderFactory.h"
#include "COrder.h"
#include "CMath.h"
#include "CStringTokenizer.h"
#include "CApplicationLog.h"
#include "CUserManager.h"


using namespace Mts::Algorithm;


CAlgorithmTemplate::CAlgorithmTemplate(unsigned int uiAlgoID,
        const std::string &strAlgoName,
        unsigned int uiEventBufferSizeBytes,
        unsigned int uiUpdateMSec,
        unsigned int uiExecAlgoID)
        : CAlgorithm(uiAlgoID, strAlgoName, uiEventBufferSizeBytes, uiUpdateMSec),
          m_iExecAlgoID(uiExecAlgoID) {

    m_dtLastUpdate = 0;

    memset(m_iTradeTimeMap, 0, sizeof(unsigned int) * MINS_PER_DAY);
    memset(m_dLastBidPx, 0, sizeof(double) * Mts::Core::CConfig::MAX_NUM_SYMBOLS);
    memset(m_dLastAskPx, 0, sizeof(double) * Mts::Core::CConfig::MAX_NUM_SYMBOLS);
}


CAlgorithmTemplate::~CAlgorithmTemplate() {

}


void CAlgorithmTemplate::onCreate() {

    CAlgorithm::onCreate();
}


void CAlgorithmTemplate::onStart() {

    CAlgorithm::onStart();
}


void CAlgorithmTemplate::onStartOfDay(Mts::Core::CDateTime dtTimestamp) {

}


void CAlgorithmTemplate::onEvent(const Mts::Core::CDateTimeEvent &objDateTime) {

}


void CAlgorithmTemplate::onEvent(const Mts::OrderBook::CTrade &objTrade) {

}


void CAlgorithmTemplate::onEvent(const Mts::OrderBook::CKeyValue &objKeyValue) {

    // update sub-models (key-value pair)
    for (auto ptrModel: m_Models) {

        ptrModel->onEvent(objKeyValue);
    }
}


void CAlgorithmTemplate::onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) {

    auto dtStart = std::chrono::high_resolution_clock::now();

    bool bRet = onEvent_Internal(objBidAsk);
    CAlgorithm::onEvent(objBidAsk);

    auto dtEnd = std::chrono::high_resolution_clock::now();

    long long iNumMicroSecs = std::chrono::duration_cast<std::chrono::microseconds>(dtEnd - dtStart).count();

    if (bRet == true) {

        char szBuffer[512];
        sprintf(szBuffer, "Elapsed time microseconds %llu", iNumMicroSecs);

        LogInfo(m_Logger, szBuffer);
    }
}


bool CAlgorithmTemplate::onEvent_Internal(const Mts::OrderBook::CBidAsk &objBidAsk) {

    boost::mutex::scoped_lock objScopedLock(m_Mutex);

    char szBuffer[512];

    try {

        Mts::Core::CDateTime dtNow = objBidAsk.getTimestamp();
        const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider("TT");
        unsigned int iProviderID = objProvider.getProviderID();

        m_dLastBidPx[objBidAsk.getSymbolID()] = objBidAsk.getBid().getPrice();
        m_dLastAskPx[objBidAsk.getSymbolID()] = objBidAsk.getAsk().getPrice();
        m_dtLast[objBidAsk.getSymbolID()] = dtNow;


        // abort if exchange recovery process not completed
        if (isRecoveryDone() == false) {

            return false;
        }


        // update sub-models (market data)
        for (auto ptrModel: m_Models) {

            ptrModel->onEvent(objBidAsk);
        }


        // abort if trading disabled
        if (getOperationalMode() == PASSIVE) {

            return false;
        }


        // check if this is a trade time based upon the trading schedule
        if (m_iTradeTimeMap[dtNow.getTimeInMin()] == 0 || dtNow.getTimeInMin() == m_dtLastUpdate.getTimeInMin())
            return false;


        // check all symbols have updated price since the event trigger
        if (false) {

            for (auto &objRefSymbol: m_TradedInstruments) {

                unsigned int iRefSymbolID = objRefSymbol.getSymbolID();

                // FIXME
                // getTimeInMin() gets intra-day min, problem with turn of a day
                if (m_dtLast[iRefSymbolID].getTimeInMin() < dtNow.getTimeInMin())
                    return false;
            }
        }

        // cancel all open orders. 
        // NOTE: the getQuantity() is not changed, 
        // so the symbol in cancel will not issue new orders
        // cancel any TWAP executions still in progress
        boost::optional<std::reference_wrapper<Mts::Exchange::CExchange> > objExchange = Mts::Exchange::CExchangeBroker::getInstance().getExchange(
                iProviderID);

        unsigned int iNumOpenOrders = 0;

        for (auto &objRefSymbol: m_TradedInstruments) {

            unsigned int iRefSymbolID = objRefSymbol.getSymbolID();

            if (m_Orders[iRefSymbolID].getQuantity() != 0) {

                if (strlen(m_CancelReqs[iRefSymbolID].getMtsCancelReqID()) == 0) {

                    m_CancelReqs[iRefSymbolID] =
                            Mts::Order::COrderFactory::getInstance().createCancellationRequest(
                                    dtNow, getAlgoID(), m_Orders[iRefSymbolID]
                            );

                    objExchange.value().get().cancelOrder(m_CancelReqs[iRefSymbolID]);

                    LogInfo(m_Logger, "CANCEL REQUESTED");
                }

                ++iNumOpenOrders;
            }
        }

        if (iNumOpenOrders != 0)
            return false;

        sprintf(szBuffer, "%llu RECALCULATING....", dtNow.getCMTime());
        LogInfo(m_Logger, szBuffer);


        // ready to implement new desired position
        m_dtLastUpdate = dtNow;


        // update sub-models (signals)
        for (auto ptrModel: m_Models) {

            ptrModel->updateSignal(dtNow);
        }


        // trade all symbols in the model
        for (auto &objRefSymbol: m_TradedInstruments) {

            unsigned int iRefSymbolID = objRefSymbol.getSymbolID();
            double dRefMidPx = 0.5 * m_dLastBidPx[iRefSymbolID] + 0.5 * m_dLastAskPx[iRefSymbolID];


            // check if we are within a blackout period
            bool bInTradingWindow = objRefSymbol.isWithinTradingWindow(dtNow);
            bool bInSettWindow = objRefSymbol.isWithinSettWindow(dtNow);

            // abort if outside of trading hours or in the settlement window
            if (bInTradingWindow == false || bInSettWindow == true)
                continue;


            // internalize trades if possible
            boost::unordered_map<std::string, std::tuple<boost::shared_ptr<Mts::Model::CModel>, int, int>> objBuyers;
            boost::unordered_map<std::string, std::tuple<boost::shared_ptr<Mts::Model::CModel>, int, int>> objSellers;

            int iTotalCurrPos = 0;
            int iTotalDesiredPos = 0;
            int iTotalQtyToBuy = 0;
            int iTotalQtyToSell = 0;

            for (auto ptrModel: m_Models) {

                const std::string &strModelName = ptrModel->getModelName();
                int iModelCurrPos = ptrModel->getCurrPosition(iRefSymbolID);
                int iModelDesiredPos = ptrModel->getDesiredPosition(iRefSymbolID);
                int iModelOrderQty = iModelDesiredPos - iModelCurrPos;

                if (iModelOrderQty > 0) {

                    objBuyers.insert(
                            std::pair<std::string, std::tuple<boost::shared_ptr<Mts::Model::CModel>, int, int>>(
                                    ptrModel->getModelName(), std::make_tuple(ptrModel, iModelOrderQty, 0)));
                    iTotalQtyToBuy += iModelOrderQty;
                }

                if (iModelOrderQty < 0) {

                    objSellers.insert(
                            std::pair<std::string, std::tuple<boost::shared_ptr<Mts::Model::CModel>, int, int>>(
                                    ptrModel->getModelName(), std::make_tuple(ptrModel, abs(iModelOrderQty), 0)));
                    iTotalQtyToSell += abs(iModelOrderQty);
                }

                iTotalCurrPos += iModelCurrPos;
                iTotalDesiredPos += iModelDesiredPos;

                sprintf(szBuffer, "Model: %s Current position: %d Desired position: %d", strModelName.c_str(),
                        iModelCurrPos, iModelDesiredPos);
                LogInfo(m_Logger, szBuffer);
            }

            int iNettableQty = std::min(iTotalQtyToBuy, iTotalQtyToSell);
            int iOrderQty = iTotalQtyToBuy - iTotalQtyToSell;


            int iCurrPos = m_PositionManager.getPosition(objRefSymbol).getPosition();
            int iPosDelta = iTotalDesiredPos - iCurrPos;

            // safety check - aggregate of sub-models should tie with algorithm level
            if (iPosDelta != iOrderQty)
                throw Mts::Exception::CMtsException("Error: sub-model and algorithm position mismatch");


            sprintf(szBuffer, "Current agg position: %d Desired agg position: %d Trade required: %d", iCurrPos,
                    iTotalDesiredPos, iPosDelta);
            LogInfo(m_Logger, szBuffer);


            if (iNettableQty != 0) {

                sprintf(szBuffer, "Buy Qty: %d Sell Qty: %d Nettable Qty: %d", iTotalQtyToBuy, iTotalQtyToSell,
                        iNettableQty);
                LogInfo(m_Logger, szBuffer);

                // internalize buy orders then sell orders
                for (int i = 0; i != 2; ++i) {

                    auto &objSameSide = i == 0 ? objBuyers : objSellers;
                    int iTotalQtyOnSide = i == 0 ? iTotalQtyToBuy : iTotalQtyToSell;
                    Mts::Order::COrder::BuySell iDirection =
                            i == 0 ? Mts::Order::COrder::BUY : Mts::Order::COrder::SELL;

                    int iQtyAllocated = 0;

                    // pass 1 - pro-rata, rounding down to prevent overfill
                    for (auto &iter: objSameSide) {

                        int iQtyToFill = static_cast<int>(std::floor(
                                static_cast<double>(iNettableQty) * static_cast<double>(std::get<1>(iter.second)) /
                                static_cast<double>(iTotalQtyOnSide)));

                        std::get<2>(iter.second) += iQtyToFill;

                        iQtyAllocated += iQtyToFill;
                    }

                    // pass 2 - round robin allocate whatever is left
                    while (iQtyAllocated != iNettableQty) {

                        for (auto &iter: objSameSide) {

                            if (std::get<1>(iter.second) != std::get<2>(iter.second) && iQtyAllocated != iNettableQty) {

                                std::get<2>(iter.second) += 1;
                                ++iQtyAllocated;
                            }
                        }
                    }

                    // pass 3 - create internal fill
                    for (auto &iter: objSameSide) {

                        if (std::get<2>(iter.second) != 0) {

                            Mts::Order::COrder objInternalOrder(0, std::get<0>(iter.second)->getModelName().c_str(),
                                                                dtNow, iRefSymbolID, iProviderID, iDirection,
                                                                std::get<2>(iter.second), Mts::Order::COrder::IOC,
                                                                dRefMidPx, "Cross");
                            Mts::Order::COrderFill objInternalFill(objInternalOrder, dtNow, std::get<2>(iter.second),
                                                                   dRefMidPx, 0);

                            std::get<0>(iter.second)->onEvent(objInternalFill);
                            LogInfo(m_Logger, objInternalFill.toString());
                        }
                    }
                }
            }


            if (iOrderQty == 0)
                continue;

            // Because these symbols are in cancel
            if (m_Orders[iRefSymbolID].getQuantity() != 0)
                continue;

            Mts::Order::COrder::BuySell iDirection = iOrderQty > 0 ? Mts::Order::COrder::BUY : Mts::Order::COrder::SELL;
            unsigned int iQuantity = static_cast<unsigned int>(abs(iOrderQty));
            double dPrice =
                    m_iExecAlgoID == ALGO_ICEBERG ? (iDirection == Mts::Order::COrder::BUY ? m_dLastBidPx[iRefSymbolID]
                                                                                           : m_dLastAskPx[iRefSymbolID])
                                                  : (iDirection == Mts::Order::COrder::BUY ? m_dLastAskPx[iRefSymbolID]
                                                                                           : m_dLastBidPx[iRefSymbolID]);
            std::string strExecBrokerCode = "TT";

            Mts::Order::COrderFactory::getInstance().createIOCOrder(
                    dtNow, getAlgoID(), iRefSymbolID, iProviderID, iDirection,
                    iQuantity, dPrice, getAlgoName(), strExecBrokerCode,
                    m_Orders[iRefSymbolID]
            );

            if (isOrderCompliant(m_Orders[iRefSymbolID]) == true) {

                switch (m_iExecAlgoID) {

                    case ALGO_TWAP:
                        objExchange.value().get().submitTWAPOrder(m_Orders[iRefSymbolID]);
                        break;

                    case ALGO_ICEBERG:
                        objExchange.value().get().submitIcebergOrder(m_Orders[iRefSymbolID]);
                        break;

                    default:
                        break;
                }
            }

            LogInfo(m_Logger, m_Orders[iRefSymbolID].toString());
        }
    }
    catch (std::exception &e) {

        LogError(m_Logger, e.what());
    }

    return true;
}


void CAlgorithmTemplate::onEvent(const Mts::Order::COrderStatus &objOrderStatus) {

    boost::mutex::scoped_lock objScopedLock(m_Mutex);

    LogInfo(m_Logger, objOrderStatus.toString());

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrderStatus.getOrigOrder().getSymbolID());

    if (objOrderStatus.getStatus() == Mts::Order::COrder::FILLED ||
        objOrderStatus.getStatus() == Mts::Order::COrder::CANCELLED ||
        //objOrderStatus.getStatus() == Mts::Order::COrder::REJECTED ||
        objOrderStatus.getStatus() == Mts::Order::COrder::LIMIT_VIOLATION) {

        m_Orders[objSymbol.getSymbolID()].setQuantity(0);
        m_CancelReqs[objSymbol.getSymbolID()].setMtsCancelReqID("");
    }

    if (objOrderStatus.getStatus() == Mts::Order::COrder::REJECTED) {

        LogError(m_Logger, "ORDER CANCEL REJECTED");
    }

    if (objOrderStatus.getStatus() == Mts::Order::COrder::CANCELLED) {

        LogInfo(m_Logger, "ORDER CANCELLED");
    }
}


void CAlgorithmTemplate::allocateFillToModels(const Mts::Order::COrderFill &objOrderFill) {

    unsigned int iSymbolID = objOrderFill.getOrigOrder().getSymbolID();

    // at this point any internalization will have been done so if the actual and desired positions differ, we should allocate a fill
    boost::unordered_map<std::string, std::tuple<boost::shared_ptr<Mts::Model::CModel>, int, int>> objSameSide;

    unsigned int iTotalQtyOnSide = 0;

    for (auto ptrModel: m_Models) {

        const std::string &strModel = ptrModel->getModelName();
        int iModelCurrPos = ptrModel->getCurrPosition(iSymbolID);
        int iModelDesiredPos = ptrModel->getDesiredPosition(iSymbolID);
        int iModelOrderQty = iModelDesiredPos - iModelCurrPos;

        char szBuffer[512];
        sprintf(szBuffer, "Alloc Model: %s Current position: %d Desired position: %d", strModel.c_str(), iModelCurrPos,
                iModelDesiredPos);
        LogInfo(m_Logger, szBuffer);

        // because we assue all models have open orders at the same side, otherwise, internalize
        if ((iModelOrderQty > 0 && objOrderFill.getOrigOrder().getDirection() == Mts::Order::COrder::SELL) ||
            (iModelOrderQty < 0 && objOrderFill.getOrigOrder().getDirection() == Mts::Order::COrder::BUY))
            throw Mts::Exception::CMtsException("Error: Internalization failed");

        objSameSide.insert(std::pair<std::string, std::tuple<boost::shared_ptr<Mts::Model::CModel>, int, int>>(
                ptrModel->getModelName(), std::make_tuple(ptrModel, abs(iModelOrderQty), 0)));
        iTotalQtyOnSide += abs(iModelOrderQty);
    }

    if (objSameSide.size() == 0) {
        LogInfo(m_Logger, "No models are defined.  Fill is from external (manual) orders, not allocated.");
        return;
    }

    unsigned int wantSize = objOrderFill.getFillQuantity();
    if (iTotalQtyOnSide < objOrderFill.getFillQuantity()) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Over Fill: models ordered %d, filled %d.  All are allocated to models.\n",
                 (int) iTotalQtyOnSide, (int) objOrderFill.getFillQuantity());
        LogInfo(m_Logger, std::string(buf));
        wantSize = iTotalQtyOnSide;
    }

    int iQtyAllocated = 0;

    // allocate all wantSize
    while (iQtyAllocated != wantSize) {
        for (auto &iter: objSameSide) {
            if (std::get<1>(iter.second) != std::get<2>(iter.second) && iQtyAllocated != wantSize) {
                std::get<2>(iter.second) += 1;
                ++iQtyAllocated;
            }
        }
    }

    // allocate overfills round robin
    while (iQtyAllocated < (int) objOrderFill.getFillQuantity()) {
        for (auto &iter: objSameSide) {
            if (iQtyAllocated != objOrderFill.getFillQuantity()) {
                std::get<2>(iter.second) += 1;
                ++iQtyAllocated;
            }
        }
    }

    // create internal fill - they should all be in the same direction
    // as the order
    for (auto &iter: objSameSide) {

        if (std::get<2>(iter.second) != 0) {

            Mts::Order::COrder objInternalOrder(0, std::get<0>(iter.second)->getModelName().c_str(),
                                                objOrderFill.getFillTimestamp(), iSymbolID,
                                                objOrderFill.getOrigOrder().getProviderID(), objOrderFill.getBuySell(),
                                                std::get<2>(iter.second), Mts::Order::COrder::IOC,
                                                objOrderFill.getFillPrice(), "Pro-rata");
            Mts::Order::COrderFill objInternalFill(objInternalOrder, objOrderFill.getFillTimestamp(),
                                                   (unsigned int) std::get<2>(iter.second), objOrderFill.getFillPrice(),
                                                   Mts::Core::CDateTime(0.0));

            std::get<0>(iter.second)->onEvent(objInternalFill);
            LogInfo(m_Logger, objInternalFill.toString());
        }
    }
}


void CAlgorithmTemplate::onEvent(const Mts::Order::COrderFill &objOrderFill) {

    boost::mutex::scoped_lock objScopedLock(m_Mutex);

    const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrderFill.getOrigOrder().getSymbolID());

    CAlgorithm::onEvent(objOrderFill);
    LogInfo(m_Logger, objOrderFill.toString());

    // do not allocate to models if a recovery phase fill, models restart with zero position state
    if (objOrderFill.isRecoveryFill() == false) {

        allocateFillToModels(objOrderFill);
    } else {
        LogInfo(m_Logger, "Processing recovery fill");
    }

    if (objOrderFill.getOrigOrder().getOrderState() == Mts::Order::COrder::FILLED) {

        m_Orders[objSymbol.getSymbolID()].setQuantity(0);
        m_CancelReqs[objSymbol.getSymbolID()].setMtsCancelReqID("");
        LogInfo(m_Logger, "ORDER COMPLETE");
    }

    // broadcast updated position and P&L
    const Mts::Accounting::CPosition &objPosition = m_PositionManager.getPosition(objSymbol);

    LogInfo(m_Logger, objPosition.toString());
    broadcastPosition(objSymbol);

    char szBuffer[255];

    for (auto ptrModel: m_Models) {

        sprintf(szBuffer, "%llu %s PnL %.0f Fills %u", objOrderFill.getFillTimestamp().getCMTime(),
                ptrModel->getModelName().c_str(), ptrModel->getPnLUSD(), ptrModel->getNumFills());
        LogInfo(m_Logger, szBuffer);
    }

    double dUSDPnL = getPnLUSD();

    sprintf(szBuffer, "%llu Port PnL %.0f", objOrderFill.getFillTimestamp().getCMTime(), dUSDPnL);
    LogInfo(m_Logger, szBuffer);
}

void CAlgorithmTemplate::onEvent(const Mts::OrderBook::CManualCommand &objMC) {
    const std::string strUserCommand(objMC.getCommand());
    CAlgorithmCommand::runCommand(strUserCommand, *this, m_Logger);
}


void CAlgorithmTemplate::onEndOfDay(Mts::Core::CDateTime dtTimestamp) {

}


void CAlgorithmTemplate::onStop() {

    CAlgorithm::onStop();
}


void CAlgorithmTemplate::onDestroy() {

    CAlgorithm::onDestroy();
}


std::vector<std::string> CAlgorithmTemplate::getMessageHistory() const {

    std::vector<std::string> messageHistory;

    return messageHistory;
}


void CAlgorithmTemplate::addTradedInstrument(const Mts::Core::CSymbol &objSymbol) {

    CAlgorithm::addTradedInstrument(objSymbol);
}


void CAlgorithmTemplate::addModel(boost::shared_ptr<Mts::Model::CModel> ptrModel) {

    m_Models.push_back(ptrModel);
}


void CAlgorithmTemplate::setTradeTime(unsigned int iTimeOfDayMins) {

    m_iTradeTimeMap[iTimeOfDayMins] = 1;
}

bool CAlgorithmTemplate::sendOrder(
        bool isBuy,
        const std::string &symbol,
        long long qty,
        const std::string &priceStr) {
    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider("TT");
    unsigned int iProviderID = objProvider.getProviderID();
    Mts::Core::CDateTime dtNow = Mts::Core::CDateTime::now();
    Mts::Order::COrder::BuySell iDirection = isBuy ? Mts::Order::COrder::BUY : Mts::Order::COrder::SELL;
    std::string strExecBrokerCode = "TT";
    boost::optional<std::reference_wrapper<Mts::Exchange::CExchange> > objExchange = Mts::Exchange::CExchangeBroker::getInstance().getExchange(
            iProviderID);

    for (auto &objRefSymbol: m_TradedInstruments) {
        if (objRefSymbol.getSymbol() != symbol) {
            continue;
        }
        unsigned int iRefSymbolID = objRefSymbol.getSymbolID();

        // check if we have working order 
        if (m_Orders[iRefSymbolID].getQuantity() != 0) {
            LogError(m_Logger, std::string("Working order found: ") + m_Orders[iRefSymbolID].toString() +
                               " Please wait or cancel before sending new!");
            return false;
        }

        // parse the price str
        double dPx = 0.0;
        bool bMarketOrder = false;
        double bidPx = m_dLastBidPx[iRefSymbolID], askPx = m_dLastAskPx[iRefSymbolID];
        double midPx = (bidPx + askPx) / 2.0;
        double tickSz = objRefSymbol.getTickSize();
        try {
            if (!CAlgorithmCommand::parsePriceStr(priceStr, bidPx, askPx,
                                                  tickSz, dPx, bMarketOrder)) {
                LogError(m_Logger, std::string("problem parsing the price string ") + priceStr);
                return false;
            }
        }
        catch (const std::exception &e) {
            LogError(m_Logger, std::string("problem parsing the price string ") + priceStr + " : " + e.what());
            return false;
        }
        // getting current position
        int iCurrPos = m_PositionManager.getPosition(objRefSymbol).getPosition();

        // make sure we set the price for market order
        if (bMarketOrder) dPx = midPx;
        m_Orders[iRefSymbolID] = Mts::Order::COrderFactory::getInstance().createGTCOrder(
                dtNow, "clOrdId", getAlgoID(), iRefSymbolID, iProviderID, iDirection,
                (unsigned int) qty, dPx, getAlgoName(), strExecBrokerCode
        );
        Mts::Order::COrder &order = m_Orders[iRefSymbolID];

        // check compliance
        bool bSent = false;
        if (isOrderCompliant(order) == true) {
            if (bMarketOrder) {
                // send market order
                bSent = objExchange.value().get().submitMktOrder(order);
            } else {
                // send limit order, GTC
                bSent = objExchange.value().get().submitLmtOrder(order, Mts::Exchange::CExchange::TimeInForce::GTC);
            }
        }
        if (bSent)
            LogInfo(m_Logger, std::string("Order sent: ") + order.toString());
        else
            LogError(m_Logger, std::string("Failed to send order: ") + order.toString());
        return bSent;
    }
    LogError(m_Logger, std::string("Failed to send order: ") + symbol + std::string(" not found!"));
    return false;
};

bool CAlgorithmTemplate::cancelOrder(const std::string &symbol) {
    const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider("TT");
    unsigned int iProviderID = objProvider.getProviderID();
    Mts::Core::CDateTime dtNow = Mts::Core::CDateTime::now();
    std::string strExecBrokerCode = "TT";
    boost::optional<std::reference_wrapper<Mts::Exchange::CExchange> > objExchange = Mts::Exchange::CExchangeBroker::getInstance().getExchange(
            iProviderID);

    for (auto &objRefSymbol: m_TradedInstruments) {
        if (objRefSymbol.getSymbol() != symbol) {
            continue;
        }
        unsigned int iRefSymbolID = objRefSymbol.getSymbolID();
        // check if we have working order 
        if (m_Orders[iRefSymbolID].getQuantity() != 0) {
            if (strlen(m_CancelReqs[iRefSymbolID].getMtsCancelReqID()) == 0) {
                m_CancelReqs[iRefSymbolID] = Mts::Order::COrderFactory::getInstance().createCancellationRequest(
                        dtNow, getAlgoID(), m_Orders[iRefSymbolID]);
                bool bSent = objExchange.value().get().cancelOrder(m_CancelReqs[iRefSymbolID]);
                if (bSent) {
                    LogInfo(m_Logger, std::string("CANCEL REQUESTED: ") + m_CancelReqs[iRefSymbolID].toString());
                } else {
                    LogError(m_Logger,
                             std::string("Failed to send cancel request: ") + m_CancelReqs[iRefSymbolID].toString());
                }
                return bSent;
            } else {
                LogInfo(m_Logger,
                        std::string("Found pending cancel, please wait for: ") + m_CancelReqs[iRefSymbolID].toString());
            }
        } else {
            LogInfo(m_Logger, std::string("No quantity to be canceled for ") + symbol);
        }
        return false;
    }
    LogError(m_Logger, std::string("Symbol not found: ") + symbol);
    return false;
};

std::string CAlgorithmTemplate::getBidAsk(const std::string &symbol) const {
    for (const auto &objRefSymbol: m_TradedInstruments) {
        if (objRefSymbol.getSymbol() == symbol) {
            unsigned int iRefSymbolID = objRefSymbol.getSymbolID();
            double bidPx = m_dLastBidPx[iRefSymbolID], askPx = m_dLastAskPx[iRefSymbolID];
            double tickSz = objRefSymbol.getTickSize();
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: [%.7lf, %.7lf] ticksize: %.7lf", symbol.c_str(), bidPx, askPx, tickSz);
            return std::string(buf);
        }
    }
    return std::string("cannot find symbol ") + symbol;
}

std::string CAlgorithmTemplate::getPosition(const std::string &symbol) const {
    for (const auto &objRefSymbol: m_TradedInstruments) {
        if (objRefSymbol.getSymbol() == symbol) {
            return dynamic_cast<const CAlgorithm *>(this)->getPosition(objRefSymbol).toString();
        }
    }
    return std::string("cannot find symbol ") + symbol;
};

std::string CAlgorithmTemplate::getOpenOrders(const std::string &symbol) const {
    for (const auto &objRefSymbol: m_TradedInstruments) {
        if (objRefSymbol.getSymbol() == symbol) {
            unsigned int iRefSymbolID = objRefSymbol.getSymbolID();
            if (m_Orders[iRefSymbolID].getQuantity() == 0) {
                return std::string("No Open Orders Found for ") + symbol;
            }
            return m_Orders[iRefSymbolID].toString();
        }
    }
    return std::string("cannot find symbol ") + symbol;
}
