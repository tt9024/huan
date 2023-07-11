#include "boost/lexical_cast.hpp"
#include "CMath.h"
#include "quickfix/FileStore.h"
#include "quickfix/FileLog.h"
#include "quickfix/SocketInitiator.h"
#include "quickfix/SessionSettings.h"
#include "CExchangeFIX_TT.h"
#include "CSymbol.h"
#include "COrderFactory.h"
#include "CMtsException.h"
#include "CApplicationLog.h"

#include "symbol_map.h"
#include "RiskMonitor.h"

using namespace Mts::Exchange;


#pragma warning( disable : 4503 4355 4786 )


CExchangeFIX_TT::CExchangeFIX_TT(const Mts::Core::CProvider &objProvider,
        const std::string &strSenderCompID,
        const std::string &strTargetCompID,
        const std::string &strUsername,
        const std::string &strPassword,
        const std::string &strAccount,
        unsigned int iHeartbeatsecs)
        : CExchange(true),
          m_Provider(objProvider),
          m_SenderCompID(strSenderCompID),
          m_TargetCompID(strTargetCompID),
          m_strUsername(strUsername),
          m_strPassword(strPassword),
          m_strAccount(strAccount),
          m_iHeartbeatsecs(iHeartbeatsecs),
          m_bRecoveryComplete(false) {

    m_strTestReqID = "Mts";
}


bool CExchangeFIX_TT::connect() {

    return true;
}


bool CExchangeFIX_TT::disconnect() {

    FIX44::Logout objMessage;

    queryHeader(objMessage.getHeader());
    FIX::Session::sendToTarget(objMessage, getSessionQualifier());

    return true;
}


bool CExchangeFIX_TT::submitMktOrder(const Mts::Order::COrder &objOrder) {

    try {

        bool bApproved = CExchange::submitMktOrder(objOrder);

        if (bApproved == false) {
            return false;
        }

        boost::mutex::scoped_lock objScopedLock(m_Mutex);

        FIX44::NewOrderSingle objNewOrder;

        FIX::ClOrdID objClOrdID(objOrder.getMtsOrderID());
        objNewOrder.setField(objClOrdID);

        FIX::UtcTimeStamp dtTransactTime(
                objOrder.getCreateTimestamp().getHour(),
                objOrder.getCreateTimestamp().getMin(),
                objOrder.getCreateTimestamp().getSec(),
                objOrder.getCreateTimestamp().getMSec()
        );
        FIX::TransactTime objTransactTime(dtTransactTime);
        objNewOrder.setField(objTransactTime);

        // instrument component
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID());
        const std::string &strTTSecID = objSymbol.getTTSecID();
        const std::string &strExchSymbol = objSymbol.getExchSymbol();
        const std::string &strExchange = objSymbol.getExchange();

        objNewOrder.setField(FIX::SecurityID(strTTSecID));
        objNewOrder.setField(FIX::IDSource("96"));
        objNewOrder.setField(FIX::Symbol(strExchSymbol));
        objNewOrder.setField(FIX::SecurityExchange(strExchange));

        FIX::OrderQty objOrderQty(objOrder.getQuantity());
        objNewOrder.setField(objOrderQty);

        char iBuySell = objOrder.getDirection() == Mts::Order::COrder::BUY ? '1' : '2';
        FIX::Side objSide(iBuySell);
        objNewOrder.setField(objSide);

        // market order
        FIX::OrdType objOrdType('1');
        objNewOrder.setField(objOrdType);

        // trader component
        FIX::Account objAccount(getStrAccount(objOrder.getOrderTag()).c_str());
        objNewOrder.setField(objAccount);

        // Simplified Execution Source Code 1031
        // Y - Electronic (Default)
        objNewOrder.setField(FIX::CustOrderHandlingInst("Y"));

        // set the algo tag for TT's 16558
        objNewOrder.setField(FIX::StringField(16558, objOrder.getOrderTag()));

        // cancel on disconnect option
        objNewOrder.setField(FIX::ExecInst("o 2"));

        // TimeInForce defaults to be '0' Day
        objNewOrder.setField(FIX::TimeInForce('0'));

        queryHeader(objNewOrder.getHeader(), true);


        FIX::Session::sendToTarget(objNewOrder, getSessionQualifier());

        //publishOrderNew(objOrder);

        return true;
    }
    catch (std::exception &e) {
        AppError("submitIOCOrder Error: %s", e.what());
        return false;
    }
}


// TODO - simply it to be using only algo-symbol-qty-px
// use symbol map to retrieve symbol information
bool CExchangeFIX_TT::submitLmtOrder(const Mts::Order::COrder &objOrder,
        Mts::Exchange::CExchange::TimeInForce iTIF) {
    try {
        bool bApproved = CExchange::submitLmtOrder(objOrder, iTIF);
        if (bApproved == false) {
            return false;
        }

        boost::mutex::scoped_lock objScopedLock(m_Mutex);

        FIX44::NewOrderSingle objNewOrder;
        FIX::ClOrdID objClOrdID(objOrder.getMtsOrderID());
        objNewOrder.setField(objClOrdID);

        FIX::UtcTimeStamp dtTransactTime(objOrder.getCreateTimestamp().getHour(),
                                         objOrder.getCreateTimestamp().getMin(), objOrder.getCreateTimestamp().getSec(),
                                         objOrder.getCreateTimestamp().getMSec());
        FIX::TransactTime objTransactTime(dtTransactTime);
        objNewOrder.setField(objTransactTime);

        // instrument component
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID());

        const std::string &strTTSecID = objSymbol.getTTSecID();
        const std::string &strExchSymbol = objSymbol.getExchSymbol();
        const std::string &strExchange = objSymbol.getExchange();

        objNewOrder.setField(FIX::SecurityID(strTTSecID));
        objNewOrder.setField(FIX::IDSource("96"));
        objNewOrder.setField(FIX::Symbol(strExchSymbol));
        objNewOrder.setField(FIX::SecurityExchange(strExchange));

        FIX::Price objOrderPx(objOrder.getPrice() / objSymbol.getTTScaleMultiplier());
        objNewOrder.setField(objOrderPx);

        FIX::OrderQty objOrderQty(objOrder.getQuantity());
        objNewOrder.setField(objOrderQty);

        char iBuySell = objOrder.getDirection() == Mts::Order::COrder::BUY ? '1' : '2';
        FIX::Side objSide(iBuySell);
        objNewOrder.setField(objSide);

        // '2' limit order
        FIX::OrdType objOrdType('2');
        objNewOrder.setField(objOrdType);

        // '0' is Day, '1' is GTC
        char iTimeInForce = iTIF == Mts::Exchange::CExchange::GTC ? '1' : '0';
        FIX::TimeInForce objTimeInForce(iTimeInForce);

        // trader component
        FIX::Account objAccount(getStrAccount(objOrder.getOrderTag()).c_str());
        objNewOrder.setField(objAccount);

        // Simplified Execution Source Code 1031
        // Y - Electronic (Default)
        objNewOrder.setField(FIX::CustOrderHandlingInst("Y"));

        // set the algo tag for text TT field 16558
        objNewOrder.setField(FIX::StringField(16558, objOrder.getOrderTag()));

        // cancel on disconnect option
        objNewOrder.setField(FIX::ExecInst("o 2"));

        queryHeader(objNewOrder.getHeader(), true);

        FIX::Session::sendToTarget(objNewOrder, getSessionQualifier());

        //publishOrderNew(objOrder);

        return true;
    }
    catch (std::exception &e) {
        AppError("submitGTCOrder error: %s", e.what());
        return false;
    }
}


bool CExchangeFIX_TT::submitTWAPOrder(const Mts::Order::COrder &objOrder) {

    try {

        bool bApproved = CExchange::submitTWAPOrder(objOrder);

        if (bApproved == false) {
            return false;
        }

        boost::mutex::scoped_lock objScopedLock(m_Mutex);

        FIX44::NewOrderSingle objNewOrder;

        FIX::ClOrdID objClOrdID(objOrder.getMtsOrderID());
        objNewOrder.setField(objClOrdID);

        FIX::UtcTimeStamp dtTransactTime(objOrder.getCreateTimestamp().getHour(),
                                         objOrder.getCreateTimestamp().getMin(), objOrder.getCreateTimestamp().getSec(),
                                         objOrder.getCreateTimestamp().getMSec());
        FIX::TransactTime objTransactTime(dtTransactTime);
        objNewOrder.setField(objTransactTime);

        // instrument component
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID());
        //const std::string &                   strSecurityType     = objSymbol.getSecurityType();
        //const std::string &                   strExchange             = objSymbol.getExchange();
        //const std::string &                   strExchSymbol           = objSymbol.getExchSymbol();
        //const std::string &                   strContractTicker   = Mts::Core::CSymbol::getSymbol2InstDictionary().getCalendarFromMtsSymbol(objSymbol.getSymbolID()).ContractTicker;

        // TWAP parameters
        unsigned int iLotsPerMin = objSymbol.getLotsPerMin();
        unsigned int iTimeSliceSecs = objSymbol.getTimeSliceSecs();
        unsigned int iLotsPerSlice = static_cast<unsigned int>(iTimeSliceSecs * static_cast<double>(iLotsPerMin) /
                                                               60.0);
        unsigned int iTWAPDurationInMin = static_cast<unsigned int>(ceil(
                static_cast<double>(objOrder.getQuantity()) / static_cast<double>(iLotsPerMin)));

        //objNewOrder.setField(FIX::SecurityType(strSecurityType));
        //objNewOrder.setField(FIX::Symbol(strExchSymbol));
        //objNewOrder.setField(FIX::SecurityExchange(strExchange));
        //objNewOrder.setField(FIX::MaturityMonthYear(strContractTicker));

        const std::string &strTTSecID = objSymbol.getTTSecID();
        const std::string &strExchSymbol = objSymbol.getExchSymbol();
        const std::string &strExchange = objSymbol.getExchange();

        objNewOrder.setField(FIX::SecurityID(strTTSecID));
        objNewOrder.setField(FIX::IDSource("96"));
        objNewOrder.setField(FIX::Symbol(strExchSymbol));
        objNewOrder.setField(FIX::SecurityExchange(strExchange));

        // algos defined as 'limit' orders
        FIX::OrdType objOrdType('2');
        objNewOrder.setField(objOrdType);

        char iBuySell = objOrder.getDirection() == Mts::Order::COrder::BUY ? '1' : '2';
        FIX::Side objSide(iBuySell);
        objNewOrder.setField(objSide);

        FIX::OrderQty objOrderQty(objOrder.getQuantity());
        objNewOrder.setField(objOrderQty);

        // reference price
        FIX::Price objPrice(objOrder.getPrice() / objSymbol.getTTScaleMultiplier());
        objNewOrder.setField(objPrice);

        // time in force set to GTC for algos
        FIX::TimeInForce objTimeInForce('1');
        objNewOrder.setField(objTimeInForce);

        // TWAP parameters
        objNewOrder.setField(16847, "TT_Time_Sliced");

        // strategy target 0 = custom algo, 1 = TT algo, 3 = bank algo
        objNewOrder.setField(16848, "1");

        // ETimeAct - action taken at end time, always 1 - cancel unfilled balance
        objNewOrder.setField(16906, "1");

        // time in force set to GTD for child orders
        objNewOrder.setField(16903, "1");

        // disclose value
        objNewOrder.setField(16904, std::to_string(iLotsPerSlice));

        // disclose value type
        objNewOrder.setField(16905, "1");

        // timeslice in ms
        objNewOrder.setField(16907, std::to_string(iTimeSliceSecs * 1000));

        // limit price type
        std::string strLimitPxType = objOrder.getDirection() == Mts::Order::COrder::BUY ? "1" : "2";
        objNewOrder.setField(16911, strLimitPxType);

        // limit ticks away
        objNewOrder.setField(16912, "0");

        // leftover action (at end of timeslice) 0 = leave, 1 = payup
        objNewOrder.setField(16909, "1");

        // leftover ticks
        objNewOrder.setField(16910, "50");

        // execution start time
        Mts::Core::CDateTime dtStartTime = Mts::Core::CDateTime::nowUTC();
        FIX::UtcTimeStamp objStartTime(dtStartTime.getHour(), dtStartTime.getMin(), dtStartTime.getSec(),
                                       dtStartTime.getMSec());
        FIX::EffectiveTime objEffectiveTime(objStartTime);
        objNewOrder.setField(objEffectiveTime);

        // execution end time
        Mts::Core::CDateTime dtEndTime = dtStartTime;
        dtEndTime.addMinutes(iTWAPDurationInMin);
        FIX::UtcTimeStamp objEndTime(dtEndTime.getHour(), dtEndTime.getMin(), dtEndTime.getSec(), dtEndTime.getMSec());
        FIX::ExpireTime objExpireTime(objEndTime);
        objNewOrder.setField(objExpireTime);

        // trader component
        FIX::Account objAccount(getStrAccount(objOrder.getOrderTag()).c_str());
        objNewOrder.setField(objAccount);

        // Simplified Execution Source Code 1031
        // Y - Electronic (Default)
        objNewOrder.setField(FIX::CustOrderHandlingInst("Y"));

        // set the algo tag for text TT field 16558
        objNewOrder.setField(FIX::StringField(16558, objOrder.getOrderTag()));

        // cancel on disconnect option
        objNewOrder.setField(FIX::ExecInst("o 2"));

        queryHeader(objNewOrder.getHeader(), true);

        FIX::Session::sendToTarget(objNewOrder, getSessionQualifier());

        //publishOrderNew(objOrder);

        return true;
    }
    catch (std::exception &e) {

        AppLogError(e.what());

        return false;
    }
}


bool CExchangeFIX_TT::submitIcebergOrder(const Mts::Order::COrder &objOrder) {

    try {

        bool bApproved = CExchange::submitIcebergOrder(objOrder);

        if (bApproved == false) {
            return false;
        }

        boost::mutex::scoped_lock objScopedLock(m_Mutex);

        FIX44::NewOrderSingle objNewOrder;

        FIX::ClOrdID objClOrdID(objOrder.getMtsOrderID());
        objNewOrder.setField(objClOrdID);

        FIX::UtcTimeStamp dtTransactTime(objOrder.getCreateTimestamp().getHour(),
                                         objOrder.getCreateTimestamp().getMin(), objOrder.getCreateTimestamp().getSec(),
                                         objOrder.getCreateTimestamp().getMSec());
        FIX::TransactTime objTransactTime(dtTransactTime);
        objNewOrder.setField(objTransactTime);

        // instrument component
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(objOrder.getSymbolID());
        //const std::string &                   strSecurityType     = objSymbol.getSecurityType();
        //const std::string &                   strExchange             = objSymbol.getExchange();
        //const std::string &                   strExchSymbol           = objSymbol.getExchSymbol();
        //const std::string &                   strContractTicker   = Mts::Core::CSymbol::getSymbol2InstDictionary().getCalendarFromMtsSymbol(objSymbol.getSymbolID()).ContractTicker;

        // Iceberg parameters
        std::string strDiscVal = boost::lexical_cast<std::string>(objSymbol.getDiscVal());
        std::string strVariance = boost::lexical_cast<std::string>(objSymbol.getVariance());
        std::string strLimitTicksAway = boost::lexical_cast<std::string>(objSymbol.getLimitTicksAway());

        //objNewOrder.setField(FIX::SecurityType(strSecurityType));
        //objNewOrder.setField(FIX::Symbol(strExchSymbol));
        //objNewOrder.setField(FIX::SecurityExchange(strExchange));
        //objNewOrder.setField(FIX::MaturityMonthYear(strContractTicker));

        const std::string &strTTSecID = objSymbol.getTTSecID();
        const std::string &strExchSymbol = objSymbol.getExchSymbol();
        const std::string &strExchange = objSymbol.getExchange();

        objNewOrder.setField(FIX::SecurityID(strTTSecID));
        objNewOrder.setField(FIX::IDSource("96"));
        objNewOrder.setField(FIX::Symbol(strExchSymbol));
        objNewOrder.setField(FIX::SecurityExchange(strExchange));

        // algos defined as 'limit' orders
        FIX::OrdType objOrdType('2');
        objNewOrder.setField(objOrdType);

        char iBuySell = objOrder.getDirection() == Mts::Order::COrder::BUY ? '1' : '2';
        FIX::Side objSide(iBuySell);
        objNewOrder.setField(objSide);

        FIX::OrderQty objOrderQty(objOrder.getQuantity());
        objNewOrder.setField(objOrderQty);

        // reference price
        FIX::Price objPrice(objOrder.getPrice() / objSymbol.getTTScaleMultiplier());
        objNewOrder.setField(objPrice);

        // time in force set to GTC for algos
        FIX::TimeInForce objTimeInForce('1');
        objNewOrder.setField(objTimeInForce);

        // Iceberg parameters
        FIX::Group objGroup(957, 958);

        std::string strLimitPriceType = objOrder.getDirection() == Mts::Order::COrder::BUY ? "1" : "2";
        objGroup.setField(958, "LimitPriceType");
        objGroup.setField(959, "1");
        objGroup.setField(960, strLimitPriceType);
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "LimitTicksAway");
        objGroup.setField(959, "1");
        objGroup.setField(960, strLimitTicksAway);
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "AutoResubExpiredGTD");
        objGroup.setField(959, "13");
        objGroup.setField(960, "N");
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "Variance");
        objGroup.setField(959, "1");
        objGroup.setField(960, strVariance);
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "DiscValType");
        objGroup.setField(959, "1");
        objGroup.setField(960, "1");
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "DiscVal");
        objGroup.setField(959, "7");
        objGroup.setField(960, strDiscVal);
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "ParentTIF");
        objGroup.setField(959, "1");
        objGroup.setField(960, "2");
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "ChildTIF");
        objGroup.setField(959, "1");
        objGroup.setField(960, "2");
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "user_disconnect_action");
        objGroup.setField(959, "1");
        objGroup.setField(960, "0");
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "Algo Type");
        objGroup.setField(959, "1");
        objGroup.setField(960, "1");
        objNewOrder.addGroup(objGroup);

        objGroup.setField(958, "Algo Name");
        objGroup.setField(959, "14");
        objGroup.setField(960, "TT_Iceberg");
        objNewOrder.addGroup(objGroup);

        // trader component
        FIX::Account objAccount(getStrAccount(objOrder.getOrderTag()).c_str());
        objNewOrder.setField(objAccount);

        // Simplified Execution Source Code 1031
        // Y - Electronic (Default)
        objNewOrder.setField(FIX::CustOrderHandlingInst("Y"));

        // set the algo tag for text TT field 16558
        objNewOrder.setField(FIX::StringField(16558, objOrder.getOrderTag()));

        // cancel on disconnect option
        objNewOrder.setField(FIX::ExecInst("o 2"));

        queryHeader(objNewOrder.getHeader());

        FIX::Session::sendToTarget(objNewOrder, getSessionQualifier());

        //publishOrderNew(objOrder);

        return true;
    }
    catch (std::exception &e) {

        AppLogError(e.what());

        return false;
    }
}


bool CExchangeFIX_TT::cancelOrder(const Mts::Order::COrderCancelRequest &objCancelReq) {

    try {

        boost::mutex::scoped_lock objScopedLock(m_Mutex);

        FIX44::OrderCancelRequest objOrderCancelRequest;

        FIX::ClOrdID objClOrdID(objCancelReq.getMtsCancelReqID());
        objOrderCancelRequest.setField(objClOrdID);

        FIX::OrigClOrdID objOrigClOrdID(objCancelReq.getOrigOrder().getMtsOrderID());
        objOrderCancelRequest.setField(objOrigClOrdID);

        // set the algo tag for text TT field 16558
        objOrderCancelRequest.setField(FIX::StringField(16558, objCancelReq.getOrigOrder().getOrderTag()));

        queryHeader(objOrderCancelRequest.getHeader(), true);

        FIX::Session::sendToTarget(objOrderCancelRequest, getSessionQualifier());

        return true;
    }
    catch (std::exception &e) {

        AppError("submitAlgoOrder error: %s", e.what());

        return false;
    }
}


bool CExchangeFIX_TT::cancelOrder(const std::string &origClOrdId, const std::string &algo) {
    try {

        boost::mutex::scoped_lock objScopedLock(m_Mutex);

        FIX44::OrderCancelRequest objOrderCancelRequest;

        FIX::ClOrdID objClOrdID("C" + std::to_string(utils::TimeUtil::cur_micro()) + "_" + algo);
        objOrderCancelRequest.setField(objClOrdID);

        FIX::OrigClOrdID objOrigClOrdID(origClOrdId);
        objOrderCancelRequest.setField(objOrigClOrdID);

        // set the algo tag for text TT field 16558
        objOrderCancelRequest.setField(FIX::StringField(16558, algo));
        queryHeader(objOrderCancelRequest.getHeader(), true);
        FIX::Session::sendToTarget(objOrderCancelRequest, getSessionQualifier());
        return true;
    }
    catch (std::exception &e) {
        AppError("submitAlgoOrder error: %s", e.what());
        return false;
    }
}

bool CExchangeFIX_TT::replaceOrder(const std::string &origClOrdId, int64_t qty, double px, const std::string &algo,
        const std::string &symbol, const std::string &newClOrdId) {
    if (qty == 0) {
        return cancelOrder(origClOrdId, algo);
    }
    try {
        boost::mutex::scoped_lock objScopedLock(m_Mutex);
        FIX44::OrderCancelReplaceRequest crRequest;

        // trader component
        FIX::Account objAccount(getStrAccount(algo).c_str());

        crRequest.setField(objAccount);

        // clOrdId
        FIX::ClOrdID objClOrdID(newClOrdId);
        crRequest.setField(objClOrdID);
        FIX::OrigClOrdID objOrigClOrdID(origClOrdId);
        crRequest.setField(objOrigClOrdID);

        const auto *tradable(utils::SymbolMapReader::get().getByTradable(symbol, false));
        /*
        // instrument component
        const std::string&                 strTTSecID = tradable->_tt_security_id;
        const std::string&                 strExchSymbol = tradable->_exch_symbol;
        const std::string&                 strExchange = tradable->_venue;

        crRequest.setField(FIX::SecurityID(strTTSecID));
        crRequest.setField(FIX::IDSource("96"));
        crRequest.setField(FIX::Symbol(strExchSymbol));
        crRequest.setField(FIX::SecurityExchange(strExchange));
        */

        FIX::Price objOrderPx(px / tradable->_px_multiplier);
        crRequest.setField(objOrderPx);

        char side = '1';
        if (qty < 0) {
            side = '2';
            qty = -qty;
        }
        FIX::Side objSide(side);
        crRequest.setField(objSide);
        FIX::OrderQty objOrderQty(qty);
        crRequest.setField(objOrderQty);


        // '2' limit order
        FIX::OrdType objOrdType('2');
        crRequest.setField(objOrdType);
        /*
        // '0' is Day, '1' is GTC
        char iTimeInForce = iTIF == Mts::Exchange::CExchange::GTC ? '1' : '0';
        FIX::TimeInForce objTimeInForce(iTimeInForce);

        // Simplified Execution Source Code 1031
        // Y - Electronic (Default)
        objNewOrder.setField(FIX::CustOrderHandlingInst("Y"));

        // cancel on disconnect option
        objNewOrder.setField(FIX::ExecInst("o 2"));
        */

        // set the algo tag for text TT field 16558
        crRequest.setField(FIX::StringField(16558, algo));
        queryHeader(crRequest.getHeader(), true);
        FIX::Session::sendToTarget(crRequest, getSessionQualifier());
        return true;
    }
    catch (std::exception &e) {
        AppError("submitReplace error: %s", e.what());
        return false;
    }
}

/*
bool CExchangeFIX_TT::replaceOrderPxOnly(const std::string& origClOrdId, const std::string& algo, double px) {
}

bool CExchangeFIX_TT::replaceOrderQtyOnly(const std::string& origClOrdId, const std::string& algo, unsigned int qty) {
}
*/

void CExchangeFIX_TT::onCreate(const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onLogon(const FIX::SessionID &objSessionID) {

    publishLogon(getProviderID());
    std::cout << std::endl << "Logon - " << objSessionID << std::endl;
    AppLog(std::string("logged on"));
}


void CExchangeFIX_TT::onLogout(const FIX::SessionID &objSessionID) {

    publishLogout(getProviderID());
    std::cout << std::endl << "Logout - " << objSessionID << std::endl;
    AppLog(std::string("logged out"));
}


// incoming, ECN -> Mts
void CExchangeFIX_TT::fromAdmin(const FIX::Message &objMessage, const FIX::SessionID &objSessionID)
    CEXCEPT(FIX::FieldNotFound,
    FIX::IncorrectDataFormat,
    FIX::IncorrectTagValue,
    FIX::RejectLogon)
{
    try {

        std::string strMsgType = objMessage.getHeader().getField(FIX::FIELD::MsgType);

        // heartbeat
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "0") {

            publishHeartbeat(Mts::LifeCycle::CHeartbeat(m_Provider.getProviderID(), false,
                                                        (double) Mts::Core::CDateTime::now().getCMTime()));
        }

        // test request - store ID as it has to be sent back with each heartbeat
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "1") {

            m_strTestReqID = objMessage.getField(FIX::FIELD::TestReqID);
        }

        // reject
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "3") {

            if (objMessage.isSetField(58) == true) {

                std::string strText = objMessage.getField(58);
                AppLog(strText.c_str());
            }
        }
    }
    catch (std::exception &e) {

        AppError("fromadmin error: %s", e.what());
    }
}


// outgoing, Mts -> ECN
void CExchangeFIX_TT::toAdmin(FIX::Message &objMessage,
        const FIX::SessionID &objSessionID) {

    std::string strMsgType = objMessage.getHeader().getField(FIX::FIELD::MsgType);

    // logon
    if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "A") {
        objMessage.setField(FIX::RawData(m_strPassword));
        objMessage.setField(FIX::EncryptMethod(0));
        objMessage.setField(FIX::HeartBtInt(m_iHeartbeatsecs));
        objMessage.setField(FIX::ResetSeqNumFlag("Y"));
        objMessage.setField(FIX::MsgSeqNum(1));
        queryHeader(objMessage.getHeader());
    }

    // heartbeat
    if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "0") {
        objMessage.setField(FIX::TestReqID(m_strTestReqID));
    }
}


// not required - no market data
void CExchangeFIX_TT::onMessage(
        const FIX44::SecurityList &objMessage,
        const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onMessage(
        const FIX44::SecurityDefinition &objMessage,
        const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onMessage(
        const FIX44::MarketDataIncrementalRefresh &objMessage,
        const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onMessage(
        const FIX44::MarketDataSnapshotFullRefresh &objMessage,
        const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onMessage(
        const FIX44::MarketDataRequestReject &objMessage,
        const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onMessage(
        const FIX44::BusinessMessageReject &objMessage,
        const FIX::SessionID &objSessionID) {
}


void CExchangeFIX_TT::onMessage(const FIX44::ExecutionReport &objMessage,
        const FIX::SessionID &objSessionID) {

    try {

        FIX::ExecType objExecType;
        objMessage.get(objExecType);

        const std::string &strExecType = objExecType.getString();
        char iExecType = strExecType[0];

        switch (iExecType) {
            // new
            case '0':
                processExecutionReport_New(objMessage, objSessionID);
                return;

                // canceled
            case '4':
                processExecutionReport_Cancel(objMessage, objSessionID);
                return;

                // partial fill
            case '1':
                processExecutionReport_Fill(objMessage, objSessionID);
                return;

                // fill
            case '2':
            case 'F':
                processExecutionReport_Fill(objMessage, objSessionID);
                return;

                // replaced
            case '5':
                processExecutionReport_Replace(objMessage, objSessionID);
                return;

                // reject
            case '8':
                processExecutionReport_Reject(objMessage, objSessionID);
                return;

            default :
                AppLogError("unknown execution type (tag 150): %s", strExecType.c_str());
        }
    }
    catch (std::exception &e) {

        AppError("execreport error: %s", e.what());
    }
}


void CExchangeFIX_TT::processExecutionReport_Fill(
        const FIX44::ExecutionReport &objMessage,
        const FIX::SessionID &objSessionID) {
    unsigned int iField = 0;
    try {
        // algo and contract
        const std::string &strAlgo = objMessage.getField(16558);
        ++iField;
        const std::string &tt_security_id = objMessage.getField(48);
        ++iField;

        // this could throw if the tt_security_id not found
        const auto *tradable(utils::SymbolMapReader::get().getByTTSecId(tt_security_id));
        const std::string &contract(tradable->_tradable);
        const double px_mul(tradable->_px_multiplier);

        // clOrdId
        FIX::ClOrdID objClOrdID;
        objMessage.get(objClOrdID);
        const std::string &strClOrdID = objClOrdID.getString();
        ++iField;

        // TT assigned execution ID
        FIX::ExecID objExecID;
        objMessage.get(objExecID);
        const std::string &strExecID = objExecID.getString();
        ++iField;

        // order status
        FIX::OrdStatus objOrdStatus;
        objMessage.get(objOrdStatus);
        const std::string &strOrdStatus = objOrdStatus.getString();
        ++iField;

        // side
        const std::string &side = objMessage.getField(54);
        ++iField;
        int side_mul;
        if (side.at(0) == '1') {
            side_mul = 1;
        } else if (side.at(0) == '2') {
            side_mul = -1;
        } else {
            throw std::runtime_error(std::string("unknown side ") + side);
        }

        // transact time
        std::string strTransactTime = objMessage.getField(60);
        ++iField;

        // local time
        auto cur_micro(utils::TimeUtil::cur_micro());

        // last shares (i.e. last fill qty)
        unsigned int iLastShares = static_cast<unsigned int>(boost::lexical_cast<double>(objMessage.getField(32)));
        ++iField;

        // last price (i.e. last fill price)
        FIX::LastPx objLastPx;
        objMessage.get(objLastPx);
        const std::string &strLastPx = objLastPx.getString();
        double dLastPx = boost::lexical_cast<double>(strLastPx);
        ++iField;

        pm::ExecutionReport er_fill(
                contract,
                strAlgo,
                strClOrdID,
                strExecID,
                strOrdStatus,
                (int) iLastShares * side_mul,
                dLastPx * px_mul,
                strTransactTime,
                "",
                cur_micro,
                0);

        publishExecutionReport(er_fill);
    }
    catch (std::exception &e) {
        char szBuffer[1024];
        sprintf(szBuffer, "processExecutionReport_Fill: %s %d", e.what(), iField);
        AppLogError(szBuffer);
    }
}

void CExchangeFIX_TT::processExecutionReport_New(
        const FIX44::ExecutionReport &objMessage,
        const FIX::SessionID &objSessionID) {
    unsigned int iField = 0;
    try {
        // algo and contract
        const std::string &strAlgo = objMessage.getField(16558);
        ++iField;
        const std::string &tt_security_id = objMessage.getField(48);
        ++iField;

        // this could throw if the tt_security_id not found
        const auto *tradable(utils::SymbolMapReader::get().getByTTSecId(tt_security_id));
        const std::string &contract(tradable->_tradable);
        const double px_mul(tradable->_px_multiplier);

        // clOrdId
        FIX::ClOrdID objClOrdID;
        objMessage.get(objClOrdID);
        const std::string &strClOrdID = objClOrdID.getString();
        ++iField;

        // TT assigned execution ID
        FIX::ExecID objExecID;
        objMessage.get(objExecID);
        const std::string &strExecID = objExecID.getString();
        ++iField;

        // side
        const std::string &side = objMessage.getField(54);
        ++iField;
        int side_mul;
        if (side.at(0) == '1') {
            side_mul = 1;
        } else if (side.at(0) == '2') {
            side_mul = -1;
        } else {
            throw std::runtime_error(std::string("unknown side ") + side);
        }

        // order status
        FIX::OrdStatus objOrdStatus;
        objMessage.get(objOrdStatus);
        std::string strOrdStatus = objOrdStatus.getString();
        ++iField;
        int64_t reserved = 0;
        if (__builtin_expect(strOrdStatus != std::string("0"), 0)) {
            AppLogInfo((std::string("New with non-0 order status: ") + strOrdStatus + " clOrdId: " +
                        strClOrdID + " ExecID: " + strExecID).c_str());

            /*  
            //could this even be possible?
            if (strOrdStatus == std::string("2")) {
                AppLogInfo("fully filled!");
                return;
            }
            */
            if ((strOrdStatus == std::string("1")) || (strOrdStatus == std::string("2"))) {
                strOrdStatus = std::string("0");
                // read cumQty into the reserved field
                const std::string &cumQty = objMessage.getField(14);
                ++iField;
                AppLogInfo((std::string("Replaced with partial fill, cumQty:  ") + cumQty).c_str());
                reserved = std::stoll(cumQty) * side_mul;  // signed cumQty
            } else {
                AppLogError("unknown order status!");
                return;
            }
        }

        // transact time
        std::string strTransactTime = objMessage.getField(60);
        ++iField;

        // local time
        auto cur_micro(utils::TimeUtil::cur_micro());

        const int ordQty(std::stoi(objMessage.getField(38)));
        ++iField;
        const double ordPx(std::stod(objMessage.getField(44)));
        ++iField;

        pm::ExecutionReport er_new(
                contract,
                strAlgo,
                strClOrdID,
                strExecID,
                strOrdStatus,
                ordQty * side_mul,
                ordPx * px_mul,
                strTransactTime,
                "",
                cur_micro,
                reserved);

        publishExecutionReport(er_new);
    }
    catch (std::exception &e) {
        char szBuffer[1024];
        sprintf(szBuffer, "processExecutionReport_New: %s %d", e.what(), iField);
        AppLogError(szBuffer);
    }
}

void CExchangeFIX_TT::processExecutionReport_Replace(
        const FIX44::ExecutionReport &objMessage,
        const FIX::SessionID &objSessionID) {

    processExecutionReport_Cancel(objMessage, objSessionID);
    processExecutionReport_New(objMessage, objSessionID);
}

void CExchangeFIX_TT::processExecutionReport_Cancel(
        const FIX44::ExecutionReport &objMessage,
        const FIX::SessionID &objSessionID) {
    unsigned int iField = 0;
    try {
        FIX::ClOrdID objClOrdID;
        // child field doesn't not have ClOrdID so ignore and process updated parent fill instead
        if (objMessage.isSetField(objClOrdID) == false)
            return;

        // Cancel uses OrigClOrdID for referencing original ClOrdId, which is m_Order's key
        std::string strOrigClOrdID;
        FIX::OrigClOrdID objOrigClOrdID;
        if (objMessage.isSetField(objOrigClOrdID) == true) {
            objMessage.get(objOrigClOrdID);
            strOrigClOrdID = objOrigClOrdID.getString();
        } else {
            AppLogInfo("unsolicited cancel: original client order not found, using clOrdId instead");
            objMessage.get(objClOrdID);
            strOrigClOrdID = objClOrdID.getString();
            // trying to get a reject reason
            try {
                std::string strReason = objMessage.getField(58);
                AppLogError(strOrigClOrdID + (std::string(" Cancel reason: ") + strReason).c_str());
            } catch (const std::exception &e) {
                AppLogInfo(strOrigClOrdID + std::string(" Cannot get cancel reason"));
            }
        }
        ++iField;
        const std::string &strAlgo = objMessage.getField(16558);
        ++iField;
        const std::string &tt_security_id = objMessage.getField(48);
        ++iField;
        const std::string &contract(utils::SymbolMapReader::get().getByTTSecId(tt_security_id)->_tradable);

        // assigned execution ID
        FIX::ExecID objExecID;
        objMessage.get(objExecID);
        const std::string &strExecID = objExecID.getString();
        ++iField;

        // transact time
        std::string strTransactTime = objMessage.getField(60);
        ++iField;

        auto new_micro = utils::TimeUtil::cur_micro();
        pm::ExecutionReport er_cancel(
                contract,        // symbol (tag 55)
                strAlgo,         // algo
                strOrigClOrdID,  // tag 11 clOrdId
                strExecID,       // execId, together with tag 11 form a hash in pm
                "4",         // tag39 = cancel
                0,           // qty + or -
                0,           // px
                strTransactTime, // exchange time gmt
                "",  // additional tag
                new_micro,   // receive local micro
                0
        );
        publishExecutionReport(er_cancel);
    }

    catch (std::exception &e) {
        char szBuffer[1024];
        sprintf(szBuffer, "processExecutionReport_Canel: %s %d", e.what(), iField);
        AppLogError(szBuffer);
    }
}


void CExchangeFIX_TT::processExecutionReport_Reject(
        const FIX44::ExecutionReport &objMessage,
        const FIX::SessionID &objSessionID) {

    unsigned int iField = 0;

    try {
        FIX::ClOrdID objClOrdID;
        // child field doesn't not have ClOrdID so ignore and process updated parent fill instead
        if (objMessage.isSetField(objClOrdID) == false)
            return;
        objMessage.get(objClOrdID);
        const std::string &strClOrdID = objClOrdID.getString();
        ++iField;

        const std::string &strAlgo = objMessage.getField(16558);
        ++iField;
        const std::string &tt_security_id = objMessage.getField(48);
        ++iField;
        const std::string &contract(utils::SymbolMapReader::get().getByTTSecId(tt_security_id)->_tradable);

        // assigned execution ID
        FIX::ExecID objExecID;
        objMessage.get(objExecID);
        const std::string &strExecID = objExecID.getString();
        ++iField;

        // transact time
        std::string strTransactTime = objMessage.getField(60);
        ++iField;

        // reject reason
        std::string strReason = objMessage.getField(58);
        ++iField;

        auto new_micro = utils::TimeUtil::cur_micro();
        pm::ExecutionReport er_reject(
                contract,        // symbol (tag 55)
                strAlgo,         // algo
                strClOrdID,  // tag 11 clOrdId
                strExecID,       // execId, together with tag 11 form a hash in pm
                "8",         // tag39 = reject
                0,           // qty + or -
                0,           // px
                strTransactTime, // exchange time gmt
                strReason,  // additional tag
                new_micro,  // receive local micro
                0
        );
        AppLogError("Reject Reason: %s", strReason.c_str());
        publishExecutionReport(er_reject);
    }

    catch (std::exception &e) {
        char szBuffer[1024];
        sprintf(szBuffer, "processExecutionReport order rejection: %sz %d", e.what(), iField);
        AppLogError(szBuffer);
    }
}

void CExchangeFIX_TT::onMessage(const FIX44::OrderCancelReject &objMessage,
        const FIX::SessionID &objSessionID) {
//20220412-22:24:37.222453000 : 8=FIX.4.4^A9=00541^A35=9^A49=TT_ORDER^A56=massar_fix_trading^A34=89^A142=US,FL^A52=20220412-22:24:37.219^A129=ZFu^A37=9908c599-22b0-4a87-8a14-47b1056c1491^A198=8037389452407^A11=3-0-1649802277200177726^A10011=21^A41=3-0-1649802277068129157^A39=0^A1=00MSR-TSD^A60=20220412-22:24:37.211000^A434=2^A102=2179^A58=Order price is outside bands 'Ask of 10109.00 violates Low Band 10110.00 using Delta .00, C.Last equals 10160.00'^A10553=zhenghua.fu@massarcapital.com^A18220=NE2^A18221=MSC^A16999=00MSR^A16558=TSD-7000-370^A16561=20220412-22:24:37.211639^A16117=10^A1031=Y^A18226=0^A18216=2LJ714^A10=150^A

    unsigned int iField = 0;

    try {
        FIX::ClOrdID objClOrdID;
        // child field doesn't not have ClOrdID so ignore and process updated parent fill instead
        if (objMessage.isSetField(objClOrdID) == false)
            return;
        objMessage.get(objClOrdID);
        const std::string &strClOrdID = objClOrdID.getString();
        ++iField;

        std::string strAlgo;
        try {
            strAlgo = objMessage.getField(16558);
        } catch (const std::exception &e) {
            strAlgo = "";
        }

        // assigned execution ID
        //FIX::ExecID objExecID;
        //objMessage.get(objExecID);
        //const std::string & strExecID = objExecID.getString();
        //++iField;

        // transact time
        std::string strTransactTime = objMessage.getField(60);
        ++iField;

        // reject reason
        std::string strReason = objMessage.getField(58);
        ++iField;

        auto new_micro = utils::TimeUtil::cur_micro();
        pm::ExecutionReport er_reject(
                "",        // symbol (tag 55)
                strAlgo,         // algo
                strClOrdID,  // tag 11 clOrdId
                "",       // execId, together with tag 11 form a hash in pm
                "8",         // tag39 = reject
                0,           // qty + or -
                0,           // px
                strTransactTime, // exchange time gmt
                strReason,  // additional tag
                new_micro,  // receive local micro
                0
        );
        AppLogError("Cancel Reject Reason: %s", strReason.c_str());
        publishExecutionReport(er_reject);
    }

    catch (std::exception &e) {
        char szBuffer[1024];
        sprintf(szBuffer, "processExecutionReport for cancel rejection (35=9): %sz %d", e.what(), iField);
        AppLogError(szBuffer);
    }
}


void CExchangeFIX_TT::onMessage(const FIX44::OrderMassCancelReport &objMessage,
        const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onMessage(const FIX44::News &objMessage,
        const FIX::SessionID &objSessionID) {

    m_bRecoveryComplete = true;
    AppLog("news");
    publishExchangeMessage("Recovery Done");
}


std::string CExchangeFIX_TT::getSenderCompID() const {

    return m_SenderCompID.getString();
}


std::string CExchangeFIX_TT::getSessionQualifier() const {

    return "";
}


// do nothing, QuickFIX will create a separate thread for the session so let the IRunnable (CExchange) thread terminate
void CExchangeFIX_TT::operator()() {

}


void CExchangeFIX_TT::queryHeader(FIX::Header &objHeader, bool set_operator_id) {
    objHeader.setField(m_SenderCompID);
    objHeader.setField(m_TargetCompID);

    objHeader.setField(FIX::OnBehalfOfSubID(m_strUsername));

    if (__builtin_expect(set_operator_id, 1)) {
        const auto tag50(pm::risk::Monitor::get().status().get_operator_id());
        objHeader.setField(FIX::SenderSubID(tag50));
    }

    /*
    objHeader.setField(FIX::StringField(25004, m_strUsername));
    objHeader.setField(FIX::StringField(1028, "N"));
    */
}

const std::string CExchangeFIX_TT::getStrAccount(const std::string &algo) const {
    if (__builtin_expect(m_strAccount == "Massar", 0)) {
        // uat
        return m_strAccount;
    }
    char buf[64];
    std::size_t pos = algo.find("-");
    snprintf(buf, sizeof(buf), "%s-%s", m_strAccount.c_str(), algo.substr(0, pos).c_str());
    return std::string(buf);
}


Mts::Order::COrder::OrderState CExchangeFIX_TT::tt2MtsOrderState(const std::string &strECNOrderState) const {

    char iOrdStatus = strECNOrderState[0];

    switch (iOrdStatus) {
        case '0':
            return Mts::Order::COrder::NEW;

        case '1':
            return Mts::Order::COrder::PARTIALLY_FILLED;

        case '2':
            return Mts::Order::COrder::FILLED;

        case '4':
            return Mts::Order::COrder::CANCELLED;

        case '6':
            return Mts::Order::COrder::CANCEL_PENDING;

        case '8':
            return Mts::Order::COrder::REJECTED;

        default:
            throw Mts::Exception::CMtsException("unable to match MCM order state");
    }
}


unsigned int CExchangeFIX_TT::getProviderID() const {

    return m_Provider.getProviderID();
}


void CExchangeFIX_TT::onMessage(const FIX44::UserResponse &objMessage,
        const FIX::SessionID &objSessionID) {

}


void CExchangeFIX_TT::onMessage(const FIX44::CollateralReport &objMessage,
        const FIX::SessionID &objSessionID) {

}

