#include "CMath.h"
#include "quickfix/SocketInitiator.h"
#include "quickfix/SessionSettings.h"
#include "CDropCopyFIX_TT.h"
#include "CMtsException.h"
#include "CApplicationLog.h"

#include "CSymbol.h"
#include "ExecutionReport.h"

using namespace Mts::Exchange;

//#pragma warning( disable : 4503 4355 4786 )

CDropCopyFIX_TT::CDropCopyFIX_TT(
    const Mts::Core::CProvider & objProvider,
    const std::string & strSenderCompID,
    const std::string &	strTargetCompID,
    const std::string &	strUsername,
    const std::string &	strPassword,
    const std::string &	strAccount,
    unsigned int        iHeartbeatsecs) 
    : CExchangeFIX_TT(
        objProvider,
        strSenderCompID,
        strTargetCompID,
        strUsername,
        strPassword,
        strAccount,
        iHeartbeatsecs),
    m_recover(),
    m_bDone(false)
{
    // the session credentials are stored 
    // and retrieved during logon
    AppLog(std::string("DropCopy Recovery Session created ") );
};

void CDropCopyFIX_TT::onLogon(const FIX::SessionID & objSessionID) {
    publishLogon(getProviderID());
    std::cout << std::endl << "Logon - " << objSessionID << std::endl;
    AppLog("DropCopy Logged On");
    if (isDone()) {
        AppLog("DropCopy Session No Recovery Request Found!");
        disconnect();
        return;
    }
    sendRecoveryRequet();
};

void CDropCopyFIX_TT::setRequest(const std::string& startLocalTime, const std::string& endLocalTime) {
    m_recover.init(startLocalTime, endLocalTime);
    setDone(false);
}

void CDropCopyFIX_TT::onLogout(const FIX::SessionID & objSessionID) {
    publishLogout(getProviderID());
    std::cout << std::endl << "Logout - " << objSessionID << std::endl;
    AppLog("DropCopy Logged Out");
    setDone(true);
};

void CDropCopyFIX_TT::sendRecoveryRequet() {
    if (m_recover.startUTC()  == "" || m_recover.endUTC() == "") {
        AppLog("Start/End time not defined, recovery request not sent");
        disconnect();
    }

    FIX::Message U2;
    U2.getHeader().setField(FIX::BeginString("FIX.4.4"));
    U2.getHeader().setField(FIX::StringField(35, "U2"));
    U2.setField(FIX::StringField(916, m_recover.startUTC()));
    U2.setField(FIX::StringField(917, m_recover.endUTC()));
    queryHeader(U2.getHeader());

    AppLog(std::string("Sending recovery request: ") + U2.toString());

    bool ret = false;
    try {
        ret = FIX::Session::sendToTarget(U2, getSessionQualifier());
    }
    catch (const std::exception& e) {
        AppLog(std::string("Failed to send recovery request: ") + e.what());
    }
    if (!ret) {
        AppLog(std::string("Failed to send recovery, disconnecting!"));
        disconnect();
    }
}

void CDropCopyFIX_TT::onMessage(
    const FIX44::ExecutionReport & objMessage,
    const FIX::SessionID & objSessionID)  {
    try {
        processExecutionReport(objMessage);
    }
    catch (std::exception & e) {
        AppError("execreport: %s", e.what());
    }
}

void CDropCopyFIX_TT::onMessage(
    const FIX44::BusinessMessageReject & objMessage,
    const FIX::SessionID &objSessionID) {
    AppLogError("Received reject: %s", objMessage.toString().c_str());
    disconnect();
    setDone(true);
}

bool CDropCopyFIX_TT::getOrdId(const FIX44::ExecutionReport& objMessage, std::string& ordId)  {
    // try to get tag 11, which should be set by orders submitted by MTS
    // otherwise, get tag 37, which is TT order id, for orders submitted from web
    // in the later case, the order id is set as NONE-orderid, as a way to identify the state
    // Needs to be consistent with isWebOrder()

    const int TAG_ClOrdID = 11;
    if (objMessage.isSetField(TAG_ClOrdID) == false) {
        // try tag 37
        const int TAG_OrderId = 37;
        if (objMessage.isSetField(TAG_OrderId) == false) {
            AppLog(std::string("Error! neither ClOrdID (tag 11) nor OrderId (tag 37) is set for ") + objMessage.toString());
            return false;
        }
        ordId = std::string("NONE-") + objMessage.getField(TAG_OrderId);
    }
    else {
        ordId = objMessage.getField(TAG_ClOrdID);
    }
    return true;
}

bool CDropCopyFIX_TT::getSymbol(const FIX44::ExecutionReport& objMessage, std::string& symbol)  {
    // if TAG_NoSecurityAltID is set, then getting from the repeating group of 454: 455/456
    // get the name where 456=98 (name), i.e. CLU0
    // Otherwise, getting the symbol from tag 55 Symbol

    const int TAG_Symbol = 55;
    symbol = objMessage.getField(TAG_Symbol);

    const int TAG_NoSecurityAltID = 454;
    if (objMessage.isSetField(TAG_NoSecurityAltID)) {
        // try to read symbol name from this repeating group
        const int noGrp = atoi(objMessage.getField(TAG_NoSecurityAltID).c_str());
        const int TAG_SecurityAltID = 455;
        const int TAG_SecurityAltIDSource = 456;
        for (int i = 1; i <= noGrp; ++i) {
            // get a group and see 
            auto grp = FIX44::ExecutionReport::NoSecurityAltID();
            objMessage.getGroup(i, grp);
            if (grp.getField(TAG_SecurityAltIDSource) == "98") {
                symbol = grp.getField(TAG_SecurityAltID);
                return true;
            }
        }
        AppLog("SecurityAltID repeating group found but no SecurityAltIDSource is 98!");
        return false;
    }
    return true;
}

bool CDropCopyFIX_TT::isWebOrder(const FIX44::ExecutionReport& objMessage) {
    const int TAG_ClOrdID = 11;
    return  (objMessage.isSetField(TAG_ClOrdID) == false);
}

/////  ****** the conversion to ExecutionReport
void CDropCopyFIX_TT::processExecutionReport(const FIX44::ExecutionReport & objMessage) {
    try {
        if (isWebOrder(objMessage)) {
            AppLog("Got an order without ClOrdId, possibly a web order, ignored from recovery: %s", objMessage.toString().c_str());
            return;
        }

        // clOrdid
        std::string sClOrdId;
        if (!CDropCopyFIX_TT::getOrdId(objMessage, sClOrdId)) {
            AppLogError("Cannot get orderid, skip this fill: %s", objMessage.toString().c_str());
            return;
        }
        // Tag Algo, if exists
        std::string algo;
        const int TAG_TextTT = 16558;
        if (objMessage.isSetField(TAG_TextTT)) {
            algo = objMessage.getField(TAG_TextTT);
        } else {
            AppLogError("Cannot get algo name, skipping this report");
            return;
        }

        // symbol
        std::string strSymbol;
        if (!CDropCopyFIX_TT::getSymbol(objMessage, strSymbol)) {
            AppLogError("Cannot get symbol name, skipping this report");
            return;
        }

        // exec id
        const int TAG_ExecID = 17;
        const std::string execid = objMessage.getField(TAG_ExecID);

        // transact time
        const int TAG_TransactTime = 60;
        const std::string exchGMT = objMessage.getField(TAG_TransactTime);

        // status string
        const int TAG_OrdStatus = 39;
        const std::string st = objMessage.getField(TAG_OrdStatus);

        double px = 0;
        int qty = 0;
        switch (st[0]) {
        case '4': 
        {
            // cancel
            // Cancel uses OrigClOrdID for referencing original ClOrdId, which is m_Order's key
            const int TAG_OrigClOrdID = 41;
            sClOrdId = objMessage.getField(TAG_OrigClOrdID);
            break;
        }
        case '0':
        case '1':
        case '2':
        {
            // get the side and px mul
            auto& csymbol (Mts::Core::CSymbol::getSymbolFromExchContract(strSymbol));
            double ttmul = csymbol.getTTScaleMultiplier();
            // tag Side(54)
            // "1": BUY, "2": Sell
            const int TAG_Side = 54;
            const std::string bsString = objMessage.getField(TAG_Side);
            int sidemul;
            if (bsString == "1")
                sidemul = 1;  // buy
            else if (bsString == "2")
                sidemul = -1; // sell
            else {
                AppLogError("Unknow Buy Sell tag 54: %s", objMessage.toString().c_str());
                return;
            }
            qty = 0;
            px = 0;
            if (st[0] == '0') {
                // new
                // Order Qty
                const int TAG_OrderQty = 38;
                if (objMessage.isSetField(TAG_OrderQty)) {
                    qty = std::stoi(objMessage.getField(TAG_OrderQty).c_str());
                }

                // Limit Price
                const int TAG_Price = 44;
                if (objMessage.isSetField(TAG_Price) == true) {
                    px = atof(objMessage.getField(TAG_Price).c_str());
                }
            } else {
                // last shares (i.e. last fill qty)
                const int TAG_LastQty = 32;
                qty = static_cast<long long>(boost::lexical_cast<double>(objMessage.getField(TAG_LastQty)));

                // last price (i.e. last fill price)
                const int TAG_LastPx = 31;
                px = boost::lexical_cast<double>(objMessage.getField(TAG_LastPx));
            }
            qty *= sidemul;
            px *= ttmul;
            break;
        }
        default : 
            AppLogError("Unknow tag 39: %s", st.c_str());
            return;
        }
        m_recover.addExecutionReport
            (
                std::make_shared<pm::ExecutionReport>
                (
                    strSymbol,
                    algo,
                    sClOrdId,
                    execid,
                    st,
                    qty,
                    px,
                    exchGMT,
                    "",
                    utils::TimeUtil::cur_micro(),
                    0
                )
            );
    }
    catch (std::exception & e) {
        AppLog(std::string("DropCopy Exception during processing Fill: ") + e.what() + std::string("\n") + objMessage.toString());
    }
}
