#include "CMath.h"
#include "quickfix/FileStore.h"
#include "quickfix/FileLog.h"
#include "quickfix/SocketInitiator.h"
#include "quickfix/SessionSettings.h"
#include "CFeedFIX_TT.h"
#include "CQuote.h"
#include "CSymbol.h"
#include "CApplicationLog.h"
#include "CBidAsk.h"
#include "CStringTokenizer.h"

using namespace Mts::Feed;

#pragma warning( disable : 4503 4355 4786 4996 )

#define FullRefreshSubscriptionType 0
#define IncrementalRegreshSubscriptionType 1

CFeedFIX_TT::CFeedFIX_TT(const Mts::Core::CProvider &objProvider,
        const std::string &strSenderCompID,
        const std::string &strTargetCompID,
        const std::string &strUserName,
        const std::string &strPassword,
        unsigned int iHeartbeMtsecs)
        : CFeed(true),
          m_Provider(objProvider),
          m_SenderCompID(strSenderCompID),
          m_TargetCompID(strTargetCompID),
          m_strUserName(strUserName),
          m_strPassword(strPassword),
          m_iHeartbeMtsecs(iHeartbeMtsecs) {
    m_strTestReqID = "Mts";
}


void CFeedFIX_TT::addSubscription(const Mts::Core::CSymbol &objSymbol) {
    m_Subscriptions.insert(objSymbol.getSymbol());
}


std::string CFeedFIX_TT::getSenderCompID() const {

    return m_SenderCompID;
}


std::string CFeedFIX_TT::getSessionQualifier() const {

    return "";
}


bool CFeedFIX_TT::connect() {

    return true;
}


bool CFeedFIX_TT::disconnect() {

    FIX44::Logout objMessage;

    queryHeader(objMessage.getHeader());
    FIX::Session::sendToTarget(objMessage, getSessionQualifier());

    m_MatchedSubscriptions.clear();
    return true;
}


void CFeedFIX_TT::onCreate(const FIX::SessionID &) {
}


void CFeedFIX_TT::onLogon(const FIX::SessionID &objSessionID) {

    publishLogon(m_Provider.getProviderID());
    AppLog("Logon - " + objSessionID.toString());

    // request quotes for all symbol/counterparty combinations
    m_MatchedSubscriptions.clear();
    for (const auto &ms: m_Subscriptions) {
        requestSecurityDef(ms);
    }
}


void CFeedFIX_TT::initialize() {

    time_t utc_now = time(NULL);
    time_t check_due = utc_now + 5;

    AppLog("Waiting for contract matching of all subscriptions");
    while ((m_MatchedSubscriptions.size() != m_Subscriptions.size()) &&
           (utc_now < check_due)) {
        // removed for WIN32 compatible issues 
        // usleep (1000000) 
        utc_now = time(NULL);
    }
    if (m_MatchedSubscriptions.size() != m_Subscriptions.size()) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Failed to match contract expiration month of all subscriptions, exit! Matched (%d) Total (%d)",
                 (int) m_MatchedSubscriptions.size(), (int) m_Subscriptions.size());
        AppLogError(std::string(buf));

        // print out a trace on which symbol not subscribed and move on
        for (const auto &ms: m_Subscriptions) {
            if (m_MatchedSubscriptions.find(ms) == m_MatchedSubscriptions.end()) {
                AppLogError(std::string("!!! Failed to match security definition for ") + ms);
            }
        }

        // we still allow the system to continue
        //throw std::runtime_error(std::string(buf));
        AppLog("Only subscribe tp matched symbols...");
    } else {
        AppLog("All %d subscriptions matched!", (int) m_MatchedSubscriptions.size());
    }

    // subscription already happened upon secuirty definition response
    /*
    for (const auto& sym : m_MatchedSubscriptions) {
        requestMktData(sym, true);
        AppLog("Subscribing to " + sym);
    }
    */
}

void CFeedFIX_TT::onLogout(const FIX::SessionID &objSessionID) {

    publishLogout(m_Provider.getProviderID());
    AppLog("Logout - " + objSessionID.toString());
    m_MatchedSubscriptions.clear();
}


// incoming, ECN -> Mts
void CFeedFIX_TT::fromAdmin(const FIX::Message &objMessage,
        const FIX::SessionID &objSessionID) CEXCEPT(FIX::FieldNotFound,
        FIX::IncorrectDataFormat,
        FIX::IncorrectTagValue,
        FIX::RejectLogon)
{
    try {
        std::string strMsgType = objMessage.getHeader().getField(FIX::FIELD::MsgType);

        // heartbeat
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "0") {

        }

        // test request - store ID as it has to be sent back with each heartbeat
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "1") {

            m_strTestReqID = objMessage.getField(FIX::FIELD::TestReqID);
        }

        // reject
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "3") {

        }

        // logout
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "5") {

            std::string strMsg = objMessage.getField(FIX::FIELD::Text);
            AppLog("Logout: " + strMsg);
        }
    }
    catch (std::exception &e) {
        AppError("fromadmin: %s", e.what());
    }
}


// outgoing, Mts -> ECN
void CFeedFIX_TT::toAdmin(FIX::Message &objMessage,
        const FIX::SessionID &objSessionID)
{
    try {
        std::string strMsgType = objMessage.getHeader().getField(FIX::FIELD::MsgType);

        // logon
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "A") {
            objMessage.setField(FIX::RawData(m_strPassword));
            objMessage.setField(FIX::EncryptMethod(0));
            objMessage.setField(FIX::HeartBtInt(m_iHeartbeMtsecs));
            objMessage.setField(FIX::ResetSeqNumFlag("Y"));
            objMessage.setField(FIX::MsgSeqNum(1));
            queryHeader(objMessage.getHeader());
        }

        // reject
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "3") {

        }

        // heartbeat
        if (objMessage.getHeader().getField(FIX::FIELD::MsgType) == "0") {
            objMessage.setField(FIX::TestReqID(m_strTestReqID));
        }
    }
    catch (std::exception &e) {
        AppError("toadmin: %s", e.what());
    }

}


void CFeedFIX_TT::onMessage(const FIX44::SecurityList &,
        const FIX::SessionID &) {

}


void CFeedFIX_TT::onMessage(const FIX44::SecurityDefinition &objMessage,
        const FIX::SessionID &objSessionID) {

    unsigned int iField = 0;
    std::string strMtsSymbol = "";

    try {

        FIX::SecurityReqID objSecReqID;
        objMessage.get(objSecReqID);
        std::string strSecReqID = objSecReqID.getString();
        ++iField;

        // map back to mts symbol id
        Mts::Core::CStringTokenizer objTokenizer;
        std::vector<std::string> objTokens = objTokenizer.splitString(strSecReqID, "_");
        unsigned int iMtsSymbolID = boost::lexical_cast<unsigned int>(objTokens[3]);
        const Mts::Core::CSymbol &objMtsSymbol = Mts::Core::CSymbol::getSymbol(iMtsSymbolID);
        strMtsSymbol = objMtsSymbol.getSymbol();

        FIX::SecurityResponseID objSecRespID;
        objMessage.get(objSecRespID);
        std::string strSecRespID = objSecRespID.getString();
        ++iField;

        /*
        FIX::SecurityResponseType objSecRespType;
        objMessage.get(objSecRespType);
        int iSecRespType = objSecRespType.getValue();
        ++iField;
        */

        // TT unique security ID
        // tag 48 is not required
        /*
        FIX::SecurityID objSecID;
        objMessage.get(objSecID);
        std::string strSecID = objSecID.getString();
        ++iField;
        */

        FIX::SecurityExchange objSecExch;
        objMessage.get(objSecExch);
        std::string strSecExch = objSecExch.getString();
        ++iField;

        FIX::SecurityType objSecType;
        objMessage.get(objSecType);
        std::string strSecType = objSecType.getString();
        ++iField;

        FIX::Symbol objSymbol;
        objMessage.get(objSymbol);
        std::string strSymbol = objSymbol.getString();
        ++iField;

        FIX::SecurityDesc objSecDesc;
        objMessage.get(objSecDesc);
        std::string strSecDesc = objSecDesc.getString();
        ++iField;

        FIX::Currency objCurrency;
        objMessage.get(objCurrency);
        std::string strCurrency = objCurrency.getString();
        ++iField;

        FIX::MaturityMonthYear objMatMY;
        objMessage.get(objMatMY);
        std::string strMatMY = objMatMY.getString();
        ++iField;

        FIX::MaturityDate objMatDate;
        objMessage.get(objMatDate);
        std::string strMatDate = objMatDate.getString();
        ++iField;

        std::string strDeliveryTerm = objMessage.getField(18211);
        ++iField;

        // get the tick size and point value
        /* TODO - pending test setup in UAT
        std::string strTickSize = objMessage.getField(16552);
        std::string strPointValue = objMessage.getField(16554);
        std::string strPxMult = objMessage.getField(9787);
        std::string strTickTableEntries = objMessage.getField(16456);
        AppLog("Got tick info: 16552(%s) 16554(%s) 16456(%s) 9787(%s)", strTickSize.c_str(), strPointValue.c_str(), strTickTableEntries.c_str(), strPxMult.c_str());
        */

        // Check that the contract is monthly
        std::string strContractYM = "";
        if (strDeliveryTerm == "M") {
            // CFE does not return tag 18223
            int iFieldYM = strSecExch == "CFE" ? 200 : 18223;
            strContractYM = objMessage.getField(iFieldYM);
            ++iField;
        }

        // Show contract
        char cContractInfo[512];
        //snprintf(cContractInfo, sizeof(cContractInfo), "%s/%s/%s/%s/%s/%s/%s/%s/%s/%s", strMtsSymbol.c_str(), strCurrency.c_str(), strSecExch.c_str(), strSecType.c_str(), strSymbol.c_str(), strDeliveryTerm.c_str(), strMatMY.c_str(), strContractYM.c_str(), strSecDesc.c_str(), strSecID.c_str());
        snprintf(cContractInfo, sizeof(cContractInfo), "%s/%s/%s/%s/%s/%s/%s/%s/%s", strMtsSymbol.c_str(),
                 strCurrency.c_str(), strSecExch.c_str(), strSecType.c_str(), strSymbol.c_str(),
                 strDeliveryTerm.c_str(), strMatMY.c_str(), strContractYM.c_str(), strSecDesc.c_str());

        // map TT security ID to mts symbol
        if (strContractYM.compare(objMtsSymbol.getContractTicker()) == 0) {
            //Mts::Core::CSymbol::mapTTSecID(iMtsSymbolID, strSecID);

            char szBuffer[1024];
            snprintf(szBuffer, sizeof(szBuffer), "Matching Mts Symbol %s to %s", objMtsSymbol.getSymbol().c_str(),
                     cContractInfo);
            AppLog(szBuffer);

            const auto &p(m_MatchedSubscriptions.insert(strMtsSymbol));
            if (p.second) {
                //requestMktData(strMtsSymbol, true, FullRefreshSubscriptionType);
                requestMktData(strMtsSymbol, true, IncrementalRegreshSubscriptionType);
                AppLog("Subscribing to " + strMtsSymbol);
            } else {
                AppLogError("Received duplicated security definition response on " + strMtsSymbol);
            }
        } else {

            //Mts::Core::CSymbol::mapTTSecID(iMtsSymbolID, "");

            char szBuffer[1024];
            snprintf(szBuffer, sizeof(szBuffer),
                     "Unable to match Mts Symbol (%s doesn't match with %s) MTS Symbol: %s Feed Symbol: %s ",
                     objMtsSymbol.getContractTicker().c_str(), strContractYM.c_str(),
                     objMtsSymbol.getSymbol().c_str(), cContractInfo);
            AppLogError(szBuffer);
        }
    }
    catch (std::exception &e) {
        char szBuffer[1024];
        sprintf(szBuffer, "Failed to parse security definition response: %s %d %s", e.what(), iField,
                strMtsSymbol.c_str());
        AppLogError(szBuffer);
    }
}

void CFeedFIX_TT::onMessage(const FIX44::MarketDataIncrementalRefresh &objMessage,
        const FIX::SessionID &objSessionId) {
    unsigned int iField = 0;
    try {
        FIX::MDReqID objMDReqID;
        objMessage.get(objMDReqID);
        const std::string &strMDEntryRefID = objMDReqID.getString();
        ++iField;

        // getting symbol from reqId[6:]
        const std::string strSymbol(strMDEntryRefID.substr(6));
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);
        const std::string &strTTSecID = objSymbol.getTTSecID();
        unsigned int iMtsSymbolID = objSymbol.getSymbolID();
        double dPxMul = objSymbol.getTTScaleMultiplier();

        Mts::Core::CDateTime dtMtsTimestamp = Mts::Core::CDateTime::now();
        Mts::Core::CDateTime dtExcTimestamp = dtMtsTimestamp;

        Mts::OrderBook::CQuote quote;
        quote.setProviderID(m_Provider.getProviderID());
        quote.setMtsTimestamp(dtMtsTimestamp);
        quote.setExcTimestamp(dtExcTimestamp);
        quote.setSide(Mts::OrderBook::CQuote::BID);
        quote.setPrice(0);
        quote.setSize(0);
        quote.setSymbolID(iMtsSymbolID);
        quote.setValueDateJulian(static_cast<unsigned int>(dtMtsTimestamp.getValue()));

        m_Trade.setSymbolID(iMtsSymbolID);
        m_Trade.setProviderID(m_Provider.getProviderID());
        m_Trade.setMtsTimestamp(dtMtsTimestamp);
        m_Trade.setExcTimestamp(dtExcTimestamp);
        m_Trade.setPrice(0);
        m_Trade.setSize(0);

        FIX::NoMDEntries objNoMDEntries;
        objMessage.get(objNoMDEntries);
        int iNumGroups = Mts::Math::CMath::atoi(objNoMDEntries.getString());
        ++iField;

        // the quotes to be applied later
        std::vector<Mts::OrderBook::CQuote> quotes_;
        for (int i = 1; i <= iNumGroups; ++i) {
            iField = i * 100;
            FIX44::MarketDataIncrementalRefresh::NoMDEntries objGroup;
            objMessage.getGroup(i, objGroup);

            // action
            FIX::MDUpdateAction objMDEntryAction;
            objGroup.get(objMDEntryAction);
            int iEntryAction = Mts::Math::CMath::atoi(objMDEntryAction.getString());
            ++iField;

            // type
            FIX::MDEntryType objMDEntryType;
            objGroup.get(objMDEntryType);
            std::string strType = objMDEntryType.getString();
            if (strType[0] == 'J') {
                // clear book
                logInfo("Recieved EntryType %s, clearing book %s!", strType.c_str(), strSymbol.c_str());
                quote.setSize(0);
                quote.setPrice(0);
                quote.setSide(Mts::OrderBook::CQuote::BID);
                quotes_.push_back(quote);
                //publishHalfQuote(quote);
                quote.setSide(Mts::OrderBook::CQuote::ASK);
                quotes_.push_back(quote);
                //publishHalfQuote(quote);
                continue;
            }
            if (strType[0] == 'L') {
                // L is leg trade, take as '2'
                strType = std::string("2");
            } else if (strType[0] == 'm') {
                logInfo("Incremental refresh got 269=m (OTC trade), take as a trade");
                // OTC trade, take as '2'
                strType = std::string("2");
            }
            if (!(strType[0] >= '0' && strType[0] <= '2')) {
                // we only process 269=0,1,2, or 'J', 'L', 'm' (see above)
                logError("Incremental refresh unknown tag 269(%s), 279(%d), ignored", strType.c_str(), iEntryAction);
                continue;
            }
            int iEntryType = std::stoi(strType);
            ++iField;

            // px
            FIX::MDEntryPx objMDEntryPx;
            objGroup.get(objMDEntryPx);
            double dPrice = Mts::Math::CMath::atof(objMDEntryPx.getString()) * dPxMul;
            ++iField;

            // sz
            FIX::MDEntrySize objMDEntrySize;
            objGroup.get(objMDEntrySize);
            double dSize = Mts::Math::CMath::atof(objMDEntrySize.getString());
            ++iField;

            if (((iEntryAction == 0) || (iEntryAction == 1)) && (iEntryType != 2)) {
                // a new or change BBO, expect px/size/side, non-trade type
                if (__builtin_expect((iEntryType != 0) && (iEntryType != 1), 0)) {
                    logError("Incremental Refresh parsing error: unknown action (%d) type(%d)", iEntryAction,
                             iEntryType);
                    throw std::runtime_error("Incremental Refresh parsing error: unknown type");
                }

                // side from tag 269(type): 0-bid or 1-ask
                auto side = (iEntryType == 0 ? Mts::OrderBook::CQuote::BID : Mts::OrderBook::CQuote::ASK);
                quote.setSide(side);
                quote.setSize(dSize);
                quote.setPrice(dPrice);
                //publishHalfQuote(quote);
                quotes_.push_back(quote);
            } else if (iEntryType == 2) {
                // a trade received under action that is not new/change
                // process trade in full refresh
                m_Trade.setPrice(dPrice);
                m_Trade.setSize(dSize);
                publishTrade(m_Trade);
            } else {
                if (iEntryAction != 2) {
                    // ignore delete action, no need to do anything
                    logError("Incremental Refresh parsing error: unknown action (%d) type(%d)", iEntryAction,
                             iEntryType);
                } else {
                    // TODO - trade direction correction
                    // since the quote/trades are conflated into 100ms updates, orders of trades/quotes need
                    // to be inferred for trade direction.
                    //
                    // if there is delete quote, then consider apply the trade before applying the quote,
                    // except the initial bid/ask before deletion cannot determine the direction, 
                    // i.e. trade in the middle.  
                    // In this case publish trade after quote.
                    //
                    // For example, bid/ask is 1/2
                    // 0. delete ask 2, add 3
                    // 1. delete bid 1, add 2
                    // 2. trade at 2
                    // the trade should be applied before quote
                    //
                    // For example, bid/ask is 1/3
                    // 0. delete ask 3, add 2
                    // 1. trade at 2
                    // the trade should be applied after quote
                }
            }
        }
        for (const auto &qt: quotes_) {
            publishHalfQuote(qt);
        }
    }

    catch (std::exception &e) {
        char szBuffer[256];
        snprintf(szBuffer, sizeof(szBuffer), "MktData: %s %d", e.what(), iField);
        AppLogError(szBuffer);
    }
}

void CFeedFIX_TT::onMessage(const FIX44::MarketDataSnapshotFullRefresh &objMessage,
        const FIX::SessionID &objSessionID) {

    unsigned int iField = 0;

    try {
        // traded exchange
        FIX::SecurityExchange objSecExch;
        objMessage.get(objSecExch);
        const std::string &strExch = objSecExch.getValue();
        ++iField;

        // sub-exchange
        FIX::ExDestination objExDest;
        objMessage.getField(objExDest);
        const std::string &strExDest = objExDest.getValue();
        ++iField;

        // ccy
        FIX::Currency objCcy;
        objMessage.getField(objCcy);
        ++iField;

        // security ID
        FIX::SecurityID objSecID;
        objMessage.getField(objSecID);
        const std::string &strSecID = objSecID.getValue();
        ++iField;

        // todo Need to map by TT symbol here
        const Mts::Core::CSymbol &objMtsSymbol = Mts::Core::CSymbol::getSymbolFromTTSecID(strSecID);
        unsigned int iMtsSymbolID = objMtsSymbol.getSymbolID();

        FIX::NoMDEntries objNoMDEntries;
        objMessage.get(objNoMDEntries);
        ++iField;

        int iNumGroups = Mts::Math::CMath::atoi(objNoMDEntries.getString());

        Mts::Core::CDateTime dtMtsTimestamp = Mts::Core::CDateTime::now();
        Mts::Core::CDateTime dtExcTimestamp = dtMtsTimestamp;

        // empty message is the signal to clear all quotes in the book for this provider/symbol
        if (iNumGroups == 0) {
            for (unsigned int i = 0; i != 2; ++i) {

                Mts::OrderBook::CQuote &objQuote = i == 0 ? m_QuoteBid : m_QuoteAsk;
                Mts::OrderBook::CQuote::Side iSide = i == 0 ? Mts::OrderBook::CQuote::BID : Mts::OrderBook::CQuote::ASK;

                objQuote.setSymbolID(iMtsSymbolID);
                objQuote.setProviderID(m_Provider.getProviderID());
                objQuote.setMtsTimestamp(dtMtsTimestamp);
                objQuote.setExcTimestamp(dtExcTimestamp);
                objQuote.setSide(iSide);
                objQuote.setPrice(0);
                objQuote.setSize(0);
                objQuote.setExch(strExch);
                objQuote.setExDest(strExDest);
            }

            publishQuote(Mts::OrderBook::CBidAsk(iMtsSymbolID, dtMtsTimestamp, m_QuoteBid, m_QuoteAsk));
        } else {

            FIX44::MarketDataSnapshotFullRefresh::NoMDEntries objGroup;

            m_QuoteBid.setSymbolID(iMtsSymbolID);
            m_QuoteBid.setProviderID(m_Provider.getProviderID());
            m_QuoteBid.setMtsTimestamp(dtMtsTimestamp);
            m_QuoteBid.setExcTimestamp(dtExcTimestamp);
            m_QuoteBid.setSide(Mts::OrderBook::CQuote::BID);
            m_QuoteBid.setPrice(0);
            m_QuoteBid.setSize(0);
            m_QuoteBid.setExch(strExch);
            m_QuoteBid.setExDest(strExDest);
            m_QuoteBid.setValueDateJulian(static_cast<unsigned int>(dtMtsTimestamp.getValue()));

            m_QuoteAsk.setSymbolID(iMtsSymbolID);
            m_QuoteAsk.setProviderID(m_Provider.getProviderID());
            m_QuoteAsk.setMtsTimestamp(dtMtsTimestamp);
            m_QuoteAsk.setExcTimestamp(dtExcTimestamp);
            m_QuoteAsk.setSide(Mts::OrderBook::CQuote::ASK);
            m_QuoteAsk.setPrice(0);
            m_QuoteAsk.setSize(0);
            m_QuoteAsk.setExch(strExch);
            m_QuoteAsk.setExDest(strExDest);
            m_QuoteAsk.setValueDateJulian(static_cast<unsigned int>(dtMtsTimestamp.getValue()));

            m_Trade.setSymbolID(iMtsSymbolID);
            m_Trade.setProviderID(m_Provider.getProviderID());
            m_Trade.setMtsTimestamp(dtMtsTimestamp);
            m_Trade.setExcTimestamp(dtExcTimestamp);
            m_Trade.setPrice(0);
            m_Trade.setSize(0);
            m_Trade.setExch(strExch);
            m_Trade.setExDest(strExDest);

            for (int i = 1; i <= iNumGroups; ++i) {

                iField = i * 100;

                objMessage.getGroup(i, objGroup);

                FIX::MDEntryPx objMDEntryPx;
                objGroup.get(objMDEntryPx);
                double dPrice = Mts::Math::CMath::atof(objMDEntryPx.getString());
                ++iField;

                FIX::MDEntrySize objMDEntrySize;
                objGroup.get(objMDEntrySize);
                double dSize = Mts::Math::CMath::atof(objMDEntrySize.getString());
                ++iField;

                FIX::MDEntryType objMDEntryType;
                objGroup.get(objMDEntryType);

                std::string strType = objMDEntryType.getString();
                if (strType[0] == 'L') {
                    strType = std::string("2");
                };
                if (!(strType[0] >= '0' && strType[0] <= '2')) {
                    // we don't know how to process this type
                    continue;
                }
                int iEntryType = std::stoi(strType);
                ++iField;

                if (iEntryType == 2) {

                    m_Trade.setSymbolID(iMtsSymbolID);
                    m_Trade.setProviderID(m_Provider.getProviderID());
                    m_Trade.setMtsTimestamp(dtMtsTimestamp);
                    m_Trade.setExcTimestamp(dtExcTimestamp);
                    m_Trade.setPrice(dPrice * objMtsSymbol.getTTScaleMultiplier());
                    m_Trade.setSize(static_cast<unsigned int>(dSize));
                    m_Trade.setExch(strExch);
                    m_Trade.setExDest(strExDest);

                    publishTrade(m_Trade);
                } else if ((iEntryType == 0) || (iEntryType == 1)) {
                    Mts::OrderBook::CQuote::Side iSide =
                            iEntryType == 0 ? Mts::OrderBook::CQuote::BID : Mts::OrderBook::CQuote::ASK;

                    // value date
                    Mts::Core::CDateTime dtSettDate = dtMtsTimestamp;

                    Mts::OrderBook::CQuote &objQuote = iSide == Mts::OrderBook::CQuote::BID ? m_QuoteBid : m_QuoteAsk;

                    objQuote.setSymbolID(iMtsSymbolID);
                    objQuote.setProviderID(m_Provider.getProviderID());
                    objQuote.setMtsTimestamp(dtMtsTimestamp);
                    objQuote.setExcTimestamp(dtExcTimestamp);
                    objQuote.setSide(iSide);
                    objQuote.setPrice(dPrice * objMtsSymbol.getTTScaleMultiplier());
                    objQuote.setSize(static_cast<unsigned int>(dSize));
                    objQuote.setExch(strExch);
                    objQuote.setExDest(strExDest);
                    objQuote.setValueDateJulian(static_cast<unsigned int>(dtSettDate.getValue()));

                    // IEEE 754 specifies +0.0==-0.0, the comparison is platform independent
                    if ((m_QuoteBid.getPrice() != 0.0) && (m_QuoteAsk.getPrice() != 0.0))
                        publishQuote(Mts::OrderBook::CBidAsk(iMtsSymbolID, dtMtsTimestamp, m_QuoteBid, m_QuoteAsk));
                } else {
                    logError("received unknown market data entry type %d", iEntryType);
                }
            }
        }

    }
    catch (std::exception &e) {

        char szBuffer[1024];
        sprintf(szBuffer, "MktData: %s %d", e.what(), iField);
        AppLogError(szBuffer);
    }
}


void CFeedFIX_TT::onMessage(const FIX44::MarketDataRequestReject &objMessage,
        const FIX::SessionID &objSessionID) {

    try {

        FIX::MDReqID objMDReqID;
        objMessage.get(objMDReqID);

        FIX::MDReqRejReason objMDReqRejReason;
        objMessage.get(objMDReqRejReason);

        FIX::Text objText;
        objMessage.get(objText);

        // Cannot pass non-trivial types to variadic function (ensure getValue() returns char)
        AppLogError("Recieved MarketDataRequestReject: Reason=%c, Message=%s",
                    objMDReqRejReason.getValue(), objText.getValue().c_str());
    }
    catch (std::exception &e) {
        AppError("MarketDataRequestReject: %s", e.what());
    }
}


void CFeedFIX_TT::onMessage(const FIX44::BusinessMessageReject &,
        const FIX::SessionID &) {
    AppLogError("Received BusinessMessageReject!");
}


void CFeedFIX_TT::onMessage(const FIX44::ExecutionReport &,
        const FIX::SessionID &) {

}


void CFeedFIX_TT::onMessage(const FIX44::OrderCancelReject &,
        const FIX::SessionID &) {

}


void CFeedFIX_TT::onMessage(const FIX44::OrderMassCancelReport &,
        const FIX::SessionID &) {

}


void CFeedFIX_TT::onMessage(const FIX44::News &,
        const FIX::SessionID &) {

}


// QuickFIX sessions are already on a separate thread so let this IRunnable (CFeed) thread terminate
void CFeedFIX_TT::operator()() {

}


void CFeedFIX_TT::requestSecurityDef(const std::string &strSymbol) {
    // symSymbol is a MTS symbol, i.e. WTI_N1
    try {


        FIX44::SecurityDefinitionRequest objMessage;

        // unique request ID
        Mts::Core::CDateTime dtNow = Mts::Core::CDateTime::now();

        // symbol
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);
        const std::string &strSecurityType = objSymbol.getSecurityType();
        const std::string &strExch = objSymbol.getTTExchange();
        const std::string &strExchSymbol = objSymbol.getExchSymbol();
        const std::string &strContractTicker = objSymbol.getContractTicker();
        const std::string &strContractExchSymbol = objSymbol.getContractExchSymbol();
        const std::string &strContractExpiration = objSymbol.getContractMonth();
        const std::string &strTTSecID = objSymbol.getTTSecID();

        char szSecReqID[255];
        sprintf(szSecReqID, "SRID_TT_%llu_%d", dtNow.getCMTime(), objSymbol.getSymbolID());

        char msg[511];
        sprintf(msg, "%s,%s,%s,%s,%s,%s",
                strSecurityType.c_str(),
                strExch.c_str(),
                strExchSymbol.c_str(),
                strContractTicker.c_str(),
                strContractExchSymbol.c_str(),
                strContractExpiration.c_str());
        AppLog(msg);


        // unique request ID
        FIX::SecurityReqID objSecReqID(szSecReqID);
        objMessage.setField(objSecReqID);

        objMessage.setField(FIX::SecurityExchange(strExch));
        objMessage.setField(FIX::SecurityType(strSecurityType));
        objMessage.setField(FIX::SecurityID(strTTSecID));
        objMessage.setField(FIX::IDSource("96"));
        objMessage.setField(FIX::Symbol(strExchSymbol));

        // request tick size and point value
        // TODO - pending test in UAT setup
        //objMessage.setField(17000, "Y");

        queryHeader(objMessage.getHeader());
        FIX::Session::sendToTarget(objMessage, getSessionQualifier());
    }
    catch (std::exception &e) {
        AppError("requestSecurityDef: %s", e.what());
    }
}


// one off subscription request, made once successful logon is confirmed
void CFeedFIX_TT::requestMktData(const std::string &strSymbol,
        bool bSubscribe,
        unsigned int iSubscriptionType) {

    try {

        FIX44::MarketDataRequest objMessage;
        int cm = (int) (utils::TimeUtil::cur_micro() % 1000);
        // the ID[6:] is strSymbol
        char szMarketDataReqID[255];
        sprintf(szMarketDataReqID, "%c%c%c%cTT%s",
                (char) (cm % 10 + '0'),
                (char) ((cm / 10) % 10 + '0'),
                (char) ((cm / 100) % 10 + '0'),
                (char) (iSubscriptionType % 10 + '0'),
                strSymbol.c_str());

        // unique request ID
        FIX::MDReqID objMDReqID(szMarketDataReqID);
        objMessage.setField(objMDReqID);

        // 0 = snapshot, 1 = subscribe, 2 = unsubscribe
        char iSubFlag = bSubscribe == true ? '1' : '2';
        FIX::SubscriptionRequestType objSubType(iSubFlag);
        objMessage.setField(objSubType);

        // market depth
        FIX::MarketDepth objMarketDepth(1);
        objMessage.setField(objMarketDepth);

        // 0 = full refresh (1000ms throttle), 1 = incremental (100ms throttle)
        FIX::MDUpdateType objMDUpdateType(iSubscriptionType);
        objMessage.setField(objMDUpdateType);

        // NoMDEntryTypes
        FIX44::MarketDataRequest::NoMDEntryTypes objNoMDEntryTypes;
        objNoMDEntryTypes.setField(FIX::MDEntryType('0'));  // bid
        objMessage.addGroup(objNoMDEntryTypes);
        objNoMDEntryTypes.setField(FIX::MDEntryType('1'));  // ask
        objMessage.addGroup(objNoMDEntryTypes);
        objNoMDEntryTypes.setField(FIX::MDEntryType('2'));  // trade
        objMessage.addGroup(objNoMDEntryTypes);

        // NoRelatedSym
        FIX44::MarketDataRequest::NoRelatedSym objNoRelatedSym;
        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);
        const std::string &strTTSecID = objSymbol.getTTSecID();
        const std::string &strSecurityType = objSymbol.getSecurityType();
        const std::string &strExchSymbol = objSymbol.getExchSymbol();
        const std::string &strContractTicker = objSymbol.getContractTicker();
        const std::string &strExchange = objSymbol.getExchange();

        // skip if we were unable to match the MTS symbol to a TT security ID
        if (strTTSecID.length() == 0) {

            char szBuffer[512];
            sprintf(szBuffer, "CFeedFIX_TT::requestMktData - Unable to match Mts Symbol %s to a TT security ID",
                    objSymbol.getSymbol().c_str());
            AppLog(szBuffer);
            return;
        }

        objNoRelatedSym.setField(FIX::SecurityID(strTTSecID));
        objNoRelatedSym.setField(FIX::IDSource("96"));
        objNoRelatedSym.setField(FIX::Symbol(strExchSymbol));
        objNoRelatedSym.setField(FIX::SecurityExchange(strExchange));
        objMessage.addGroup(objNoRelatedSym);

        queryHeader(objMessage.getHeader());
        FIX::Session::sendToTarget(objMessage, getSessionQualifier());
    }
    catch (std::exception &e) {
        AppError("requestMktData: %s", e.what());
    }
}


void CFeedFIX_TT::queryHeader(FIX::Header &objHeader) {

    // add fields not provided by default
    objHeader.setField(m_SenderCompID);
    objHeader.setField(m_TargetCompID);
    objHeader.setField(FIX::OnBehalfOfSubID(m_strUserName));
}


void CFeedFIX_TT::onMessage(const FIX44::UserResponse &,
        const FIX::SessionID &) {

}


void CFeedFIX_TT::onMessage(const FIX44::CollateralReport &,
        const FIX::SessionID &) {

}


