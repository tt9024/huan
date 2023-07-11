#include <quickfix/FileStore.h>
#include <quickfix/FileLog.h>
#include <quickfix/SocketInitiator.h>
#include <quickfix/SessionSettings.h>

#include "CFIXEngine.h"
#include "CApplicationLog.h"

using namespace Mts::FIXEngine;


// disable reoccuring warnings from the QuickFIX libs
//#pragma warning( disable : 4503 4355 4786 )


CFIXEngine::CFIXEngine(const std::string &strSettingsFile)
        : m_strSettingsFile(strSettingsFile) {

}


void CFIXEngine::addSession(boost::shared_ptr<IFIXSession> ptrSession) {

    std::string strUniqueSessioID = ptrSession->getSenderCompID() + ptrSession->getSessionQualifier();
    m_Sessions[strUniqueSessioID] = ptrSession;
}


int CFIXEngine::getNumSessions() const {

    return m_Sessions.size();
}


void CFIXEngine::run() {

    if (getNumSessions() == 0)
        return;

    m_ptrThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::ref(*this)));
}


void CFIXEngine::operator()() {

    AppLog("Initializing QuickFIX");

    try {
        FIX::SessionSettings objSettings(m_strSettingsFile);
        FIX::FileStoreFactory objStoreFactory(objSettings);
        FIX::FileLogFactory objLogFactory(objSettings);

        // note, FIX::SocketInitiator uses a single thread to handle all sessions (reader, i.e. data, or write, i.e. venues) so has a small performance hit
        FIX::SocketInitiator objInitiator(*this, objStoreFactory, objSettings, objLogFactory);

        objInitiator.start();

        m_bRunning = true;
        m_bShutdown = false;

        while (m_bRunning == true) {
            // just wait for stop
            sleep_milli(200);
        }

        AppLog("Stopping QuickFIX");

        objInitiator.stop(true);
        m_bShutdown = true;
    }
    catch (std::exception &e) {

        AppLogError(e.what());
    }
}


void CFIXEngine::onCreate(const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onCreate(objSessionID);
}


void CFIXEngine::onLogon(const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onLogon(objSessionID);
}


void CFIXEngine::onLogout(const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onLogout(objSessionID);
}


// incoming ECN -> Mts
void CFIXEngine::fromApp(const FIX::Message &objMessage,
        const FIX::SessionID &objSessionID) CEXCEPT(FIX::FieldNotFound,
        FIX::IncorrectDataFormat,
        FIX::IncorrectTagValue,
        FIX::UnsupportedMessageType) {

    try {
        crack(objMessage, objSessionID);
    }
    catch (std::exception &e) {
        AppLogError(e.what());
    }
}


// incoming, ECN -> Mts
void CFIXEngine::fromAdmin(const FIX::Message &objMessage,
        const FIX::SessionID &objSessionID) CEXCEPT(FIX::FieldNotFound,
        FIX::IncorrectDataFormat,
        FIX::IncorrectTagValue,
        FIX::RejectLogon)
{

    try {
        getSession(objSessionID)->fromAdmin(objMessage, objSessionID);
    }
    catch (std::exception &e) {

        AppLogError(e.what());
    }
}


// outgoing, Mts -> ECN
void CFIXEngine::toApp(FIX::Message &objMessage,
        const FIX::SessionID &objSessionID) CEXCEPT(FIX::DoNotSend)
{

    try {
        FIX::PossDupFlag objPossDupFlag;

        if (objMessage.isSetField(objPossDupFlag)) {
            objMessage.getHeader().getField(objPossDupFlag);

            if (objPossDupFlag)
                throw FIX::DoNotSend();
        }
    }
    catch (FIX::FieldNotFound &e) {

        AppLogError(e.what());
    }
}


// outgoing, Mts -> ECN
void CFIXEngine::toAdmin(FIX::Message &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->toAdmin(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::MarketDataIncrementalRefresh &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::MarketDataSnapshotFullRefresh &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::MarketDataRequestReject &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::ExecutionReport &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::OrderCancelReject &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::SecurityList &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::SecurityDefinition &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


boost::shared_ptr<IFIXSession> CFIXEngine::getSession(const FIX::SessionID &objSessionID) {

    std::string strUniqueSessioID = objSessionID.getSenderCompID().getString() + objSessionID.getSessionQualifier();
    return m_Sessions[strUniqueSessioID];
}


void CFIXEngine::stopFIXEngine() {

    m_bRunning = false;
}


bool CFIXEngine::isShutdown() const {

    return m_bShutdown;
}


void CFIXEngine::onMessage(const FIX44::UserResponse &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::CollateralReport &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


void CFIXEngine::onMessage(const FIX44::News &objMessage,
        const FIX::SessionID &objSessionID) {

    getSession(objSessionID)->onMessage(objMessage, objSessionID);
}


