#include <ctime>
#include <set>
#include <iostream>

#include "bbtp.h"

using namespace BloombergLP;
using namespace blpapi;

namespace tp {
namespace bpipe {

BPIPE_Config::BPIPE_Config(const char* config_file) {
    const utils::ConfigureReader rdr( 
            (
                ((!config_file) || (!config_file[0]))?
                    plcc_getString("MDProviders.BPipe").c_str(): 
                    config_file
            )
        );

    d_pub = rdr.get<std::string>("Publisher");
    d_hosts = rdr.getArr<std::string>("IP");
    d_port = rdr.get<int>("Port");
    d_service = rdr.get<std::string>("Service");
    setTopics();
    setFields();
    // leave the d_options empty
    // Application Name set to a string of  "App:User"
    d_authOptions.append("AuthenticationMode=APPLICATION_ONLY;"
                         "ApplicationAuthenticationType=APPNAME_AND_KEY;"
                         "ApplicationName=");
    d_authOptions.append(rdr.get<std::string>("App")+":"+rdr.get<std::string>("User"));
    d_clientCredentials = rdr.get<std::string>("Key");
    d_clientCredentialsPassword = rdr.get<std::string>("Passwd");
    d_trustMaterial = rdr.get<std::string>("Certificate");

    logInfo("BPIPE Config created: %s", toString().c_str());
}

std::string BPIPE_Config::toString() const {
    return std::string("Publisher:") + d_pub + ", Service:" + d_service + ", Topics:" + std::to_string(d_topics.size());
}

TlsOptions BPIPE_Config::getTlsOptions() const {
    return TlsOptions::createFromFiles(
               d_clientCredentials.c_str(),
               d_clientCredentialsPassword.c_str(),
               d_trustMaterial.c_str()
           );
}

SessionOptions BPIPE_Config::getSessionOptions() const {
    SessionOptions sessionOptions;
    for (size_t i = 0; i < d_hosts.size(); ++i) {
        sessionOptions.setServerAddress(d_hosts[i].c_str(), d_port, i);
    }
    sessionOptions.setServerPort(d_port);
    sessionOptions.setAuthenticationOptions(d_authOptions.c_str());
    sessionOptions.setAutoRestartOnDisconnection(true);
    sessionOptions.setNumStartAttempts(d_hosts.size());
    sessionOptions.setTlsOptions(getTlsOptions());
    return sessionOptions;
}

void BPIPE_Config::setFields() {
    d_fields.push_back("MKTDATA_EVENT_TYPE");
    d_fields.push_back("MKTDATA_EVENT_SUBTYPE");

    d_fields.push_back("EVT_TRADE_PRICE_RT");
    d_fields.push_back("EVT_TRADE_SIZE_RT");
    d_fields.push_back("EVT_TRADE_CONDITION_CODE_RT");
    //d_fields.push_back("EVT_TRADE_LOCAL_EXCH_SOURCE_RT");
    //d_fields.push_back("EVT_TRADE_TIME_RT");
    
    d_fields.push_back("EVT_QUOTE_BID_PRICE_RT");
    d_fields.push_back("EVT_QUOTE_BID_SIZE_RT");
    //d_fields.push_back("EVT_QUOTE_BID_TIME_RT");

    d_fields.push_back("EVT_QUOTE_ASK_PRICE_RT");
    d_fields.push_back("EVT_QUOTE_ASK_SIZE_RT");
    //d_fields.push_back("EVT_QUOTE_ASK_TIME_RT");

    /* potentially useful
    d_fields.push_back("EVT_TRADE_ACTION_REALTIME");
    d_fields.push_back("EVT_TRADE_INDICATOR_REALTIME");
    d_fields.push_back("EVT_TRADE_RPT_PRTY_SIDE_RT");
    d_fields.push_back("EVT_TRADE_RPT_PARTY_TYP_TR");
    d_fields.push_back("EVT_TRADE_RTP_CONTRA_TYP_RT");
    
    d_fields.push_back("EVT_SOURCE_TIME_RT");
    d_fields.push_back("EVT_TRADE_BLOOMBERG_STD_CC_RT");

    d_fields.push_back("EVT_TRADE_ORIGINAL_IDENTIFIER_RT");
    d_fields.push_back("EVT_TRADE_EXECUTION_TIME_RT");
    d_fields.push_back("EVT_TRADE_AGGRESSOR_RT");
    d_fields.push_back("EVT_TRADE_BUY_BROKER_RT");
    d_fields.push_back("EVT_TRADE_SELL_BROKER_RT");
    */
}

void BPIPE_Config::setTopics() {
    // read from main.cfg and symbol_map
    /*
    const auto& symbols (utils::SymbolMapReader::get().getSubscriptions(d_pub));
    for (const auto& s : symbols.first) {
        const std::string topic_str = std::string("/ticker/") + utils::SymbolMapReader::get().getTradableInfo(s)->_bbg_id;
        logInfo("adding primary symbol %s as topic", s.c_str(), topic_str.c_str());
        d_topics.push_back(topic_str);
        d_topics_primary.insert(topic_str);
    }
    for (const auto& s : symbols.second) {
        const std::string topic_str = std::string("/ticker/") + utils::SymbolMapReader::get().getTradableInfo(s)->_bbg_id;
        logInfo("adding secondary symbol %s as topic", s.c_str(), topic_str.c_str());
        d_topics.push_back(topic_str);
    }

    */
    const char* sym = "/ticker/ENJ2COJ2 COMDTY";
    d_topics.push_back(sym);
    d_topics_primary.insert(sym);

    //d_topics.push_back("/ticker/CLG2CLH2 COMDTY");
    //d_topics_primary.insert("/ticker/CLG2CLH2 COMDTY");
}



BPIPE_Thread::BPIPE_Thread(const char* config_file)
: m_cfg(config_file) , m_should_run(false) {}

void BPIPE_Thread::kill() {
    if (m_should_run) {
        logInfo("Killing %s", m_cfg.toString().c_str());
        m_should_run = false; 
    } else {
        logInfo("Not killing %s, already in stopping", m_cfg.toString().c_str());
    }
};


bool BPIPE_Thread::authorize(const Service &authService,
              Identity *subscriptionIdentity,
              Session *session)
{
    static const Name TOKEN_SUCCESS("TokenGenerationSuccess");
    static const Name TOKEN_FAILURE("TokenGenerationFailure");
    static const Name AUTHORIZATION_SUCCESS("AuthorizationSuccess");
    static const Name TOKEN("token");

    EventQueue tokenEventQueue;
    session->generateToken(CorrelationId(), &tokenEventQueue);
    std::string token;
    Event event = tokenEventQueue.nextEvent();
    if (event.eventType() == Event::TOKEN_STATUS ||
        event.eventType() == Event::REQUEST_STATUS) {
        MessageIterator iter(event);
        while (iter.next()) {
            Message msg = iter.message();
            if (msg.messageType() == TOKEN_SUCCESS) {
                token = msg.getElementAsString(TOKEN);
            }
            else if (msg.messageType() == TOKEN_FAILURE) {
                msg.print(std::cout);
                break;
            }
        }
    }
    if (token.length() == 0) {
        logError("BPipe failed to get token!");
        return false;
    }

    Request authRequest = authService.createAuthorizationRequest();
    authRequest.set(TOKEN, token.c_str());
    session->sendAuthorizationRequest(authRequest, subscriptionIdentity);

    time_t startTime = time(0);
    const int WAIT_TIME_SECONDS = 10;
    while (true) {
        Event event = session->nextEvent(WAIT_TIME_SECONDS * 1000);
        if (event.eventType() == Event::RESPONSE ||
            event.eventType() == Event::REQUEST_STATUS ||
            event.eventType() == Event::PARTIAL_RESPONSE)
        {
            MessageIterator msgIter(event);
            while (msgIter.next()) {
                Message msg = msgIter.message();
                if (msg.messageType() == AUTHORIZATION_SUCCESS) {
                    logInfo("BPipe authorized!");
                    return true;
                }
                else {
                    msg.print(std::cout);
                    logError("BPipe authorization failed!");
                    return false;
                }
            }
        }
        time_t endTime = time(0);
        if (endTime - startTime > WAIT_TIME_SECONDS) {
            logError("BPipe authorization timed out after %d seconds", WAIT_TIME_SECONDS);
            return false;
        }
    }
}

void BPIPE_Thread::connect(Session& session) {
    if (!session.start()) {
        logError("Failed to start session for %s", m_cfg.toString().c_str());
        throw std::runtime_error("Failed to start session!");
    }

    // authorize with user and app
    Identity subscriptionIdentity;
    if (!m_cfg.d_authOptions.empty()) {
        subscriptionIdentity = session.createIdentity();
        bool isAuthorized = false;
        const char* authServiceName = "//blp/apiauth";
        if (session.openService(authServiceName)) {
            Service authService = session.getService(authServiceName);
            isAuthorized = authorize(authService, &subscriptionIdentity, &session);
        }
        if (!isAuthorized) {
            logError("Failed to authorize!");
            throw std::runtime_error("Failed to authorize!");
        }
    }

    if (m_cfg.d_topics.size()==0) {
        return;
    }

    SubscriptionList subscriptions;
    const auto& svc (m_cfg.d_service);
    for (const auto& topic: m_cfg.d_topics) {
        subscriptions.add((svc+topic).c_str(),
                          m_cfg.d_fields,
                          m_cfg.d_options,
                          CorrelationId((char*)topic.c_str()));
        const std::string symbol (getSymbol(topic.c_str()));
        //test
        //m_sym_pxmul_map[symbol] = utils::SymbolMapReader::get().getTradableInfo(symbol)->_bbg_px_multiplier;
    }
    session.subscribe(subscriptions, subscriptionIdentity);
}

const char* BPIPE_Thread::getTopic(const Message& msg) const {
    return (char*)msg.correlationId().asPointer();
}

const char* BPIPE_Thread::getSymbol(const char* topic) const {
    // removing the initial "/ticker/"
    return topic + 8;
}

bool BPIPE_Thread::checkFailure(const Event::EventType& eventType, const Message& message) const {
    static const Name SESSION_TERMINATED("SessionTerminated");
    static const Name SESSION_STARTUP_FAILURE("SessionStartupFailure");
    static const Name SERVICE_OPEN_FAILURE("ServiceOpenFailure");
    static const Name SUBSCRIPTION_FAILURE("SubscriptionFailure");
    static const Name SUBSCRIPTION_TERMINATED("SubscriptionTerminated");

    const Name& messageType = message.messageType();
    if (eventType == Event::SUBSCRIPTION_STATUS) {
        if (messageType == SUBSCRIPTION_FAILURE || 
            messageType == SUBSCRIPTION_TERMINATED) 
        {
            const char* error = message.getElement("reason").getElementAsString("description");
            logError("Subscription failed for %s: %s", getTopic(message), error); 
            // sub failed shouldn't kill the tp
            return false;
        }
    } else if (eventType == Event::SESSION_STATUS) {
        if (messageType == SESSION_TERMINATED ||
            messageType == SESSION_STARTUP_FAILURE) 
        {
            const char* error = message.getElement("reason").getElementAsString("description");
            logError("Session failed to start or terminated: %s", error);
            return true;
        }
    } else if (eventType == Event::SERVICE_STATUS) {
        if (messageType == SERVICE_OPEN_FAILURE) 
        {
            const char* serviceName = message.getElementAsString("serviceName");
            const char* error = message.getElement("reason").getElementAsString("description");
            logError("Failed to open %s: %s", serviceName, error);
            return true;
        }
    }
    return false;
}

void BPIPE_Thread::run() {
    try {
        md::MD_Publisher<BPIPE_BOOK_LEVEL> pub(m_cfg.d_pub);
        Session session(m_cfg.getSessionOptions());
        connect(session);
        m_should_run = true;

        // keep track of the unsubscribed topic, for logging purpose
        std::set<std::string> sub_topics (m_cfg.d_topics.begin(), m_cfg.d_topics.end());

        // stay and wait for the kill!
        if (sub_topics.size() == 0) {
            logError("Bpipe have no subscriptions, waiting to be killed!");
            while (m_should_run) {
                utils::TimeUtil::micro_sleep(1000000ULL);
            }
            return;
        }

        while (m_should_run) {
            // this is a blocking call
            Event event = session.nextEvent(); 
            MessageIterator msgIter(event);
            while (msgIter.next()) {
                Message msg = msgIter.message();
                const std::string& msg_name (msg.asElement().name().string());
                if (__builtin_expect(msg_name == "MarketDataEvents", 1)) {
                    //const char* topic = getTopic(msg);
                    //const char* symbol = getSymbol(topic);
                    try {
                        //process(symbol, msg, pub);
                        msg.print(std::cout);

                    } catch (const std::exception& e) {
                        logError("Exception in processing MarketDataEvents: %s", e.what());
                        logError("MarketData Message Dump: %s", printMsg(msg).c_str());
                        msg.print(std::cout);
                    }
                } else if (msg_name == "SubscriptionStreamsActivated") {
                    const char* topic = getTopic(msg);
                    std::string pubname = m_cfg.d_pub;
                    std::string vname;
                    if (sub_topics.erase(topic)) {
                        if (m_cfg.d_topics_primary.find(topic)!=m_cfg.d_topics_primary.end()) {
                            pubname = "";
                            logInfo("BPipep subscribed to %s as primary (%s)", getSymbol(topic), pubname.c_str());
                        } else {
                            logInfo("BPipep subscribed to %s as secondary (%s)", getSymbol(topic), pubname.c_str());
                        }
                        //pub.addBookQ(getSymbol(topic), vname, pubname);
                        msg.print(std::cout);
                    } else {
                        msg.print(std::cout);
                        logError("duplicated or unknown subscription returned: %s, msg: %s", topic, printMsg(msg).c_str());
                    }
                    if (sub_topics.size() == 0) {
                        logInfo("All symbols subscribed successfully!");
                    }
                } else { 
                    Event::EventType eventType = event.eventType();
                    if (checkFailure(eventType, msg)) {
                        msg.print(std::cout);
                        kill();
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        logError("Exception in bpipe publisher: %s", e.what());
        return;
    }
}

void BPIPE_Thread::process(const std::string& symbol, const Message& msg, md::MD_Publisher<BPIPE_BOOK_LEVEL>& pub) {
    // tight loop for processing the Event::SUBSCRIPTION_DATA
    // get the event type
    static const Name nmet("MKTDATA_EVENT_TYPE");
    static const Name nmes("MKTDATA_EVENT_SUBTYPE");
    static const Name nqbp("EVT_QUOTE_BID_PRICE_RT");
    static const Name nqbs("EVT_QUOTE_BID_SIZE_RT");
    static const Name nqap("EVT_QUOTE_ASK_PRICE_RT");
    static const Name nqas("EVT_QUOTE_ASK_SIZE_RT");
    static const Name ntpx("EVT_TRADE_PRICE_RT");
    static const Name ntsz("EVT_TRADE_SIZE_RT");
    static const Name ntcc("EVT_TRADE_CONDITION_CODE_RT");

    const auto iter = m_sym_pxmul_map.find(symbol);
    if (__builtin_expect(iter == m_sym_pxmul_map.end(), 0)) {
        logError("unknown symbol received in bbg publisher: %s", symbol.c_str());
        throw std::runtime_error("unknown symbol received in bbg publisher: " + symbol);
    }
    const double pxmul = iter->second;

    const char* type {msg.getElement(nmet).getValueAsString()};
    const char* sub_type {msg.getElement(nmes).getValueAsString()};
    if (strcmp(type ,"QUOTE")==0) {
        if (strcmp(sub_type, "BID")==0) {
            // get the nqbp and nqbs
            if (msg.hasElement(nqbp,true) && msg.hasElement(nqbs, true)) {
                double px = msg.getElement(nqbp).getValueAsFloat64() * pxmul;
                unsigned sz = msg.getElement(nqbs).getValueAsInt32();
                pub.getWriter(symbol,"")->updBBO(px,sz,true, utils::TimeUtil::cur_micro());
            }
        } else if (strcmp(sub_type, "ASK")==0) {
            // get the nqap and nqas
            if (msg.hasElement(nqap,true) && msg.hasElement(nqas, true)) {
                double px = msg.getElement(nqap).getValueAsFloat64() * pxmul;
                unsigned sz = msg.getElement(nqas).getValueAsInt32();
                pub.getWriter(symbol,"")->updBBO(px,sz,false, utils::TimeUtil::cur_micro());
            }
        } else if (strcmp(sub_type, "PAIRED")==0) {
            // get both bid/ask 
            if (msg.hasElement(nqbp,true) && msg.hasElement(nqbs, true) &&
                msg.hasElement(nqap,true) && msg.hasElement(nqas, true)) {

                // there could be empty value in px and sz
                // not an error, just continue
                double bpx = msg.getElement(nqbp).getValueAsFloat64() * pxmul;
                unsigned bsz = msg.getElement(nqbs).getValueAsInt32();
                double apx = msg.getElement(nqap).getValueAsFloat64() * pxmul;
                unsigned asz = msg.getElement(nqas).getValueAsInt32();
                pub.getWriter(symbol,"")->updBBO(bpx,bsz,true, utils::TimeUtil::cur_micro());
                pub.getWriter(symbol,"")->updBBO(apx,asz,false, utils::TimeUtil::cur_micro());
            }
        } else {
            logError("unknown QUOTE sub type %s: %s", sub_type, printMsg(msg).c_str());
        }
    } else if (strcmp(type, "TRADE")==0) {
        if (__builtin_expect(strcmp(sub_type, "NEW")==0,1)) {
            double px = msg.getElement(ntpx).getValueAsFloat64() * pxmul;
            unsigned sz = msg.getElement(ntsz).getValueAsInt32();
            if (msg.hasElement(ntcc,true)) {
                const char* cc {msg.getElement(ntcc).getValueAsString()};
                // Only publish if code is "TSUM"
                if (strcmp(cc, "TSUM")==0) {
                    pub.getWriter(symbol,"")->updTrade(px,sz,utils::TimeUtil::cur_micro());
                }
                // TODO - consider adding "OR"
            } else {
                // markets such as ICE don't have condition code
                // px and sz is already TSUM
                pub.getWriter(symbol,"")->updTrade(px,sz,utils::TimeUtil::cur_micro());
            }
        } else {
            if (strcmp(sub_type, "CANCEL")!=0) {
                logError("unknown TRADE sub type %s", sub_type);
            }
        }
    } else if (strcmp(type, "SUMMARY")==0) {
        // do nothing
    } else {
        logError("unknown message type %s: %s", type, printMsg(msg).c_str());
    }
}

std::string BPIPE_Thread::printMsg(const Message& msg) const {
    // msg name, items, topic, fields
    const std::string& msg_name (msg.asElement().name().string());
    std::string ret = std::string(getTopic(msg)) + "("+msg_name + ") { ";
    for (const auto& n : m_cfg.d_fields) {
        try {
            Name fn(n.c_str());
            Element field;
            field = msg.getElement(fn);
            ret += (n + "(" + field.getValueAsString() + ") ");
        } catch (const std::exception& e) {
        }
    }
    ret += "}";
    return ret;
}

} // namespace bpipe
} // namespace tp


