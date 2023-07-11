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

    // Save blapi version string for logging/reference
    setBlapiVersion();

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
    return std::string("Publisher:") + d_pub +
        ", Service:" + d_service +
        ", Topics:" + std::to_string(d_topics.size()) +
        ", API Version: " + blapiVersion();
}

void BPIPE_Config::setBlapiVersion()
{
    blpapi::VersionInfo blpapiVersion;
    blpapi_version_ = std::to_string(blpapiVersion.majorVersion()) + ".";
    blpapi_version_ += std::to_string(blpapiVersion.minorVersion()) + ".";
    blpapi_version_ += std::to_string(blpapiVersion.patchVersion()) + ".";
    blpapi_version_ += std::to_string(blpapiVersion.buildVersion());
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
    d_fields.push_back("EVT_TRADE_TIME_RT");
    
    d_fields.push_back("EVT_QUOTE_BID_PRICE_RT");
    d_fields.push_back("EVT_QUOTE_BID_SIZE_RT");
    d_fields.push_back("EVT_QUOTE_BID_TIME_RT");

    d_fields.push_back("EVT_QUOTE_ASK_PRICE_RT");
    d_fields.push_back("EVT_QUOTE_ASK_SIZE_RT");
    d_fields.push_back("EVT_QUOTE_ASK_TIME_RT");

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
    if (token.empty()) {
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

    if (m_cfg.d_topics.empty()) {
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
        m_sym_pxmul_map[symbol] = utils::SymbolMapReader::get().getTradableInfo(symbol)->_bbg_px_multiplier;
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
        resetTradeDay(utils::TimeUtil::cur_utc());
        m_should_run = true;

        // keep track of the unsubscribed topic, for logging purpose
        std::set<std::string> sub_topics (m_cfg.d_topics.begin(), m_cfg.d_topics.end());

        // stay and wait for the kill!
        if (sub_topics.empty()) {
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
                    const char* topic = getTopic(msg);
                    const char* symbol = getSymbol(topic);
                    try {
                        process(symbol, msg, pub);
                    } catch (const std::exception& e) {
                        logError("Exception in processing MarketDataEvents: %s", e.what());
                        logError("MarketData Message Dump: %s", printMsg(msg).c_str());
                        msg.print(std::cout);
                    }
                } else if (msg_name == "SubscriptionStreamsActivated") {
                    const char* topic = getTopic(msg);
                    const char* symbol = getSymbol(topic);
                    std::string pubname = m_cfg.d_pub;
                    std::string vname;
                    if (sub_topics.erase(topic)) {
                        addTCC(symbol);
                        if (m_cfg.d_topics_primary.find(topic)!=m_cfg.d_topics_primary.end()) {
                            pubname = "";
                            logInfo("BPipep subscribed to %s as primary (%s)", getSymbol(topic), pubname.c_str());
                        } else {
                            logInfo("BPipep subscribed to %s as secondary (%s)", getSymbol(topic), pubname.c_str());
                        }
                        pub.addWriter(symbol, vname, pubname);
                    } else {
                        msg.print(std::cout);
                        logError("duplicated or unknown subscription returned: %s, msg: %s", topic, printMsg(msg).c_str());
                    }
                    if (sub_topics.empty()) {
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
    static const Name nqbt("EVT_QUOTE_BID_TIME_RT");

    static const Name nqap("EVT_QUOTE_ASK_PRICE_RT");
    static const Name nqas("EVT_QUOTE_ASK_SIZE_RT");
    static const Name nqat("EVT_QUOTE_ASK_TIME_RT");

    static const Name ntpx("EVT_TRADE_PRICE_RT");
    static const Name ntsz("EVT_TRADE_SIZE_RT");
    static const Name nttm("EVT_TRADE_TIME_RT");

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

                const auto& t = msg.getElement(nqbt).getValueAsDatetime();
                /*
                logInfo("BID TIME - %u-%u-%u %u:%u:%u.%u (%d)",
                    t.year(), t.month(), t.day(), t.hours(), t.minutes(), t.seconds(),
                    t.microseconds(), (int)t.offset());
                */
                pub.getWriter(symbol, "")->updBBO(px,sz,true,getMicro(t));
            }
        } else if (strcmp(sub_type, "ASK")==0) {
            // get the nqap and nqas
            if (msg.hasElement(nqap,true) && msg.hasElement(nqas, true)) {
                double px = msg.getElement(nqap).getValueAsFloat64() * pxmul;
                unsigned sz = msg.getElement(nqas).getValueAsInt32();
                const auto& t = msg.getElement(nqat).getValueAsDatetime();
                /*
                logInfo("ASK TIME - %u-%u-%u %u:%u:%u.%u (%d)",
                    t.year(), t.month(), t.day(), t.hours(), t.minutes(), t.seconds(),
                    t.microseconds(), (int)t.offset());
                */
                pub.getWriter(symbol, "")->updBBO(px,sz,false,getMicro(t));
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

                const auto& bt = msg.getElement(nqbt).getValueAsDatetime();
                const auto& at = msg.getElement(nqat).getValueAsDatetime();
                const auto bmicro=getMicro(bt);
                const auto amicro=getMicro(at);
                pub.getWriter(symbol,"")->updBBO(bpx,bsz,true,bmicro);
                pub.getWriter(symbol,"")->updBBO(apx,asz,false,amicro);
                // TODO -
                //pub.getWriter(symbol,"")->updBBO(bpx,bsz,apx,asz,_MAX_(bmicro, amicro))
            }
        } else {
            logError("unknown QUOTE sub type %s: %s", sub_type, printMsg(msg).c_str());
        }
    } else if (strcmp(type, "TRADE")==0) {
        if (__builtin_expect(strcmp(sub_type, "NEW")==0,1)) {
            double px = msg.getElement(ntpx).getValueAsFloat64() * pxmul;
            unsigned sz = msg.getElement(ntsz).getValueAsInt32();
            if (__builtin_expect(px*sz != 0, 1)) {
                const auto& t = msg.getElement(nttm).getValueAsDatetime();
                const auto& t_micro=getMicro(t);
                if (msg.hasElement(ntcc,true)) {
                    const char* cc {msg.getElement(ntcc).getValueAsString()};
                    const int tt0 = checkTCC(symbol, cc); // trade type defined in cc_map
                    if (__builtin_expect(tt0 != -1, 1)) {
                        pub.getWriter(symbol,"")->updTrade(px,sz,t_micro,tt0);
                    } else {
                        // debug
                        /*
                        logInfo("Unknown trade code: %s, %s, %s", 
                                cc, symbol.c_str(), printMsg(msg).c_str());
                        */
                    }
                } else {
                    // markets such as ICE don't have condition code
                    // px and sz is already TSUM
                    pub.getWriter(symbol,"")->updTrade(px,sz,t_micro);
                }
            }
            /*
            logInfo("TRADE TIME - %u-%u-%u %u:%u:%u.%u (%d)",
                    t.year(), t.month(), t.day(), t.hours(), t.minutes(), t.seconds(),
                    t.microseconds(), (int)t.offset());
            */
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

uint64_t BPIPE_Thread::getMicro(const Datetime& dt) const {
    // check for change of trading day
    auto cur_utc=utils::TimeUtil::cur_utc();
    if (__builtin_expect(cur_utc >= m_utc0+24*3600, 0)) {
        resetTradeDay(cur_utc);
    }

    char tbuf[64];
    sprintf(tbuf, "%s-%02u:%02u:%02u",m_trade_day_yyyymmdd.c_str(), 
                   dt.hours(), dt.minutes(), dt.seconds());
    time_t utc1 = utils::TimeUtil::string_to_frac_UTC(tbuf,0,"%Y%m%d-%H:%M:%S",true);

    // offset set?
    if (dt.hasParts(DatetimeParts::OFFSET)) {
        utc1+=(dt.offset()*60);
    }
    if (__builtin_expect(cur_utc<m_utc0,0)) {
        // case when cur_utc is after 5pm but before 6pm
        // with a trading day that is tomorrow
        utc1 -= (24*3600);
    } else{
        // allow cur_utc couple days ahead of m_utc0
        utc1= (utc1-m_utc0)%(24*3600)+m_utc0;
    }
    return (uint64_t)utc1*1000ULL*1000ULL+(uint64_t)dt.microseconds();
}

void BPIPE_Thread::resetTradeDay(time_t cur_utc) const {
        //snap forward
        m_trade_day_yyyymmdd = utils::TimeUtil::tradingDay(cur_utc,-6,0,17,0,0,2);
        m_utc0 = utils::TimeUtil::startUTC(cur_utc,-6,0,17,0,0,2);
}

/*
 * 6/27/2023 - disable the CC initially, TODO - add one-by-one with TickData matching
 */
void BPIPE_Thread::addTCC(const std::string& symbol) {
    static std::map<std::string, int> cc_map = {
        {"TSUM", 0}, 
        {"ST",10}, {"CT",11},  {"IO",12},  {"NDOO",13}, {"NDOT",14}, //default_imp
        //{"ST",0}, {"CT",0},  {"IO",0},  {"NDOO",0}, {"NDOT",0}, //default_imp as normal trade
        {"BT",20}, {"BL",21},  {"SBT",22}, {"SBL",23},               // ice_cc
        {"EFS",30},{"VOLA",31},{"TES",32}, {"EFPF",33},{"EFPI",34},{"OPEN",35} // eurex
    };

    // default
    static const std::set<std::string> default_cc{"TSUM"};
    static const std::set<std::string> default_imp_cc{"ST","CT","IO","NDOO","NDOT"};
    //static const std::set<std::string> default_imp_cc;

    // CME venues and trade codes
    static const std::set<std::string> cme_set {"CBT", "CEC", "CME", "NYM", "MGE"};
    static const std::set<std::string> cme_cc; // nothing to add

    // ICE venus and trade codes
    static const std::set<std::string> ice_set {"IFUS", "IFEU", "IFLL", "IFLX", "IFCA"};
    static const std::set<std::string> ice_cc {"BT","BL","SBT", "SBL"};
    //static const std::set<std::string> ice_cc;

    // Eurex venus and trade codes
    static const std::set<std::string> eurex_set {"EUR","EOP"};
    static const std::set<std::string> eurex_cc {"EFS","VOLA","TES","EFPF","EFPI", "OPEN"};
    //static const std::set<std::string> eurex_cc;

    if (m_symbol_tcc.find(symbol) != m_symbol_tcc.end()) {
        logInfo("addTCC: %s already added!", symbol.c_str());
        return;
    }
    std::set<std::string> cc_set = default_cc;
    try {
        const auto& mts_venue = utils::SymbolMapReader::get().getByBbgId(symbol)->_venue;
        if (cme_set.find(mts_venue) != cme_set.end()) {
            // CME venues takes only TSUM, ignore all the block/imp trades
            // TODO - CME does have impled trades reported from bpipe, but 
            //        Tickdata (historical data) doesn't report them.
            //        Add them back for a new colume or a new data vendor
            cc_set.insert(cme_cc.begin(), cme_cc.end());
        } else if (ice_set.find(mts_venue) != ice_set.end()) {
            // implied + block for ICE venues
            cc_set.insert(default_imp_cc.begin(), default_imp_cc.end());
            cc_set.insert(ice_cc.begin(), ice_cc.end());
        } else if (eurex_set.find(mts_venue) != eurex_set.end()) {
            // implied + various derived
            cc_set.insert(default_imp_cc.begin(), default_imp_cc.end());
            cc_set.insert(eurex_cc.begin(), eurex_cc.end());
        } else {
            logError("cannot find venue set for venue %s, trades with any condition codes other than default will not be parsed/updated", mts_venue.c_str());
        }
    } catch (const std::exception& e) {
        logError("bpipe thread cannot add trade code set for %s: %s", 
               symbol.c_str(), e.what());
        return;
    }

    // populate tcc from set
    auto& sym_cc_map=m_symbol_tcc[symbol];

    // trace str
    std::string cc0;
    for (const auto& c : cc_set) {
        sym_cc_map.emplace(c, cc_map[c]);
        cc0 += c;
        cc0 += " ";
    }
    logInfo("addTCC: %s -> %s",symbol.c_str(), cc0.c_str());
}

int BPIPE_Thread::checkTCC(const std::string& symbol, const char* cc) const {
    const auto iter=m_symbol_tcc.find(symbol);
    if (__builtin_expect(iter==m_symbol_tcc.end(), 0)) {
        logError("bpipe thread found unknown symbol (%s) when checking trade code (%s)", symbol.c_str(), cc);
        return -1;
    }
    // ge the first in case multiple codes in cc
    const auto& c(utils::CSVUtil::read_line(cc));
    const auto& cmap(iter->second);
    const auto cctp = cmap.find(c[0]);
    if (__builtin_expect(cctp!=cmap.end(),1)) {
        return cctp->second;
    }
    return -1;
}

} // namespace bpipe
} // namespace tp
