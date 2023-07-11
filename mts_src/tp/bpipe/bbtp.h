#pragma once

#include <blpapi_defs.h>
#include <blpapi_correlationid.h>
#include <blpapi_element.h>
#include <blpapi_event.h>
#include <blpapi_message.h>
#include <blpapi_session.h>
#include <blpapi_subscriptionlist.h>
#include <blpapi_tlsoptions.h>
#include <blpapi_versioninfo.h>

#include <unordered_map>
#include <set>

// for tp publish
#include "md_bar.h"

#define BPIPE_BOOK_LEVEL 1

namespace tp {
namespace bpipe {

struct BPIPE_Config
{
public:
    std::string              d_pub;
    std::vector<std::string> d_hosts;
    int                      d_port;
    std::string              d_service;
    std::vector<std::string> d_topics;
    std::set<std::string>    d_topics_primary;
    std::vector<std::string> d_fields;
    std::vector<std::string> d_options;  // subscription options, not used
    std::string              d_authOptions;  // app and user
    std::string              d_clientCredentials;
    std::string              d_clientCredentialsPassword;
    std::string              d_trustMaterial;

    explicit BPIPE_Config(const char* config_file = nullptr);
    BloombergLP::blpapi::TlsOptions getTlsOptions() const;
    BloombergLP::blpapi::SessionOptions getSessionOptions() const;

    [[nodiscard]] std::string toString() const;
    [[nodiscard]] std::string blapiVersion() const { return blpapi_version_; };
private:
    void setFields();
    void setTopics();

    std::string blpapi_version_{};
    void setBlapiVersion();
};

class BPIPE_Thread
{
public:
    explicit BPIPE_Thread(const char* config_file = nullptr);
    void run();
    void kill();

private:
    const BPIPE_Config m_cfg;
    volatile bool m_should_run;
    std::unordered_map<std::string, double> m_sym_pxmul_map;
    mutable std::string m_trade_day_yyyymmdd; // current trading day
    mutable time_t m_utc0;  // starting utc of current trading day
                                // set as previous day's 6pm NewYork time

    // symbol-->set of trade condition codes, for inclusion decision
    std::unordered_map<std::string, std::map<std::string,int>> m_symbol_tcc; // map of symbol -> {TCC(str) -> TYPE(int)}

    void connect(BloombergLP::blpapi::Session& session);
    void process(const std::string& symbol, 
                 const BloombergLP::blpapi::Message& msg, 
                 md::MD_Publisher<BPIPE_BOOK_LEVEL>& pub);
    bool authorize(const BloombergLP::blpapi::Service &authService,
                         BloombergLP::blpapi::Identity *subscriptionIdentity,
                         BloombergLP::blpapi::Session *session);

    bool checkFailure(const BloombergLP::blpapi::Event::EventType& eventType, const BloombergLP::blpapi::Message& message) const;
    const char* getTopic(const BloombergLP::blpapi::Message& msg) const;
    const char* getSymbol(const char* topic) const;
    void resetTradeDay(time_t cur_utc) const;
    uint64_t getMicro(const BloombergLP::blpapi::Datetime& dt) const;
    std::string printMsg(const BloombergLP::blpapi::Message& msg) const;

    // called upon subscription to update the m_symbol_tcc
    void addTCC(const std::string& symbol);
    int checkTCC(const std::string& symbol, const char* cc) const;
};

} // namespace bpipe
} // namespace tp
