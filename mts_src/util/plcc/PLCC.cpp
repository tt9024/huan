#include "plcc/PLCC.hpp"

namespace utils {

const char* PLCC::LoggerConfigKey = DefaultLoggerConfigKey;
const char* PLCC::ConfigFilePath =  DefaultConfigFilePath;
PLCC* PLCC::default_plcc=nullptr;

const char* PLCC::getConfigPath() {
    return ConfigFilePath;
};

void PLCC::setConfigPath(const char* cfg_path) {
    ConfigFilePath = cfg_path;
}

const std::string PLCC::getLogFileName(std::string logfile,
        std::string instname) 
{
    // use the trading day instead of calendar day
    const auto trading_ymd = TimeUtil::tradingDay(0, -6, 0, 17, 30, 0, 2);

    return logfile+
            (instname.size()>0?(std::string("_")+instname):std::string(""))+
            std::string("_")+trading_ymd+
            std::string(".txt");
}

void PLCC::ToggleTest(bool is_on) {
    if (PLCC::default_plcc) {
        free (PLCC::default_plcc);
        PLCC::default_plcc = nullptr;
    }
    if (is_on) {
        setConfigPath(nullptr);
    } else {
        setConfigPath(DefaultConfigFilePath);
    }
    PLCC::default_plcc = new PLCC(getConfigPath(), "");
}


PLCC& PLCC::instance(const char* instname) {
    // note this is NOT thread safe
    static std::map<std::string, PLCC*>plcc_map;

    if (__builtin_expect(!default_plcc, 0)) {
        default_plcc=new PLCC(getConfigPath(), "");
        plcc_map[""] = default_plcc;
        //return *default_plcc;
    }
    if (__builtin_expect(!instname, 1)) {
            return *default_plcc;
    }
    const std::string inststr=std::string(instname);
    auto iter=plcc_map.find(inststr);
    if (iter !=plcc_map.end()) {
            return *(iter->second);
    }
    PLCC* inst=new PLCC(getConfigPath(), inststr);
    plcc_map[inststr]=inst;
    return *inst;
}

PLCC::PLCC(const char* configFileName, const std::string& instname) :
    ConfigureReader(configFileName),
    FileLogger(configFileName? 
       getLogFileName(get<std::string>(LoggerConfigKey, nullptr, "stdout"),instname).c_str() : 
       "stdout"),
    m_configFileName(configFileName?configFileName:"")
{}

PLCC::~PLCC() {}
}
