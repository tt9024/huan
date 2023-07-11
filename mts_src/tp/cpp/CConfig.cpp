#include "CConfig.h"


using namespace Mts::Core;


CConfig &CConfig::getInstance() {

    static CConfig objTheInstance;
    return objTheInstance;
}


const std::string &CConfig::getLogFileDirectory() const {

    return m_strLogFileDirectory;
}


unsigned int CConfig::getLogBufferSize() const {

    return m_iLogFileBufferSize;
}


unsigned int CConfig::getLogIntervalSec() const {

    return m_iLogFileLogIntervalSecs;
}


void CConfig::setEnginePath(const std::string &path) {
    m_engine_path = path;
    m_config_path = PathJoin(m_engine_path, CONFIG_PATH);
}

void CConfig::setLogFileDirectory(const std::string &strLogFileDirectory) {
    m_strLogFileDirectory = PathJoin(m_engine_path, strLogFileDirectory);
}


void CConfig::setLogBufferSize(unsigned int iBufferSize) {

    m_iLogFileBufferSize = iBufferSize;
}


void CConfig::setLogIntervalSec(unsigned int iIntervalSecs) {

    m_iLogFileLogIntervalSecs = iIntervalSecs;
}


const std::string &CConfig::getSQLServer() const {

    return m_strSQLServer;
}


const std::string &CConfig::getSQLDatabase() const {

    return m_strSQLDatabase;
}


void CConfig::setSQLServer(const std::string &strSQLServer) {

    m_strSQLServer = strSQLServer;
}


void CConfig::setSQLDatabase(const std::string &strSQLDatabase) {

    m_strSQLDatabase = strSQLDatabase;
}


const std::string &CConfig::getSMTPServer() const {

    return m_strSMTPServer;
}


const std::string &CConfig::getFromEmailAddr() const {

    return m_strFromEmailAddr;
}


const std::string &CConfig::getToEmailAddr() const {

    return m_strToEmailAddr;
}


bool CConfig::isEmailEnabled() const {

    return m_bEnableEmailer;
}


void CConfig::setSMTPServer(const std::string &strSMTPServer) {

    m_strSMTPServer = strSMTPServer;
}


void CConfig::setFromEmailAddr(const std::string &strFromEmailAddr) {

    m_strFromEmailAddr = strFromEmailAddr;
}


void CConfig::setToEmailAddr(const std::string &strToEmailAddr) {

    m_strToEmailAddr = strToEmailAddr;
}


void CConfig::setEnableEmailer(bool bEnableEmailer) {

    m_bEnableEmailer = bEnableEmailer;
}

unsigned int CConfig::getEngineID() const {

    return m_iEngineID;
}


void CConfig::setEngineID(unsigned int iEngineID) {

    m_iEngineID = iEngineID;
}


const std::string &CConfig::getRecoveryFileDirectory() const {

    return m_strRecoveryFileDirectory;
}

const std::string CConfig::getConfigFile(const std::string &filename) const {
    return PathJoin(m_config_path, filename);
}

const std::string CConfig::getAlgoConfigFile(const std::string &algoname) const {
    return PathJoin(PathJoin(m_config_path, ALGO_PATH), algoname);
}

const std::string CConfig::getCalendarFile(const std::string &filename) const {
    return PathJoin(PathJoin(m_config_path, CALENDAR_PATH), filename);
}

const std::string CConfig::getEngineConfigFile() const {
    return PathJoin(m_config_path, ENGINE_CONFIG_FILE);
}

const std::string CConfig::getDropCopyFIXConfigFile() const {
    return PathJoin(m_config_path, DROPCOPY_FIX_CONFIG);
}

const std::string CConfig::getDropCopySessionConfigFile() const {
    return PathJoin(m_config_path, DROPCOPY_SESSION_CONFIG);
}

void CConfig::setRecoveryFileDirectory(const std::string &strRecoveryFileDirectory) {
    m_strRecoveryFileDirectory = PathJoin(m_engine_path, strRecoveryFileDirectory);
}
