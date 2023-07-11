#include "CLogText.h"
#include "CConfig.h"
#include "CApplicationLog.h"


using namespace Mts::Log;


CLogText::CLogText(const std::string &strLogFile) {

    bool bRet = openLogFile(strLogFile);
}


CLogText::~CLogText() {

    closeLogFile();
}


bool CLogText::openLogFile(const std::string &strLogFile) {

    try {

        char szBuffer[255];
        sprintf(szBuffer, "%s/%s", Mts::Core::CConfig::getInstance().getLogFileDirectory().c_str(), strLogFile.c_str());

        m_objOutputStream.open(szBuffer, std::ios::out);

        return m_objOutputStream.is_open();
    }
    catch (std::exception &e) {

        AppLogError(e.what());
        return false;
    }
}


void CLogText::flush() {

    m_objOutputStream.flush();
}


void CLogText::closeLogFile() {

    try {

        m_objOutputStream.flush();
        m_objOutputStream.close();
    }
    catch (std::exception &e) {

        AppLogError(e.what());
    }
}


void CLogText::log(const std::string &strData) {

    try {

        m_objOutputStream << strData;
    }
    catch (std::exception &e) {

        AppLogError(e.what());
        throw;
    }
}


void CLogText::log(const std::vector<std::string> &objData) {

    try {

        for (int i = 0; i != objData.size(); ++i) {

            m_objOutputStream << objData[i] << std::endl;
        }
    }
    catch (std::exception &e) {

        AppLogError(e.what());
        throw;
    }
}

