#include <iostream>
#include "CLog.h"
#include "CConfig.h"
#include "CMtsException.h"
#include "CApplicationLog.h"


using namespace Mts::Log;


CLog::CLog()
        : m_BufferA(Mts::Core::CConfig::getInstance().getLogBufferSize()),
          m_BufferB(Mts::Core::CConfig::getInstance().getLogBufferSize()) {

    m_iLogIntervalMillis = Mts::Core::CConfig::getInstance().getLogIntervalSec() * 1000;
    m_ptrCurrentBuffer = &m_BufferA;
}


CLog::CLog(const std::string &strLogFile)
        : m_strLogFile(strLogFile),
          m_BufferA(Mts::Core::CConfig::getInstance().getLogBufferSize()),
          m_BufferB(Mts::Core::CConfig::getInstance().getLogBufferSize()) {

    m_iLogIntervalMillis = Mts::Core::CConfig::getInstance().getLogIntervalSec() * 1000;
    m_ptrCurrentBuffer = &m_BufferA;
}


void CLog::setLogFile(const std::string &strLogFile) {

    m_strLogFile = strLogFile;
}


void CLog::run() {

    m_ptrThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::ref(*this)));
}


void CLog::operator()() {

    while (!isStopped()) {

        Buffer *ptrBufferToFlush = NULL;

        {
            boost::mutex::scoped_lock lock(m_Mutex);
            ptrBufferToFlush = m_ptrCurrentBuffer;
            m_ptrCurrentBuffer = (m_ptrCurrentBuffer == &m_BufferA) ? &m_BufferB : &m_BufferA;
        }

        flushBuffer(*ptrBufferToFlush);

        // flush every 5 secs
        boost::this_thread::sleep(boost::posix_time::milliseconds(m_iLogIntervalMillis));
    }
}


void CLog::flushBuffer(Buffer &objBuffer) {

    try {

        while (objBuffer.getNumItems() != 0) {

            unsigned int iNumBytes = 0;
            objBuffer.pop(reinterpret_cast<char *>(&iNumBytes), sizeof(iNumBytes), false);

            if (iNumBytes > BUFFER_SIZE) {

                char szError[255];
                sprintf(szError, "%s text buffer too small, item size = %u, buffer size = %d", m_strLogFile.c_str(),
                        iNumBytes, BUFFER_SIZE);
                throw Mts::Exception::CMtsException(szError);
            }

            char szBuffer[BUFFER_SIZE];
            objBuffer.pop(szBuffer, iNumBytes, false);

            m_objOutputStream << szBuffer << std::endl;
        }
    }
    catch (std::exception &e) {

        std::cout << e.what() << std::endl;
    }
}


bool CLog::startLogging() {

    try {

        m_objOutputStream.open(m_strLogFile.c_str(), std::ios::out);

        if (m_objOutputStream.is_open()) {

            m_bStopped = false;
            run();

            return true;
        } else
            return false;
    }
    catch (std::exception &e) {

        AppLogError(e.what());
        return false;
    }
}


void CLog::log(const std::string &strData) {

    try {

        boost::mutex::scoped_lock lock(m_Mutex);
        unsigned int iNumBytes = strData.length() + 1;

        // check the buffer has enough space for both string length plus the string itself
        bool bOverRun = m_ptrCurrentBuffer->checkForBufferOverRun(iNumBytes + sizeof(iNumBytes));

        if (bOverRun == true)
            return;

        m_ptrCurrentBuffer->push(reinterpret_cast<const char *>(&iNumBytes), sizeof(iNumBytes));
        m_ptrCurrentBuffer->push(strData.c_str(), iNumBytes);
    }
    catch (std::exception &e) {

        AppLogError(e.what());
        throw;
    }
}


bool CLog::isStopped() const {

    return m_bStopped;
}


bool CLog::stopLogging() {

    try {
        m_bStopped = true;
        m_ptrThread->join();
        flushBuffer(*m_ptrCurrentBuffer);
        m_objOutputStream.close();

        return true;
    }
    catch (std::exception &e) {

        AppLogError(e.what());
        return false;
    }
}


