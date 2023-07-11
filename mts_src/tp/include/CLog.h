#ifndef CLOG_HEADER

#define CLOG_HEADER

#include <string>
#include <fstream>
#include <exception>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "IRunnable.h"
#include "CCircCharBuffer.h"

namespace Mts {
    namespace Log {
        class CLog : Mts::Thread::IRunnable
        {
        public:
            CLog();

            CLog(const std::string &strLogFile);

            // IRunnable implementation
            void run();

            void operator()();

            bool startLogging();

            void log(const std::string &strData);

            bool isStopped() const;

            bool stopLogging();

            void setLogFile(const std::string &strLogFile);

        private:
            typedef CCircCharBuffer Buffer;

            void flushBuffer(Buffer &objBuffer);

        private:
            enum
            {
                BUFFER_SIZE = 512000
            };

            std::string m_strLogFile;
            std::ofstream m_objOutputStream;
            boost::shared_ptr<boost::thread> m_ptrThread;
            bool m_bStopped;
            unsigned int m_iLogIntervalMillis;

            boost::mutex m_Mutex;
            Buffer m_BufferA;
            Buffer m_BufferB;
            Buffer *m_ptrCurrentBuffer;
        };
    }
}

#endif

