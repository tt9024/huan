#ifndef CLOGBINARYBUFFERED_HEADER

#define CLOGBINARYBUFFERED_HEADER

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <vector>
#include <stdio.h>
#include "IRunnable.h"
#include "CCircCharBuffer.h"

namespace Mts {
    namespace Log {
        // double buffered, multi-threaded, binary data logging
        class CLogBinaryBuffered : Mts::Thread::IRunnable
        {
        public:
            CLogBinaryBuffered();

            CLogBinaryBuffered(const std::string &strLogFile,
                    unsigned int iBufferSize,
                    unsigned int iLogIntervalMillis);

            // IRunnable implementation
            void run();

            void operator()();

            bool startLogging();

            template<typename T>
            void log(const T &objData);

            bool isStopped() const;

            bool stopLogging();

            void setLogFile(const std::string &strLogFile);

            template<typename T>
            void readFromFile(std::vector<T> objDate);

        private:
            typedef CCircCharBuffer Buffer;

            void flushBuffer(Buffer &objBuffer);

        private:
            enum
            {
                BUFFER_SIZE = 16384
            };

            std::string m_strLogFile;
            FILE *m_ptrFile;
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

#include "CLogBinaryBuffered.hxx"

#endif

