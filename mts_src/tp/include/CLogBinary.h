#ifndef CLOGBINARY_HEADER

#define CLOGBINARY_HEADER

#include <vector>
#include <string>
#include <stdio.h>

namespace Mts {
    namespace Log {
        // simple, single threaded, non-buffered, binary data logging
        class CLogBinary
        {
        public:
            CLogBinary();

            ~CLogBinary();

            void openLog(const std::string &strFileName);

            void openLogReadOnly(const std::string &strFileName);

            void closeLog();

            long getSize() const;

            template<typename T>
            void writeToFile(const T &objData);

            template<typename T>
            void readFromFile(std::vector<T> &objDate);

            template<typename T>
            void writeToFileHomogenous(const T &objData);

            template<typename T>
            void readFromFileHomogenous(T &objDataItem,
                    long iRecNum);

        private:
            FILE *m_ptrFile;
        };
    }
}

#include "CLogBinary.hxx"

#endif

