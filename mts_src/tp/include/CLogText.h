#ifndef CLOGTEXT_HEADER

#define CLOGTEXT_HEADER

#include <string>
#include <fstream>
#include <exception>
#include <vector>

namespace Mts {
    namespace Log {
        class CLogText
        {
        public:
            CLogText(const std::string &strLogFile);

            ~CLogText();

            void log(const std::vector<std::string> &objData);

            void log(const std::string &strData);

            void flush();

        private:
            bool openLogFile(const std::string &strLogFile);

            void closeLogFile();

        private:
            std::ofstream m_objOutputStream;
        };
    }
}

#endif

