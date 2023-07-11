#pragma once

#include "stdio.h"
#include "stdarg.h"
#include "stdlib.h"
#include <iostream>
#include <string>
#include <map>
#include "CDateTime.h"

#define LogInfo(logger, ...) logger.log("INFO",__FILE__,__LINE__,__VA_ARGS__)
#define LogError(logger, ...) logger.log("ERR",__FILE__,__LINE__,__VA_ARGS__)

#include "plcc/PLCC.hpp"

namespace Mts {
    namespace Log {
        template<typename Logger>
        class CMTSLog
        {
        public:
            explicit CMTSLog(const std::string &logname) :
                    m_sLogname(logname), m_Logger(logname) {
            }

            void log(const char *level, const char *file, int line, const char *fmt, ...) {
                va_list ap;
                va_start(ap, fmt);
                const size_t bufSize = 2048 - 1;
                char buf[bufSize + 1];
                size_t linesz = prepareLogString(level, file, line, buf, bufSize);
                linesz += vsnprintf(buf + linesz, bufSize - linesz, fmt, ap);
                buf[linesz++] = '\n';
                logRaw(buf, linesz);
            }

            // legacy
            void log(const char *level, const char *file, int line, const std::string &str) {
                const size_t bufSize = 2048 - 1;
                char buf[bufSize + 1];
                size_t linesz = prepareLogString(level, file, line, buf, bufSize);
                linesz += snprintf(buf + linesz, bufSize - linesz, str.c_str());
                buf[linesz++] = '\n';
                logRaw(buf, linesz);
            }

            void logRaw(const char *line, size_t lineSize) {
                m_Logger.write(line, lineSize);
            }

            // legacy interface
            void log(const std::string &str) {
                log("LEVEL", "FILE", 0, str);
            }

        protected:
            const std::string m_sLogname;
            Logger m_Logger;

            size_t prepareLogString(const char *level, const char *file, int line, char *buf, size_t bufSize) {
                size_t sz = Mts::Core::CDateTime::now().toStringLogger(buf, bufSize - 1);
                return sz + snprintf(buf + sz, bufSize - sz, ",%s,%s:%d,", level, file, line);
            }

        };

        class FileLogger
        {
        public:
            FileLogger(const std::string &filename) :
                    m_name(filename), m_fp(0) {
                m_fp = fopen(m_name.c_str(), "a+");
                if (!m_fp) {
                    throw std::runtime_error(filename + " not found in FileLogger!");
                }
            }

            ~FileLogger() {
                if (m_fp) {
                    fclose(m_fp);
                    m_fp = 0;
                }
            }

            void write(const char *buf, const size_t bufSize) {
                fwrite(buf, 1, bufSize, m_fp);
                fflush(m_fp);
            }

        private:
            const std::string m_name;
            FILE *m_fp;
        };

        using MTSFileLogger = CMTSLog<FileLogger>;

    }
}
