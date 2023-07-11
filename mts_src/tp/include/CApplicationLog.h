#pragma once

#include "CMTSLogger.hxx"
#include <memory>

#include "plcc/PLCC.hpp"

#define AppInfo(...) logInfo(__VA_ARGS__)
#define AppError(...) logError(__VA_ARGS__)

#define AppLog(...) logInfo(__VA_ARGS__ )

#define AppLogInfo(...) AppLog(__VA_ARGS__)
#define AppLogError(...) AppError(__VA_ARGS__)

// micros for logging to the APP log, the Engine_YYYYDDMM.txt
//#define AppInfo(...) LogInfo((Mts::Log::CApplicationLog::getInstance()), __VA_ARGS__)
//#define AppError(...) LogError((Mts::Log::CApplicationLog::getInstance()), __VA_ARGS__)

// legacy interfaces, takes only one string
//#define AppLog(...) LogInfo((Mts::Log::CApplicationLog::getInstance()), __VA_ARGS__ )
//#define AppLogInfo(...) AppLog(__VA_ARGS__)
//#define AppLogError(...)  LogError((Mts::Log::CApplicationLog::getInstance()),  __VA_ARGS__ )



namespace Mts {
    namespace Log {
        class CApplicationLog : public MTSFileLogger
        {
        public:
            static CApplicationLog &getInstance();

            static CApplicationLog &algoLog(const std::string &algoName);

            // Singleton interface
            CApplicationLog(const CApplicationLog &objRhs) = delete;

            CApplicationLog &operator=(const CApplicationLog &objRhs) = delete;

        private:
            explicit CApplicationLog(const std::string &filename);
        };
    }
}
