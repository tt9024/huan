#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <string>
#include <stdexcept>

#include "Logger.hpp"
#include "ConfigureReader.hpp"
#include "csv_util.h"

#include <map>
#include <cstring>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/')+1 : __FILE__)

// ISO C++ does not permit named variadic macros -Werror=variadic-macros

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"

#ifndef NUMPY_COMPILE

#define logDebug(a...)
#define logInfo(a...) utils::PLCC::instance().logInfo(__FILENAME__,__LINE__,a)
#define logError(a...) utils::PLCC::instance().logError(__FILENAME__,__LINE__,a)

#define plcc_getInt(a...) utils::PLCC::instance().get<int>(a)
#define plcc_getDouble(a...) utils::PLCC::instance().get<double>(a)
#define plcc_getString(a...) utils::PLCC::instance().get<std::string>(a)
#define plcc_getStringArr(a...) utils::PLCC::instance().getArr<std::string>(a)

#else

#define logDebug(a...)
#define logInfo(a...) 
#define logError(a...) 

#define plcc_getInt(a...)  0
#define plcc_getDouble(a...) 0.0
#define plcc_getString(a...)  std::string("")
#define plcc_getStringArr(a...) std::vector<std::string>()

#endif

#pragma GCC diagnostic pop /* "-Werror=variadic-macros" */

#define PRICE_PRECISION 8
#define PriceString(px)  utils::CSVUtil::printDouble((px),  PRICE_PRECISION)
#define PriceCString(px) utils::CSVUtil::printDouble((px),  PRICE_PRECISION).c_str()

#define PnlString(pnl)   utils::CSVUtil::printPnl(pnl)
#define PnlCString(pnl)  utils::CSVUtil::printPnl(pnl).c_str()
#define PnlStringColor(pnl)  utils::CSVUtil::printPnlColor(pnl)

#define _MAX_(x, y) ((x)>=(y)?(x):(y))
#define _MIN_(x, y) ((x)<=(y)?(x):(y))
#define _ABS_(x)    ((x)>=0 ?(x):(-x))

#define DefaultLoggerConfigKey "Logger"
#define DefaultConfigFilePath  "config/main.cfg"

namespace utils {

class PLCC : public ConfigureReader, public FileLogger {
public:
    static const char* LoggerConfigKey;
    static const char* ConfigFilePath;

    static const char* getConfigPath();
    static void setConfigPath(const char* cfg_path);

    // setup without main cfg, logging to stdout
    static void ToggleTest(bool is_on=true);
    static const std::string getLogFileName(std::string logfile, std::string instname);
    static PLCC& instance(const char* instname=NULL);

private:
    explicit PLCC(const char* configFileName, const std::string& instname);
    ~PLCC();
    static PLCC* default_plcc;
    std::string m_configFileName;
};
}
