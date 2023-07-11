#include "CApplicationLog.h"
#include "CConfig.h"

using namespace Mts::Log;

CApplicationLog::CApplicationLog(const std::string &filename) :
        MTSFileLogger(filename) {};

CApplicationLog &CApplicationLog::getInstance() {
    static std::unique_ptr<CApplicationLog> s_pInst;
    if (!s_pInst) {
        std::string filename;
        try {
            filename = Mts::Core::PathJoin(
                    Mts::Core::CConfig::getInstance().getLogFileDirectory(),
                    std::string("Engine_") + Mts::Core::CDateTime::now().toStringYMD()
            );
            s_pInst.reset(new CApplicationLog(filename));
        }
        catch (const std::exception &e) {
            std::cout << "Problem creating AppLog: FilePath: " << filename << "\n" << e.what() << std::endl;
            throw e;
        }
    }
    return *s_pInst;
}

CApplicationLog &CApplicationLog::algoLog(const std::string &algoName) {
    static std::map<std::string, std::unique_ptr<CApplicationLog> > s_mLogs;
    auto iter = s_mLogs.find(algoName);
    if (iter != s_mLogs.end()) {
        return *iter->second;
    }
    std::string filename;
    try {
        filename = Mts::Core::PathJoin(
                Mts::Core::CConfig::getInstance().getLogFileDirectory(),
                algoName + std::string("_") + Mts::Core::CDateTime::now().toStringYMD()
        );
        s_mLogs[algoName].reset(new CApplicationLog(filename));
        return *s_mLogs[algoName];
    }
    catch (const std::exception &e) {
        std::cout << "Problem creating AlgoLog! AlgoName: "
                  << algoName << ", FilePath: " << filename << "\n"
                  << e.what() << std::endl;
        throw e;
    }
}


