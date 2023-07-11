#include <boost/filesystem.hpp>
#include "CLogUnorderedMap.h"
#include "CConfig.h"
#include "CApplicationLog.h"


using namespace Mts::Log;


CLogUnorderedMap::CLogUnorderedMap(const std::string &strLogFile)
        : m_strLogFile(strLogFile) {

}


bool CLogUnorderedMap::save(std::unordered_map<std::string, double> &objNumericMap,
        std::unordered_map<std::string, std::string> &objStringMap) {

    try {

        std::ofstream os(Mts::Core::CConfig::getInstance().getLogFileDirectory() + "\\" + m_strLogFile,
                         std::ios::binary | std::ios::out);

        boost::archive::binary_oarchive oa(os);

        oa << objNumericMap;
        oa << objStringMap;

        os.close();

        return true;
    }
    catch (std::exception &err) {

        AppLogError(err.what());
        return false;
    }
}


bool CLogUnorderedMap::load(std::unordered_map<std::string, double> &objNumericMap,
        std::unordered_map<std::string, std::string> &objStringMap) {

    try {

        objNumericMap.clear();
        objStringMap.clear();

        std::string strLogFile = Mts::Core::CConfig::getInstance().getLogFileDirectory() + "\\" + m_strLogFile;

        if (boost::filesystem::exists(strLogFile) == false) {

            return false;
        }

        std::ifstream is(strLogFile, std::ios::binary | std::ios::in);
        boost::archive::binary_iarchive ia(is);

        ia >> objNumericMap;
        ia >> objStringMap;

        is.close();

        return true;
    }
    catch (std::exception &err) {

        AppLogError(err.what());
        return false;
    }
}

