#ifndef CLOGUNORDEREDMAP_HEADER

#define CLOGUNORDEREDMAP_HEADER

#include <unordered_map>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <fstream>
#include <iostream>

namespace Mts {
    namespace Log {
        class CLogUnorderedMap
        {
        public:
            CLogUnorderedMap(const std::string &strLogFile);

            bool save(std::unordered_map<std::string, double> &objNumericMap,
                    std::unordered_map<std::string, std::string> &objStringMap);

            bool load(std::unordered_map<std::string, double> &objNumericMap,
                    std::unordered_map<std::string, std::string> &objStringMap);

        private:
            std::string m_strLogFile;
        };
    }
}

#endif

