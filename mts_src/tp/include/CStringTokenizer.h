#ifndef CSTRINGTOKENIZER_HEADER

#define CSTRINGTOKENIZER_HEADER

#include <vector>

namespace Mts {
    namespace Core {
        class CStringTokenizer
        {
        public:
            std::vector<std::string> splitString(const std::string &strDelimitedData,
                    const std::string &strDelimiter) const;
        };
    }
}

#endif

