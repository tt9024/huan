#ifndef CXMLPARSER_HEADER

#define CXMLPARSER_HEADER

#include <boost/property_tree/ptree.hpp>

namespace Mts {
    namespace XML {
        class CXMLParser
        {
        public:
            CXMLParser();

            boost::property_tree::ptree parseXMLFile(const std::string &strXMLFile) const;

        private:
        };
    }
}

#endif

