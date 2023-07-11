#ifndef CEXCHANGEFACTORY_HEADER

#define CEXCHANGEFACTORY_HEADER

#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>
#include "CExchange.h"
#include "CDropCopyFIX_TT.h"

namespace Mts {
    namespace Exchange {
        class CExchangeFactory
        {
        public:
            static CExchangeFactory &getInstance();

            CExchangeFactory() = default;

            CExchangeFactory(const CExchangeFactory &objRhs) = delete;

            boost::shared_ptr<CExchange> createExchange(const std::string &strExchangeDefXML);

            // return an instance of dropcopy session
            boost::shared_ptr<CDropCopyFIX_TT> createDropCopy(const std::string &strDropCopyXml);

        private:
            boost::shared_ptr<CExchange> createFillSimulator(const boost::property_tree::ptree &objPTree);

            boost::shared_ptr<CExchange> createExchangeFIX_TT(const boost::property_tree::ptree &objPTree);
        };
    }
}

#endif

