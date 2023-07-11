#ifndef CALGORITHMFACTORY_HEADER

#define CALGORITHMFACTORY_HEADER

#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>
#include "CAlgorithm.h"

namespace Mts {
    namespace Algorithm {
        class CAlgorithmFactory
        {
        public:
            static CAlgorithmFactory &getInstance();

            CAlgorithmFactory() = default;

            CAlgorithmFactory(const CAlgorithmFactory &objRhs) = delete;

            boost::shared_ptr<CAlgorithm> createAlgorithm(const std::string &strAlgoDefXML);

        private:
            // one method per algorithm the factory can create
            boost::shared_ptr<CAlgorithm> createAlgorithmTemplate(const boost::property_tree::ptree &objPTree);

            boost::shared_ptr<CAlgorithm> createAlgorithmFIXTest(const boost::property_tree::ptree &objPTree);

            void loadRiskLimits(const boost::property_tree::ptree &objPTree,
                    boost::shared_ptr<Mts::Algorithm::CAlgorithm> ptrAlgo);
        };
    }
}

#endif

