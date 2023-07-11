#ifndef CMODELFACTORY_HEADER

#define CMODELFACTORY_HEADER

#include <boost/property_tree/ptree.hpp>
#include "CModel.h"

namespace Mts {
    namespace Model {
        class CModelFactory
        {
        public:
            static CModelFactory &getInstance();

            CModelFactory() = default;

            CModelFactory(const CModelFactory &) = delete;

            boost::shared_ptr<CModel> createModel(const boost::property_tree::ptree &objPTree,
                    const std::string &strModelName,
                    const std::string &strModelTag,
                    const std::string &strTagPrefix);

            boost::shared_ptr<CModel> createModelPython(const boost::property_tree::ptree &objPTree,
                    const std::string &strModelName,
                    const std::string &strTagPrefix);

            boost::shared_ptr<CModel> createModelCpp(const boost::property_tree::ptree &objPTree,
                    const std::string &strModelName,
                    const std::string &strTagPrefix);
        };
    }
}

#endif

