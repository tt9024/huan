#ifndef CMODELCPP_HEADER

#define CMODELCPP_HEADER

#include <boost/circular_buffer.hpp>
#include <unordered_map>
#include "CModel.h"
#include "CLogUnorderedMap.h"

namespace Mts {
    namespace Model {
        class CModelCpp : public Mts::Model::CModel
        {
        public:
            CModelCpp(const std::string &strModelName,
                    unsigned int iBarSizeSecs,
                    unsigned int iFormationPeriod,
                    unsigned int iClipSize);

            // overrides
            ~CModelCpp() override;

            bool initialize() override;

            void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) override;

            void onEvent(const Mts::OrderBook::CKeyValue &objKeyValue) override;

            void updateSignal(const Mts::Core::CDateTime &dtTimestamp) override;

        private:
            typedef std::unordered_map<std::string, boost::circular_buffer<Mts::OrderBook::CKeyValue>> DataDictionary;

            // key->array of key-value pairs with timestamp. read periodically from the database, not persisted
            DataDictionary m_DataDict;

            // model state is stored as a key-value pair map and is persisted to a flat file (R/W latency ~ 3ms)
            std::unordered_map<std::string, double> m_NumericStateMap;
            std::unordered_map<std::string, std::string> m_StringStateMap;
            Mts::Log::CLogUnorderedMap m_StateLog;

            // other model parameters
            unsigned int m_iClipSize;
        };
    }
}

#endif

