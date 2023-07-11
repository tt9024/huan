#ifndef CMODELPYTHON_HEADER

#define CMODELPYTHON_HEADER

#include <boost/circular_buffer.hpp>
#include <unordered_map>
#include <Python.h>
#include "CModel.h"
#include "CLogUnorderedMap.h"

namespace Mts {
    namespace Model {
        class CModelPython : public Mts::Model::CModel
        {
        public:
            CModelPython(const std::string &strModelName,
                    const std::string &strPythonModule,
                    const std::string &strPythonFunction,
                    unsigned int iBarSizeSecs,
                    unsigned int iFormationPeriod);

            // overrides
            ~CModelPython() override;

            bool initialize() override;

            void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) override;

            void onEvent(const Mts::OrderBook::CKeyValue &objKeyValue) override;

            void updateSignal(const Mts::Core::CDateTime &dtTimestamp) override;

            // model specific
            bool updateSignal_cplusplus();

            bool updateSignal_python();

            PyObject *createPythonArray(const int *piArray,
                    size_t iNumItems);

            PyObject *createPythonMatrix(const std::vector<std::vector<int>> &objMatrix);

            void initializePython();

        private:
            typedef std::unordered_map<std::string, boost::circular_buffer<Mts::OrderBook::CKeyValue>> DataDictionary;

            std::string m_strModule;
            std::string m_strFunction;

            bool m_bInitializePython;

            PyObject *m_pName;
            PyObject *m_pModule;
            PyObject *m_pFunc;
            PyObject *m_pArgs;

            // key->array of key-value pairs with timestamp. read periodically from the database, passed to Python but not persisted
            DataDictionary m_DataDict;

            // model state is stored as a key-value pair map and is persisted to a flat file (R/W latency ~ 3ms)
            std::unordered_map<std::string, double> m_NumericStateMap;
            std::unordered_map<std::string, std::string> m_StringStateMap;
            Mts::Log::CLogUnorderedMap m_StateLog;
        };
    }
}

#endif

