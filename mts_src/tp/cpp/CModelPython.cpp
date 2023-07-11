#include "CModelPython.h"
#include "CMath.h"


using namespace Mts::Model;


CModelPython::CModelPython(const std::string &strModelName,
        const std::string &strPythonModule,
        const std::string &strPythonFunction,
        unsigned int iBarSizeSecs,
        unsigned int iFormationPeriod)
        : CModel(strModelName, iBarSizeSecs, iFormationPeriod),
          m_strModule(strPythonModule),
          m_strFunction(strPythonFunction),
          m_bInitializePython(false),
          m_StateLog(strModelName + ".dat") {

}


bool CModelPython::initialize() {

    Py_Initialize();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("print(sys.version)");
    PyRun_SimpleString("print(sys.path)");

    Py_Finalize();

    return true;
}


void CModelPython::initializePython() {

    Py_Initialize();

    // module (.py file) must be accessible on the path defined by the PYTHONPATH system variable
    m_pName = PyUnicode_FromString(m_strModule.c_str());

    m_pModule = PyImport_Import(m_pName);
    Py_DECREF(m_pName);

    if (m_pModule == nullptr) {

        PyErr_Print();
        throw Mts::Exception::CMtsException("CModelPython: Unknown Python module");
    }

    // function within the .py file
    m_pFunc = PyObject_GetAttrString(m_pModule, m_strFunction.c_str());

    if (m_pFunc == nullptr || PyCallable_Check(m_pFunc) == false) {

        throw Mts::Exception::CMtsException("CModelPython: Unknown Python function");
    }

    // tuple will contain model name, symbols, current positions, timestamp, open, high, low, close, state_dict, data_dict
    m_pArgs = PyTuple_New(10);
}


CModelPython::~CModelPython() {

    if (m_bInitializePython == true) {

        Py_DECREF(m_pArgs);
        Py_XDECREF(m_pFunc);
        Py_DECREF(m_pModule);

        Py_Finalize();
    }
}


void CModelPython::onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) {

    CModel::onEvent(objBidAsk);
}


void CModelPython::onEvent(const Mts::OrderBook::CKeyValue &objKeyValue) {

    CModel::onEvent(objKeyValue);

    if (m_DataDict.find(objKeyValue.getKey()) == m_DataDict.end()) {

        m_DataDict[objKeyValue.getKey()] = boost::circular_buffer<Mts::OrderBook::CKeyValue>(48);
    }

    m_DataDict[objKeyValue.getKey()].push_back(objKeyValue);
}


void CModelPython::updateSignal(const Mts::Core::CDateTime &dtTimestamp) {

    if (isOHLCHistComplete() == false)
        return;

    m_StateLog.load(m_NumericStateMap, m_StringStateMap);

    if (m_NumericStateMap.size() == 0) {

        m_NumericStateMap["counter"] = 0;
    }

    if (m_StringStateMap.size() == 0) {

        m_StringStateMap["model"] = getModelName();
    }

    updateSignal_python();

    m_StateLog.save(m_NumericStateMap, m_StringStateMap);
}


bool CModelPython::updateSignal_cplusplus() {
/*
	for (unsigned int i = 0; i != getNumTradedInstruments(); ++i) {

		const Mts::Core::CSymbol & objInst		 = getTradedInstrument(i);
		auto											 objOHLCBars = getOHLCBars(objInst.getSymbolID());

	auto objMidPxBars = getMidPxBars(objBidAsk.getSymbolID());

	int iCurrPos = getCurrPosition(objInst.getSymbolID());

	double dSMA	= objMidPxBars.getMean();

	int iSignal = Mts::Math::CMath::sign(objInst.getMidPx() - dSMA);

	setDesiredPosition(objInst.getSymbolID(), 100*iSignal);
*/
    return true;
}


bool CModelPython::updateSignal_python() {

    if (m_bInitializePython == false) {

        initializePython();
        m_bInitializePython = true;
    }


    // model name
    PyObject *pModelName = PyUnicode_FromString(getModelName().c_str());
    PyTuple_SetItem(m_pArgs, 0, pModelName);


    // array of traded instrument names
    unsigned int iNumInst = getNumTradedInstruments();

    PyObject *pSymbols = PyList_New(iNumInst);

    for (unsigned int i = 0; i != iNumInst; ++i) {

        const Mts::Core::CSymbol &objInst = getTradedInstrument(i);
        PyList_SET_ITEM(pSymbols, i, PyUnicode_FromString(objInst.getSymbol().c_str()));
    }

    PyTuple_SetItem(m_pArgs, 1, pSymbols);


    // array of current positions
    PyObject *pCurrPos = PyList_New(iNumInst);

    for (unsigned int i = 0; i != iNumInst; ++i) {

        const Mts::Core::CSymbol &objInst = getTradedInstrument(i);
        int iCurrPos = getCurrPosition(objInst.getSymbolID());
        PyList_SET_ITEM(pCurrPos, i, PyLong_FromLong(iCurrPos));
    }

    PyTuple_SetItem(m_pArgs, 2, pCurrPos);


    // matrix of OHLC prices
    for (unsigned int k = 0; k != 5; ++k) {

        PyObject *pMatrix = PyList_New(iNumInst);

        for (unsigned int i = 0; i != iNumInst; ++i) {

            const Mts::Core::CSymbol &objInst = getTradedInstrument(i);
            auto objOHLCBars = getOHLCBars(objInst.getSymbolID());
            unsigned int iNumBars = objOHLCBars.getNumBars();

            PyObject *pVector = PyList_New(iNumBars);

            for (int j = iNumBars - 1; j >= 0; --j) {

                switch (k) {

                    case 0:
                        PyList_SET_ITEM(pVector, j,
                                        PyLong_FromLongLong(objOHLCBars.getBarTimestampClose(j).getCMTime()));
                        break;

                    case 1:
                        PyList_SET_ITEM(pVector, j, PyFloat_FromDouble(objOHLCBars.getBarOpen(j)));
                        break;

                    case 2:
                        PyList_SET_ITEM(pVector, j, PyFloat_FromDouble(objOHLCBars.getBarHigh(j)));
                        break;

                    case 3:
                        PyList_SET_ITEM(pVector, j, PyFloat_FromDouble(objOHLCBars.getBarLow(j)));
                        break;

                    case 4:
                        PyList_SET_ITEM(pVector, j, PyFloat_FromDouble(objOHLCBars.getBarClose(j)));
                        break;
                }
            }

            PyList_SET_ITEM(pMatrix, i, pVector);
        }

        PyTuple_SetItem(m_pArgs, 3 + k, pMatrix);
    }


    // dictionary of key-value pairs storing generic state information
    PyObject *pDict = PyDict_New();

    for (auto &iter: m_NumericStateMap) {

        PyDict_SetItem(pDict, PyUnicode_FromString(iter.first.c_str()), PyFloat_FromDouble(iter.second));
    }

    for (auto &iter: m_StringStateMap) {

        PyDict_SetItem(pDict, PyUnicode_FromString(iter.first.c_str()), PyUnicode_FromString(iter.second.c_str()));
    }

    PyTuple_SetItem(m_pArgs, 8, pDict);


    // data dictionary of key-value pairs read from the database
    PyObject *pDataDict = PyDict_New();

    for (auto &iter: m_DataDict) {

        PyObject *pNextList = PyList_New(iter.second.size());

        unsigned int i = 0;

        for (auto &item: iter.second) {

            PyObject *pKeyValue = PyTuple_New(2);

            PyTuple_SetItem(pKeyValue, 0, PyLong_FromLongLong(item.getTimestamp().getCMTime()));
            PyTuple_SetItem(pKeyValue, 1, PyFloat_FromDouble(item.getValue()));

            PyList_SET_ITEM(pNextList, i, pKeyValue);
            ++i;
        }

        PyDict_SetItem(pDataDict, PyUnicode_FromString(iter.first.c_str()), pNextList);
    }

    PyTuple_SetItem(m_pArgs, 9, pDataDict);


    // run python script - must return a tuple (list,dict) containing the desired position and state dictionary respectively
    PyObject *pValue = PyObject_CallObject(m_pFunc, m_pArgs);


    int iNumInTupleResults = static_cast<int>(PyTuple_Size(pValue));

    if (iNumInTupleResults != 2) {

    }


    // extract list of desired positions
    PyObject *pList = PyTuple_GetItem(pValue, 0);

    if (pList != nullptr) {

        int iNumResults = static_cast<int>(PyList_Size(pList));

        if (iNumResults != iNumInst) {

        }

        PyObject *pTemp, *pObjRep;

        for (unsigned int i = 0; i != iNumResults; ++i) {

            pTemp = PyList_GetItem(pList, i);
            pObjRep = PyObject_Repr(pTemp);
            const char *pszData = PyUnicode_AsUTF8(pObjRep);
            int iDesiredPos = static_cast<int>(strtod(pszData, NULL));

            const Mts::Core::CSymbol &objInst = getTradedInstrument(i);
            setDesiredPosition(objInst.getSymbolID(), iDesiredPos);
        }
    }


    // extract dictionary containing state as key-value pairs
    PyObject *pStateDict = PyTuple_GetItem(pValue, 1);

    if (pStateDict != nullptr) {

        PyObject *pKey, *pValue;
        Py_ssize_t iPos = 0;

        while (PyDict_Next(pStateDict, &iPos, &pKey, &pValue)) {

            PyObject *pKeyStr = PyObject_Repr(pKey);
            std::string strKey(PyUnicode_AsUTF8(pKeyStr));

            PyObject *pValueStr = PyObject_Repr(pValue);

            // Python returns key enclosed with single quotes which need to be removed
            std::string strAdjKey = strKey.substr(1, strKey.length() - 2);

            bool bStrVal = PyUnicode_Check(pValue);

            if (bStrVal == true) {

                std::string strVal(PyUnicode_AsUTF8(pValueStr));
                std::string strAdjVal = strVal.substr(1, strVal.length() - 2);

                m_StringStateMap[strAdjKey] = strAdjVal;
            } else {

                const char *pszData = PyUnicode_AsUTF8(pValueStr);
                double dValue = strtod(pszData, NULL);

                m_NumericStateMap[strAdjKey] = dValue;
            }
        }
    }


    Py_DECREF(pValue);

    return true;
}


PyObject *CModelPython::createPythonArray(const int *piArray,
        size_t iNumItems) {

    PyObject *l = PyList_New(iNumItems);

    for (size_t i = 0; i != iNumItems; ++i) {

        PyList_SET_ITEM(l, i, PyLong_FromLong(piArray[i]));
    }

    return l;
}


PyObject *CModelPython::createPythonMatrix(const std::vector<std::vector<int>> &objMatrix) {

    size_t iNumVectors = objMatrix.size();

    PyObject *l = PyList_New(iNumVectors);

    for (size_t i = 0; i != iNumVectors; ++i) {

        size_t iVectorLen = objMatrix[i].size();

        PyObject *v = createPythonArray(objMatrix[i].data(), iVectorLen);

        PyList_SET_ITEM(l, i, v);
    }

    return l;
}




