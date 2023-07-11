#include "CModelCpp.h"
#include "CMath.h"
#include "CApplicationLog.h"


using namespace Mts::Model;


CModelCpp::CModelCpp(const std::string &strModelName,
        unsigned int iBarSizeSecs,
        unsigned int iFormationPeriod,
        unsigned int iClipSize)
        : CModel(strModelName, iBarSizeSecs, iFormationPeriod),
          m_StateLog(strModelName + ".dat"),
          m_iClipSize(iClipSize) {

}


bool CModelCpp::initialize() {

    return true;
}


CModelCpp::~CModelCpp() {

}


void CModelCpp::onEvent(const Mts::OrderBook::CBidAsk &objBidAsk) {

    CModel::onEvent(objBidAsk);
}


void CModelCpp::onEvent(const Mts::OrderBook::CKeyValue &objKeyValue) {

    CModel::onEvent(objKeyValue);

    if (m_DataDict.find(objKeyValue.getKey()) == m_DataDict.end()) {

        m_DataDict[objKeyValue.getKey()] = boost::circular_buffer<Mts::OrderBook::CKeyValue>(48);
    }

    m_DataDict[objKeyValue.getKey()].push_back(objKeyValue);
}


void CModelCpp::updateSignal(const Mts::Core::CDateTime &dtTimestamp) {

    try {

        if (isOHLCHistComplete() == false)
            return;

        m_StateLog.load(m_NumericStateMap, m_StringStateMap);

        for (unsigned int i = 0; i != getNumTradedInstruments(); ++i) {

            const Mts::Core::CSymbol &objInst = getTradedInstrument(i);
            auto objOHLCBars = getOHLCBars(objInst.getSymbolID());
            int iCurrPos = getCurrPosition(objInst.getSymbolID());
            double dSMA = 0;

            for (unsigned int j = 0; j != objOHLCBars.getNumBars(); ++j) {

                dSMA += objOHLCBars.getBarClose(j) / static_cast<double>(objOHLCBars.getNumBars());
            }

            int iSignal = Mts::Math::CMath::sign(objOHLCBars.getBarClose(0) - dSMA);

            setDesiredPosition(objInst.getSymbolID(), static_cast<int>(m_iClipSize) * iSignal);

            // state information can be stored in any member variable but to take advantage of the built-in persistance, use the maps
            if (m_NumericStateMap.find("counter") == m_NumericStateMap.end()) {

                m_NumericStateMap["counter"] = 0;
            } else {

                m_NumericStateMap["counter"] = m_NumericStateMap["counter"] + 1;
            }
        }

        m_StateLog.save(m_NumericStateMap, m_StringStateMap);
    }
    catch (std::exception &e) {

        AppError("Exception in CModelCpp::updateSignal: %s", e.what());
    }
}

