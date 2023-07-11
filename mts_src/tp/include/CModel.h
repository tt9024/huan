#ifndef CMODEL_HEADER

#define CMODEL_HEADER

#include "CPositionManager.h"
#include "CBidAsk.h"
#include "COrderFill.h"
#include "CCircularScalarBuffer.h"
#include "COHLCBarHist.h"
#include "CKeyValue.h"

namespace Mts {
    namespace Model {
        class CModel
        {
        public:
            CModel(const std::string &strModelName,
                    unsigned int iBarSizeMin,
                    unsigned int iFormationPeriod);

            // overrides
            virtual ~CModel() = default;

            virtual bool initialize();

            virtual void onEvent(const Mts::Order::COrderFill &objOrderFill);

            virtual void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk);

            virtual void onEvent(const Mts::OrderBook::CKeyValue &objKeyValue);

            virtual void updateSignal(const Mts::Core::CDateTime &dtTimestamp) = 0;

            // accessors
            const std::string &getModelName() const;

            void addTradedInstrument(const Mts::Core::CSymbol &objSymbol);

            int getCurrPosition(unsigned int iSymbolID) const;

            int getDesiredPosition(unsigned int iSymbolID) const;

            void setDesiredPosition(unsigned int iSymbolID,
                    int iDesiredPos);

            double getPnLUSD() const;

            unsigned int getNumFills() const;

            unsigned int getNumTradedInstruments() const;

            const Mts::Core::CSymbol &getTradedInstrument(unsigned int iIndex) const;

            const Mts::Indicator::COHLCBarHist &getOHLCBars(unsigned int iSymbolID);

            bool isOHLCHistComplete();

        private:
            using TradedInstrumentArray = std::vector<Mts::Core::CSymbol>;
            using Symbol2OHLCBarMap = boost::unordered_map<unsigned int, Mts::Indicator::COHLCBarHist>;

            // traded instruments
            TradedInstrumentArray m_TradedInstruments;

            std::string m_strModelName;

            // data requirements for this model
            unsigned int m_iBarSizeMin;
            unsigned int m_iFormationPeriod;

            Symbol2OHLCBarMap m_Symbol2OHLCBars;
            bool m_bOHLCHistComplete;

            // actual position for each instrument
            Mts::Accounting::CPositionManager m_PositionManager;

            // current desired position for each instrument
            int m_iDesiredPos[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
        };
    }
}

#endif


