#ifndef CMARKETDATAMANAGER_HEADER

#define CMARKETDATAMANAGER_HEADER

#include "CAlgorithm.h"
#include "CConfig.h"
#include "CTopOfBook.h"
#include "CToBSnapshot.h"

namespace Mts {
    namespace TickData {
        class CMarketDataManager : public Mts::Algorithm::CAlgorithm
        {
        public:
            CMarketDataManager(unsigned int uiEventBufferSizeBytes,
                    unsigned int uiUpdateMSec,
                    bool bUseSQL);

            ~CMarketDataManager();

            void onCreate();

            void onStart();

            void onStartOfDay(Mts::Core::CDateTime dtTimestamp);

            void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk);

            void onEvent(const Mts::OrderBook::CTrade &objTrade);

            void onEvent(const Mts::Order::COrderStatus &objOrderStatus);

            void onEvent(const Mts::Order::COrderFill &objOrderFill);

            void onEvent(const std::string &strUserCommand);

            void onEndOfDay(Mts::Core::CDateTime dtTimestamp);

            void onStop();

            void onDestroy();

        private:
            double computeIntervalEndTime(const Mts::Core::CDateTime &dtTimestamp,
                    double dWriteIntervalDayFrac) const;

        private:
            // store current TOB and last trade
            Mts::OrderBook::CToBSnapshot m_TopOfBookSnapshot[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            Mts::OrderBook::CTrade m_LastTrade[Mts::Core::CConfig::MAX_NUM_SYMBOLS];


            bool m_bUseSQL;

            double m_dIntervalEndTime[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            double m_dIntervalEndTimeHF[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
        };
    }
}

#endif


