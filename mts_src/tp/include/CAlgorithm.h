#ifndef CALGORITHM_HEADER

#define CALGORITHM_HEADER

#include <string>
#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include "CSymbol.h"
#include "CBidAsk.h"
#include "CTrade.h"
#include "CKeyValue.h"
#include "CManualCommand.h"
#include "COrderStatus.h"
#include "COrderFill.h"
#include "IRunnable.h"
#include "CEvent.h"
#include "CApplicationLog.h"
#include "CEventQueue.h"
#include "CPositionManager.h"
#include "IAlgorithmSubscriber.h"
#include "CPriceMatrix.h"
#include "CDateTimeEvent.h"
#include "CExecReport.h"

namespace Mts {
    namespace Algorithm {
        class CAlgorithm : public Mts::Thread::IRunnable
        {
        public:
            enum OperationalMode
            {
                PASSIVE,
                LIQUIDATE,
                ACTIVE
            };
        public:
            CAlgorithm(unsigned int uiAlgoID,
                    const std::string &strAlgoName,
                    unsigned int uiEventBufferSizeBytes,
                    unsigned int uiUpdateMSec);

            // main event loop will run on a separate thread
            void run();

            void operator()();

            // used to enable/disable event queue to market updates, order updates always go through (for MATLAB models to prevent queue filling up)
            void enableEventQueue();

            void disableEventQueue();

            // place items in the event queue (engine will call these methods, once queued the algo can process the events in it's own time)
            bool onMktBidAsk(const Mts::OrderBook::CBidAsk &objBidAsk);

            bool onTrade(const Mts::OrderBook::CTrade &objTrade);

            bool onOrderStatus(const Mts::Order::COrderStatus &objOrderStatus);

            bool onOrderFill(const Mts::Order::COrderFill &objOrderFill);

            bool onDateTime(const Mts::Core::CDateTimeEvent &objDateTime);

            bool onExecReport(const Mts::Order::CExecReport &objExecReport);

            bool onKeyValue(const Mts::OrderBook::CKeyValue &objKeyValue);

            bool onCommand(const Mts::OrderBook::CManualCommand &objMC);

            // this method will construct the next event from the event queue and call the overloaded onEvent() method to handle it
            template<typename T>
            void handleEvent(Mts::Event::CEvent::EventID iEventID);

            // accessors
            unsigned int getAlgoID() const;

            const std::string &getAlgoName() const;

            bool isStopped() const;

            const Mts::Accounting::CPosition &getPosition(const Mts::Core::CSymbol &objSymbol) const;

            void setPosition(const Mts::Core::CSymbol &objSymbol,
                    const Mts::Accounting::CPosition &objPosition);

            OperationalMode getOperationalMode() const;

            void setOperationalMode(OperationalMode iOperationalMode);

            double getPnLUSD() const;

            double getUsdPosition() const;

            double getCcyUsdPosition(const Mts::Core::CCurrncy &objBaseCcy) const;

            bool isTradedInstrument(unsigned int iSymbolID);

            bool isRecoveryDone() const;

            // manage subscribers
            void addAlgorithmSubscriber(IAlgorithmSubscriber *ptrAlgoritmSubscriber);

            void broadcastMessage(const std::string &strMsg) const;

            void broadcastPosition(const Mts::Core::CSymbol &objSymbol) const;

            void broadcastPnL() const;

            void broadcastRiskBreach(unsigned int iAlgoID) const;

            void broadcastInternalError(const std::string &strMsg) const;

            bool isExchangeAvailable(unsigned int iProviderID) const;

            std::vector<std::string> getState() const;

            // called once when algo is deployed to the engine (resource allocation code goes here)
            virtual void onCreate();

            // called whenever the algo is (re)started by the user
            virtual void onStart();

            // called once daily on the first event
            virtual void onStartOfDay(Mts::Core::CDateTime dtTimestamp);

            // called once, each time the engine starts up, use to initialize state with supplied data history
            virtual void onEngineStart(const Mts::TickData::CPriceMatrix &objPriceHist5Min);

            virtual void onEngineStart(const Mts::TickData::CPriceMatrix &objPriceHist5Min,
                    const Mts::TickData::CPriceMatrix &objPriceHist1Min);

            // these are the ones called in algo's thread
            // handlers for market events
            virtual void onEvent(const Mts::OrderBook::CBidAsk &objBidAsk);

            virtual void onEvent(const Mts::OrderBook::CTrade &objTrade);

            virtual void onEvent(const Mts::Order::COrderStatus &objOrderStatus);

            virtual void onEvent(const Mts::Order::COrderFill &objOrderFill);

            virtual void onEvent(const Mts::OrderBook::CManualCommand &objMC);

            virtual void onEvent(const Mts::Core::CDateTimeEvent &objTimestamp);

            virtual void onEvent(const Mts::Order::CExecReport &objExecReport);

            virtual void onEvent(const Mts::OrderBook::CKeyValue &objKeyValue);

            virtual void onRecoveryDone();

            // called once daily at the end of the trading day (calendar)
            virtual void onEndOfDay(Mts::Core::CDateTime dtTimestamp);

            // called when connections are established to feeds/exchanges
            virtual void onProviderActive(unsigned int iProviderID);

            virtual void onProviderInactive(unsigned int iProviderID);

            // called whenever the algo is stopped by the user
            virtual void onStop();

            // called once when the algo is removed from the engine (resource cleanup code goes here)
            virtual void onDestroy();

            // returns the messages which have been transmitted from the algorithm to the GUI
            virtual std::vector<std::string> getMessageHistory() const;

            // returns algo specific initialization data to the GUI
            virtual std::string getAlgoInitString() const;

            // instrument initialization
            virtual void addTradedInstrument(const Mts::Core::CSymbol &objSymbol);

            // risk limits
            void setMaxPosition(const Mts::Core::CSymbol &objSymbol,
                    int iMaxPosition);

            void setMaxOrderSize(const Mts::Core::CSymbol &objSymbol,
                    int iMaxOrderSize);

            void setMinTradeIntervalSecs(int iMinTradeIntervalInSecs);

            void enforceMaxPositionLimit(const Mts::Accounting::CPosition &objPosition,
                    const Mts::Core::CSymbol &objSymbol);

            void enforceMaxOrderSizeLimit(const Mts::Order::COrderFill &objOrderFill,
                    const Mts::Core::CSymbol &objSymbol);

            void enforceMinTradeInterval(const Mts::Order::COrderFill &objOrderFill,
                    const Mts::Core::CSymbol &objSymbol);

            bool isOrderCompliant(const Mts::Order::COrder &objOrder);

            bool isConcentrationLimitBreached(const Mts::Core::CSymbol &objSymbol,
                    Mts::Order::COrder::BuySell iDirection,
                    unsigned int iQuantityUSD,
                    unsigned int iCcyPosLimitUSD) const;

            bool isPnLStopHit() const;

            void resetStop();

            double getDailyDD() const;

            double getDailyPnL() const;

            void applyRiskMultiplier(double dMultiplier);

            Mts::Accounting::CPositionManager getPositionManager() const;

        private:
            std::string buildPositionMessage(const Mts::Core::CSymbol &objSymbol) const;

            std::string buildPnLUSDMessage() const;

        protected:
            // instruments the algorithm is interested in trading
            typedef std::vector<Mts::Core::CSymbol> TradedInstrumentArray;

            // maintains whether each provider is online or not
            typedef boost::unordered_map<unsigned int, bool> ProviderAvailabilityMap;

            // flags specifying traded instruments
            unsigned int m_TradedInstrumentMap[Mts::Core::CConfig::MAX_NUM_SYMBOLS];

            // positions across all symbols for this algorithm
            Mts::Accounting::CPositionManager m_PositionManager;

            // algorithm log file
            Mts::Log::CApplicationLog &m_Logger;

            // instruments traded by this algorithm
            TradedInstrumentArray m_TradedInstruments;

            // provider availability
            ProviderAvailabilityMap m_ProviderStatus;

        private:
            // subscribers interested in receiving messages from the algorithm (namely the engine which will broadcast the message to the GUI)
            typedef std::vector<IAlgorithmSubscriber *> AlgorithmSubscriberArray;

            AlgorithmSubscriberArray m_AlgoSubscribers;

            unsigned int m_uiAlgoID;
            std::string m_strAlgoName;
            volatile bool m_bStopped;

            volatile bool m_bEventQEnabled;

            // event queue will be processed on a separate thread
            boost::shared_ptr<boost::thread> m_ptrThread;

            // algorithm specific event queue (lock based)
            Mts::Event::CEventQueue m_EventQ;

            // throttle (millseconds) for consolidated book (level II) updates
            Mts::Core::CDateTime m_dtLastUpdate[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            unsigned int m_iUpdateMSec;
            double m_dUpdateIntervalDayFrac;
            double m_dDefaultUpdateIntervalDayFrac;
            double m_dThrottledUpdateIntervalDayFrac;

            // throttle (milliseconds) for pnl updates
            Mts::Core::CDateTime m_dtLastUpdatePnL[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            unsigned int m_iUpdateMSecPnL;

            OperationalMode m_iOperationalMode;

            bool m_bRecoveryDone;

            // algorithm specific risk limits
            int m_iMaxPosition[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            int m_iMaxOrderSize[Mts::Core::CConfig::MAX_NUM_SYMBOLS];

            // minimum time interval between trades (risk check to prevent machine gunning)
            Mts::Core::CDateTime m_dtLastFillTime[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            double m_dMinTradeIntervalDayFrac;

            // max daily loss (trailing)
            double m_dDailyMaxLoss;
            double m_dCurrPnLUSD;
            double m_dMaxPnLUSD;
            double m_dStartOfDayPnLUSD;
        };
    }
}

#include "CAlgorithm.hxx"

#endif

