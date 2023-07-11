#ifndef IEXCHANGESUBSCRIBER_HEADER

#define IEXCHANGESUBSCRIBER_HEADER

#include "COrderStatus.h"
#include "COrderFill.h"
#include "CHeartbeat.h"
#include "CExecReport.h"

// MTS execution report
#include "ExecutionReport.h"

namespace Mts {
    namespace Exchange {
        class IExchangeSubscriber
        {
        public:
            // internally generated risk limit violations
            virtual void onExchangeRiskViolation(const Mts::Order::COrderStatus &objOrderStatus) {};

            // externally generated events
            virtual void onExchangeExecReport(const Mts::Order::CExecReport &objExecReport) = 0;

            virtual void onExchangeOrderNew(const Mts::Order::COrder &objOrder) = 0;

            virtual void onExchangeOrderStatus(const Mts::Order::COrderStatus &objOrderStatus) = 0;

            virtual void onExchangeOrderFill(const Mts::Order::COrderFill &objOrderFill) = 0;

            virtual void onExchangeHeartbeat(const Mts::LifeCycle::CHeartbeat &objHeartbeat) = 0;

            virtual void onExchangeLogon(unsigned int iProviderID) = 0;

            virtual void onExchangeLogout(unsigned int iProviderID) = 0;

            virtual void onExchangeMessage(const std::string &strMsg) = 0;

            // the only method that should be used
            virtual void sendExecutionReport(const pm::ExecutionReport &er) {};
        };
    }
}

#endif


