#ifndef IALGORITHMSUBSCRIBER_HEADER

#define IALGORITHMSUBSCRIBER_HEADER

#include "CPosition.h"

namespace Mts {
    namespace Algorithm {
        class IAlgorithmSubscriber
        {
        public:
            virtual void onAlgorithmMessage(const std::string &strMsg) = 0;

            virtual void onAlgorithmPositionUpdate(unsigned int iAlgoID,
                    const Mts::Accounting::CPosition &objPosition) = 0;

            // called if an algorithm specific risk limit has been breached and the algo has set itself to PASSIVE
            virtual void onAlgorithmRiskBreach(unsigned int iAlgoID) {};

            // called if an algorithm triggers an internal error and has set itself to PASSIVE
            virtual void onAlgorithmInternalError(unsigned int iAlgoID,
                    const std::string &strMsg) = 0;
        };
    }
}

#endif


