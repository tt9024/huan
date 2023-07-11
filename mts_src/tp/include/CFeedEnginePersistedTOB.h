#ifndef CFEEDENGINEPERSISTEDTOB_HEADER

#define CFEEDENGINEPERSISTEDTOB_HEADER

#include <vector>
#include "CFeed.h"
#include "CSymbol.h"
#include "CProvider.h"
#include "CConfig.h"

namespace Mts {
    namespace Feed {
        class CFeedEnginePersistedTOB : public CFeed
        {
        public:
            CFeedEnginePersistedTOB(const Mts::Core::CProvider &objProvider,
                    const std::string &strCounterparty,
                    unsigned int iSize,
                    bool bStreamFromAllProviders,
                    double dSpreadMultipler,
                    unsigned int iQuoteThrottleMSecs,
                    unsigned int iStartTimeInMin,
                    unsigned int iEndTimeInMin);

            void addSymbol(unsigned int iSymbolID);

            void addInputFile(const std::string &strInputFile);

            // implementation of IRunnable
            void operator()();

            // overrides
            bool connect();

            bool disconnect();

            void initialize();

        private:
            typedef std::vector<std::string> InputFileArray;

            Mts::Core::CProvider m_Provider;
            std::string m_strCounterparty;
            unsigned int m_iSize;
            unsigned int m_SubscriptionStatus[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            unsigned int m_iNumSubscriptions;

            bool m_bStreamFromAllProviders;

            // spread can be adjusted up or down, or left adjusted if 1.0
            double m_dSpreadMultipler;

            // used to optionally throttle quotes
            Mts::Core::CDateTime m_dtLastQuote[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            double m_dQuoteThrottleDayFrac;

            // used to stream quotes within a specific time window only
            unsigned int m_iStartTimeInMin;
            unsigned int m_iEndTimeInMin;

            InputFileArray m_InputFiles;
        };
    }
}

#endif

