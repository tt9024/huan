#ifndef CFEEDTICKDATATOB_HEADER

#define CFEEDTICKDATATOB_HEADER

#include <vector>
#include "CFeed.h"
#include "CSymbol.h"
#include "CProvider.h"

namespace Mts {
    namespace Feed {
        class CFeedTickDataTOB : public CFeed
        {
        public:
            CFeedTickDataTOB(const Mts::Core::CProvider &objProvider,
                    double dSpreadMultipler,
                    unsigned int iQuoteThrottleMSecs,
                    unsigned int iStartTimeInMin,
                    unsigned int iEndTimeInMin,
                    bool bParseTradesFile);

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
            unsigned int m_SubscriptionStatus[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            unsigned int m_iNumSubscriptions;

            // spread can be adjusted up or down, or left adjusted if 1.0
            double m_dSpreadMultipler;

            // used to optionally throttle quotes
            Mts::Core::CDateTime m_dtLastQuote[Mts::Core::CConfig::MAX_NUM_SYMBOLS];
            double m_dQuoteThrottleDayFrac;

            // used to stream quotes within a specific time window only
            unsigned int m_iStartTimeInMin;
            unsigned int m_iEndTimeInMin;

            InputFileArray m_InputFiles;

            // input files must all be in either trade for quote file format
            bool m_bParseTradesFile;
        };
    }
}

#endif

