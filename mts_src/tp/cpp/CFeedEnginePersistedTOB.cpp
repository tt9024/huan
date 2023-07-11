#include <boost/iostreams/device/mapped_file.hpp>
#include <algorithm>
#include <iostream>
#include <cstring>
#include "CFeedEnginePersistedTOB.h"
#include "CMath.h"


using namespace Mts::Feed;


CFeedEnginePersistedTOB::CFeedEnginePersistedTOB(const Mts::Core::CProvider &objProvider,
        const std::string &strCounterparty,
        unsigned int iSize,
        bool bStreamFromAllProviders,
        double dSpreadMultipler,
        unsigned int iQuoteThrottleMSecs,
        unsigned int iStartTimeInMin,
        unsigned int iEndTimeInMin)
        : CFeed(false),
          m_Provider(objProvider),
          m_strCounterparty(strCounterparty),
          m_iSize(iSize),
          m_iNumSubscriptions(0),
          m_bStreamFromAllProviders(bStreamFromAllProviders),
          m_dSpreadMultipler(dSpreadMultipler),
          m_dQuoteThrottleDayFrac(static_cast<double>(iQuoteThrottleMSecs) / (24.0 * 60.0 * 60.0 * 1000.0)),
          m_iStartTimeInMin(iStartTimeInMin),
          m_iEndTimeInMin(iEndTimeInMin) {

    memset(m_SubscriptionStatus, 0, sizeof(unsigned int) * Mts::Core::CConfig::MAX_NUM_SYMBOLS);
}


void CFeedEnginePersistedTOB::addSymbol(unsigned int iSymbolID) {

    m_SubscriptionStatus[iSymbolID] = 1;
    ++m_iNumSubscriptions;
}


void CFeedEnginePersistedTOB::addInputFile(const std::string &strInputFile) {

    m_InputFiles.push_back(strInputFile);
}


void CFeedEnginePersistedTOB::operator()() {

    publishLogon(m_Provider.getProviderID());

    std::cout << "start of data stream " << Mts::Core::CDateTime::now().toStringFull() << std::endl;

    Mts::Core::CDateTime dtLastUpdate;

    for (size_t i = 0; i != m_InputFiles.size(); ++i) {

        char szBuffer[512];

        FILE *ptrCurrentFile = fopen(m_InputFiles[i].c_str(), "r");

        Mts::OrderBook::CQuote objBid;
        Mts::OrderBook::CQuote objAsk;

        std::string strProvider = "";
        std::string strCounterparty = "";
        std::string strSymbol = "";
        double dPrice = 0.0;
        int iSize = 0;
        char iSide = 0;

        while (fgets(szBuffer, 512, ptrCurrentFile) != NULL) {

            int iYY = Mts::Math::CMath::atoi_4(szBuffer);
            int iMM = Mts::Math::CMath::atoi_2(szBuffer + 4);
            int iDD = Mts::Math::CMath::atoi_2(szBuffer + 6);
            int iHH = Mts::Math::CMath::atoi_2(szBuffer + 8);
            int iMN = Mts::Math::CMath::atoi_2(szBuffer + 10);
            int iSS = Mts::Math::CMath::atoi_2(szBuffer + 12);
            int iMS = Mts::Math::CMath::atoi_3(szBuffer + 14);
            int iTimeInMin = iHH * 60 + iMN;

            if ((unsigned int) iTimeInMin < m_iStartTimeInMin || (unsigned int) iTimeInMin > m_iEndTimeInMin)
                continue;

            Mts::Core::CDateTime dtTimestamp = Mts::Core::CDateTime(iYY, iMM, iDD, iHH, iMN, iSS, iMS);
            char *pszStart = szBuffer + 18;

            for (int j = 0; j != 6; ++j) {

                char *pszEnd = pszStart;

                while (*pszEnd && *pszEnd != ',')
                    ++pszEnd;

                if (*pszEnd)
                    *pszEnd = '\0';

                switch (j) {

                    case 0:
                        strProvider = std::string(pszStart, pszEnd);
                        break;

                    case 1:
                        strCounterparty = std::string(pszStart, pszEnd);
                        break;

                    case 2:
                        strSymbol = std::string(pszStart, pszEnd);
                        break;

                    case 3:
                        dPrice = Mts::Math::CMath::atof(pszStart);
                        break;

                    case 4:
                        iSize = Mts::Math::CMath::atoi(pszStart);
                        break;

                    case 5:
                        iSide = *pszStart;
                        break;
                }

                pszStart = pszEnd + 1;
            }

            //if (Mts::Core::CProvider::isProvider(strProvider) == false ||
            //		Mts::Core::CProvider::isProvider(strCounterparty) == false)
            //	continue;

            const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(strSymbol);

            if (m_iNumSubscriptions > 0 && m_SubscriptionStatus[objSymbol.getSymbolID()] == 0)
                continue;

            const Mts::Core::CProvider &objProvider = Mts::Core::CProvider::getProvider("TT");
            const Mts::Core::CProvider &objCounterparty = Mts::Core::CProvider::getProvider("TT");
            Mts::OrderBook::CQuote::Side iBidAsk =
                    iSide == 'B' ? Mts::OrderBook::CQuote::BID : Mts::OrderBook::CQuote::ASK;
            Mts::OrderBook::CQuote &objQuote = iSide == 'B' ? objBid : objAsk;

            objQuote = Mts::OrderBook::CQuote(objSymbol.getSymbolID(),
                                              objProvider.getProviderID(),
                                              dtTimestamp,
                                              dtTimestamp,
                                              iBidAsk,
                                              dPrice,
                                              iSize,
                                              "dummy",
                                              "dummy");

            if (iSide == 'A' && objBid.getSymbolID() == objAsk.getSymbolID()) {

                bool bValid = (m_bStreamFromAllProviders == true) || (m_Provider.getName() == strProvider &&
                                                                      (m_strCounterparty == "ALL" ||
                                                                       m_strCounterparty == strCounterparty));

                // optionally throttle quotes
                bool bThrottledQuote = false;

                if (m_dQuoteThrottleDayFrac > 0) {

                    if (dtTimestamp.getValue() - m_dtLastQuote[objSymbol.getSymbolID()].getValue() >
                        m_dQuoteThrottleDayFrac) {

                        m_dtLastQuote[objSymbol.getSymbolID()] = dtTimestamp;
                    } else {

                        bThrottledQuote = true;
                    }
                }

                if (bThrottledQuote == false) {

                    Mts::OrderBook::CBidAsk objBidAsk(objSymbol.getSymbolID(), dtTimestamp, objBid, objAsk);
                    publishQuoteSim(objBidAsk);
                }
            }
        }

        fclose(ptrCurrentFile);
    }

    std::cout << "end of data stream " << Mts::Core::CDateTime::now().toStringFull() << std::endl;

    publishLogout(m_Provider.getProviderID());
}


bool CFeedEnginePersistedTOB::connect() {

    return true;
}


bool CFeedEnginePersistedTOB::disconnect() {

    return true;
}


void CFeedEnginePersistedTOB::initialize() {

}

