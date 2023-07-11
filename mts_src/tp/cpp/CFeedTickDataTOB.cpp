#include <algorithm>
#include <iostream>
#include <cstring>
#include "CFeedTickDataTOB.h"
#include "CMath.h"


using namespace Mts::Feed;


CFeedTickDataTOB::CFeedTickDataTOB(const Mts::Core::CProvider &objProvider,
        double dSpreadMultipler,
        unsigned int iQuoteThrottleMSecs,
        unsigned int iStartTimeInMin,
        unsigned int iEndTimeInMin,
        bool bParseTradesFile)
        : CFeed(false),
          m_Provider(objProvider),
          m_iNumSubscriptions(0),
          m_dSpreadMultipler(dSpreadMultipler),
          m_dQuoteThrottleDayFrac(static_cast<double>(iQuoteThrottleMSecs) / (24.0 * 60.0 * 60.0 * 1000.0)),
          m_iStartTimeInMin(iStartTimeInMin),
          m_iEndTimeInMin(iEndTimeInMin),
          m_bParseTradesFile(bParseTradesFile) {

    memset(m_SubscriptionStatus, 0, sizeof(unsigned int) * Mts::Core::CConfig::MAX_NUM_SYMBOLS);
}


void CFeedTickDataTOB::addSymbol(unsigned int iSymbolID) {

    m_SubscriptionStatus[iSymbolID] = 1;
    ++m_iNumSubscriptions;
}


void CFeedTickDataTOB::addInputFile(const std::string &strInputFile) {

    m_InputFiles.push_back(strInputFile);
}


void CFeedTickDataTOB::operator()() {

    publishLogon(m_Provider.getProviderID());

    std::cout << "start of data stream " << Mts::Core::CDateTime::now().toStringFull() << std::endl;

    Mts::Core::CDateTime dtLastUpdate;

    for (size_t i = 0; i != m_InputFiles.size(); ++i) {

        char szBuffer[512];

        FILE *ptrCurrentFile = fopen(m_InputFiles[i].c_str(), "r");

        std::string strTDSymbol = m_InputFiles[i].substr(0, m_InputFiles[i].find('_') - 3);
        const char *pszTDSymbol = strTDSymbol.c_str() + strTDSymbol.length();

        while (*(pszTDSymbol - 1) != '\\')
            --pszTDSymbol;

        const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbolFromTDSymbol(pszTDSymbol);

        Mts::OrderBook::CQuote objBid;
        Mts::OrderBook::CQuote objAsk;

        double dBidPx = 0.0;
        int iBidSize = 0;
        double dAskPx = 0.0;
        int iAskSize = 0;

        while (fgets(szBuffer, 512, ptrCurrentFile) != NULL) {

            unsigned int iMM = static_cast<unsigned int>(Mts::Math::CMath::atoi_2(szBuffer));
            unsigned int iDD = static_cast<unsigned int>(Mts::Math::CMath::atoi_2(szBuffer + 3));
            unsigned int iYY = static_cast<unsigned int>(Mts::Math::CMath::atoi_4(szBuffer + 6));
            unsigned int iHH = static_cast<unsigned int>(Mts::Math::CMath::atoi_2(szBuffer + 11));
            unsigned int iMN = static_cast<unsigned int>(Mts::Math::CMath::atoi_2(szBuffer + 14));
            unsigned int iSS = static_cast<unsigned int>(Mts::Math::CMath::atoi_2(szBuffer + 17));
            unsigned int iMS = static_cast<unsigned int>(Mts::Math::CMath::atoi_3(szBuffer + 20));
            unsigned int iTimeInMin = iHH * 60 + iMN;

            if (iTimeInMin < m_iStartTimeInMin || iTimeInMin > m_iEndTimeInMin)
                continue;

            Mts::Core::CDateTime dtTimestamp = Mts::Core::CDateTime(iYY, iMM, iDD, iHH, iMN, iSS, iMS);
            char *pszStart = szBuffer + 24;

            if (m_bParseTradesFile == false) {

                for (int j = 0; j != 4; ++j) {

                    char *pszEnd = pszStart;

                    while (*pszEnd && *pszEnd != ',')
                        ++pszEnd;

                    if (*pszEnd)
                        *pszEnd = '\0';

                    switch (j) {

                        case 0:
                            dBidPx = Mts::Math::CMath::atof(pszStart);
                            break;

                        case 1:
                            iBidSize = Mts::Math::CMath::atoi(pszStart);
                            break;

                        case 2:
                            dAskPx = Mts::Math::CMath::atof(pszStart);
                            break;

                        case 3:
                            iAskSize = Mts::Math::CMath::atoi(pszStart);
                            break;
                    }

                    pszStart = pszEnd + 1;
                }
            } else {

                for (int j = 0; j != 2; ++j) {

                    char *pszEnd = pszStart;

                    while (*pszEnd && *pszEnd != ',')
                        ++pszEnd;

                    if (*pszEnd)
                        *pszEnd = '\0';

                    switch (j) {

                        case 0:
                            dBidPx = Mts::Math::CMath::atof(pszStart);
                            dAskPx = dBidPx;
                            break;

                        case 1:
                            iBidSize = Mts::Math::CMath::atoi(pszStart);
                            iAskSize = iBidSize;
                            break;
                    }

                    pszStart = pszEnd + 1;
                }
            }

            if (m_iNumSubscriptions > 0 && m_SubscriptionStatus[objSymbol.getSymbolID()] == 0)
                continue;

            objBid.setSymbolID(objSymbol.getSymbolID());
            objBid.setMtsTimestamp(dtTimestamp);
            objBid.setExcTimestamp(dtTimestamp);
            objBid.setPrice(dBidPx * objSymbol.getTDScaleMultiplier());
            objBid.setSize(static_cast<unsigned int>(iBidSize));
            objBid.setSide(Mts::OrderBook::CQuote::BID);

            objAsk.setSymbolID(objSymbol.getSymbolID());
            objAsk.setMtsTimestamp(dtTimestamp);
            objAsk.setExcTimestamp(dtTimestamp);
            objAsk.setPrice(dAskPx * objSymbol.getTDScaleMultiplier());
            objAsk.setSize(static_cast<unsigned int>(iAskSize));
            objAsk.setSide(Mts::OrderBook::CQuote::ASK);

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

                if (dtTimestamp.getValue() >= dtLastUpdate.getValue()) {

                    if (m_dSpreadMultipler < 1.0) {

                        double dMid = 0.5 * objBid.getPrice() + 0.5 * objAsk.getPrice();
                        double dSpread = std::max(0.0, objAsk.getPrice() - objBid.getPrice()) * m_dSpreadMultipler;
                        double dAdjBid = dMid - 0.5 * dSpread;
                        double dAdjAsk = dMid + 0.5 * dSpread;
                        objBid.setPrice(dAdjBid);
                        objAsk.setPrice(dAdjAsk);
                    }

                    Mts::OrderBook::CBidAsk objBidAsk(objSymbol.getSymbolID(), dtTimestamp, objBid, objAsk);
                    publishQuoteSim(objBidAsk);

                    dtLastUpdate = dtTimestamp;
                }
            }
        }

        fclose(ptrCurrentFile);
    }

    std::cout << "end of data stream " << Mts::Core::CDateTime::now().toStringFull() << std::endl;

    publishLogout(m_Provider.getProviderID());
    publishNoMoreData();
}


bool CFeedTickDataTOB::connect() {

    return true;
}


bool CFeedTickDataTOB::disconnect() {

    return true;
}


void CFeedTickDataTOB::initialize() {

}

