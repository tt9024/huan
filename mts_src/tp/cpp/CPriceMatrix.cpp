#include "CPriceMatrix.h"
#include "CSymbol.h"
#include "CApplicationLog.h"


using namespace Mts::TickData;


CPriceMatrix::CPriceMatrix(const std::vector<Mts::SQL::CSQLBestBidOffer> &objPriceHist) {

    memset(m_SymbolID2Column, -1, sizeof(int) * Mts::Core::CConfig::MAX_NUM_SYMBOLS);


    // step 1: build list of unique, ordered dates and symbols
    for (int i = 0; i != objPriceHist.size(); ++i) {

        m_Timestamps.insert(objPriceHist[i].getTimestamp());
        m_SymbolIDs.insert(objPriceHist[i].getSymbolID());
    }

    m_iNumDates = m_Timestamps.size();
    m_iNumSymbols = m_SymbolIDs.size();


    // step 2: map dates/symbols to rows/columns
    std::map<unsigned long long, unsigned int> objDate2Row;
    std::map<unsigned int, unsigned int> objSymbol2Row;

    int iRow = 0;
    OrderedTimestamps::iterator iter = m_Timestamps.begin();

    for (; iter != m_Timestamps.end(); ++iter) {

        m_TimestampsArray.push_back(*iter);

        objDate2Row[iter->getCMTime()] = iRow;
        ++iRow;
    }

    int iCol = 0;
    OrderedSymbolIDs::iterator iter2 = m_SymbolIDs.begin();

    for (; iter2 != m_SymbolIDs.end(); ++iter2) {

        m_SymbolID2Column[*iter2] = iCol;

        objSymbol2Row[*iter2] = iCol;
        ++iCol;
    }


    // step 3: create matrix
    m_Matrix = new Mts::SQL::CSQLBestBidOffer *[m_iNumDates];

    for (int i = 0; i != m_iNumDates; ++i) {

        m_Matrix[i] = new Mts::SQL::CSQLBestBidOffer[m_iNumSymbols];
    }


    // step 4: remap rows to matrix
    for (int i = 0; i != objPriceHist.size(); ++i) {

        unsigned int iY = objDate2Row[objPriceHist[i].getTimestamp().getCMTime()];
        unsigned int iX = objSymbol2Row[objPriceHist[i].getSymbolID()];

        m_Matrix[iY][iX] = objPriceHist[i];
    }
}


CPriceMatrix::~CPriceMatrix() {

    for (int i = 0; i != m_iNumDates; ++i) {

        delete[] m_Matrix[i];
    }

    delete[] m_Matrix;
}


unsigned int CPriceMatrix::getNumDates() const {

    return m_iNumDates;
}


unsigned int CPriceMatrix::getNumSymbols() const {

    return m_iNumSymbols;
}


const Mts::Core::CDateTime &CPriceMatrix::getTimestamp(unsigned int iIndex) const {

    return m_TimestampsArray[iIndex];
}


double CPriceMatrix::getMidPx(unsigned int iTimestampIndex,
        const Mts::Core::CSymbol &objSymbol) const {

    unsigned int iSymbolIndex = m_SymbolID2Column[objSymbol.getSymbolID()];

    if (iSymbolIndex == -1)
        return 0.0;

    double dMidPx = 0.5 * m_Matrix[iTimestampIndex][iSymbolIndex].getBidPx() +
                    0.5 * m_Matrix[iTimestampIndex][iSymbolIndex].getAskPx();

    while (dMidPx <= 0 && iTimestampIndex > 0) {

        --iTimestampIndex;
        dMidPx = 0.5 * m_Matrix[iTimestampIndex][iSymbolIndex].getBidPx() +
                 0.5 * m_Matrix[iTimestampIndex][iSymbolIndex].getAskPx();
    }

    return dMidPx;
}


bool CPriceMatrix::saveToFile(const std::string &strSymbolsOutputFile,
        const std::string &strPricesOutputFile) const {

    try {

        char szBuffer[2048];


        // header row
        FILE *ptrOutfile = fopen(strSymbolsOutputFile.c_str(), "w");

        OrderedSymbolIDs::const_iterator iterSym = m_SymbolIDs.begin();

        for (; iterSym != m_SymbolIDs.end(); ++iterSym) {

            const Mts::Core::CSymbol &objSymbol = Mts::Core::CSymbol::getSymbol(*iterSym);

            if (iterSym == m_SymbolIDs.begin())
                sprintf(szBuffer, "%s", objSymbol.getSymbol().c_str());
            else
                sprintf(szBuffer, "%s,%s", szBuffer, objSymbol.getSymbol().c_str());
        }

        fprintf(ptrOutfile, "%s\n", szBuffer);
        fclose(ptrOutfile);


        // data rows
        ptrOutfile = fopen(strPricesOutputFile.c_str(), "w");

        OrderedTimestamps::const_iterator iterDt = m_Timestamps.begin();

        for (int i = 0; i != m_iNumDates; ++i) {

            unsigned int iYY = 0;
            unsigned int iMM = 0;
            unsigned int iDD = 0;
            unsigned int iHour = 0;
            unsigned int iMin = 0;
            unsigned int iSec = 0;
            unsigned int iMSec = 0;

            iterDt->decompose(iYY, iMM, iDD, iHour, iMin, iSec, iMSec);

            sprintf(szBuffer, "%d,%d,%d,%d,%d,%d,%d", iYY, iMM, iDD, iHour, iMin, iSec, iMSec);
            ++iterDt;

            for (int j = 0; j != m_iNumSymbols; ++j) {

                double dMid = 0.5 * m_Matrix[i][j].getAskPx() + 0.5 * m_Matrix[i][j].getBidPx();
                sprintf(szBuffer, "%s,%.5f", szBuffer, dMid);
            }

            fprintf(ptrOutfile, "%s\n", szBuffer);
        }

        fclose(ptrOutfile);

        return true;
    }
    catch (std::exception &e) {
        AppError("Exception occurred writing price matrix to file: %s", e.what());
        return false;
    }
}


bool CPriceMatrix::saveToFileSpreads(const std::string &strSpreadsOutputFile) const {

    try {

        char szBuffer[2048];

        // data rows
        FILE *ptrOutfile = fopen(strSpreadsOutputFile.c_str(), "w");

        OrderedTimestamps::const_iterator iterDt = m_Timestamps.begin();

        for (int i = 0; i != m_iNumDates; ++i) {

            unsigned int iYY = 0;
            unsigned int iMM = 0;
            unsigned int iDD = 0;
            unsigned int iHour = 0;
            unsigned int iMin = 0;
            unsigned int iSec = 0;
            unsigned int iMSec = 0;

            iterDt->decompose(iYY, iMM, iDD, iHour, iMin, iSec, iMSec);

            sprintf(szBuffer, "%d,%d,%d,%d,%d,%d,%d", iYY, iMM, iDD, iHour, iMin, iSec, iMSec);
            ++iterDt;

            for (int j = 0; j != m_iNumSymbols; ++j) {

                sprintf(szBuffer, "%s,%.5f", szBuffer, m_Matrix[i][j].getSpreadBps());
            }

            fprintf(ptrOutfile, "%s\n", szBuffer);
        }

        fclose(ptrOutfile);

        return true;
    }
    catch (std::exception &e) {
        AppError("Exception occurred writing spread matrix to file: %s", e.what());
        return false;
    }
}





