#include "CCircularScalarBuffer.h"


using namespace Mts::Indicator;


CCircularScalarBuffer::CCircularScalarBuffer(unsigned int iSize,
        unsigned int iIntervalMSec,
        bool bDummy)
        : m_Buffer(iSize),
          m_iIntervalMSec(iIntervalMSec),
          m_bFilled(false) {

    m_iHead = m_Buffer.size() - 1;
    m_iCurrent = m_iHead;
    m_iTail = 0;

    m_dAlpha = 1.5 * (1.0 / static_cast<double>(iSize));
    m_dEWMA = 0.0;
}


CCircularScalarBuffer::CCircularScalarBuffer(unsigned int iSize,
        unsigned int iIntervalSec)
        : m_Buffer(iSize),
          m_iIntervalMSec(iIntervalSec * 1000),
          m_bFilled(false) {

    m_iHead = m_Buffer.size() - 1;
    m_iCurrent = m_iHead;
    m_iTail = 0;

    m_dAlpha = 1.5 * (1.0 / static_cast<double>(iSize));
    m_dEWMA = 0.0;
}


// this method is used for preloading the buffer
void CCircularScalarBuffer::preload(const Mts::Core::CDateTime &dtNow,
        double dItem) {

    append(dtNow, dItem, true);
}


// simply overwrite current item
void CCircularScalarBuffer::update(const Mts::Core::CDateTime &dtNow,
        double dItem) {

    m_Buffer[m_iHead] = dItem;
}


void CCircularScalarBuffer::append(const Mts::Core::CDateTime &dtNow,
        double dItem) {

    append(dtNow, dItem, false);
}


void CCircularScalarBuffer::append(const Mts::Core::CDateTime &dtNow,
        double dItem,
        bool bPreload) {

    m_Buffer[m_iHead] = dItem;
    m_iCurrent = m_iHead;

    if (bPreload == true ||
        dtNow.getValue() - m_dtLastUpdateTime.getValue() >= static_cast<double>(m_iIntervalMSec) / (24 * 3600 * 1000)) {

        ++m_iHead;
        ++m_iTail;

        if (m_iHead == m_Buffer.size())
            m_iHead = 0;

        if (m_iTail == m_Buffer.size()) {
            m_bFilled = true;
            m_iTail = 0;
        }

        // update EWMA
        m_dEWMA = m_dAlpha * dItem + (1.0 - m_dAlpha) * m_dEWMA;

        m_dtLastUpdateTime = dtNow;
    }
}


double CCircularScalarBuffer::getHead() const {

    return m_Buffer[m_iCurrent];
}


double CCircularScalarBuffer::getPrevious() const {

    return m_iCurrent == 0 ? m_Buffer[m_Buffer.size() - 1] : m_Buffer[m_iCurrent - 1];
}


double CCircularScalarBuffer::getTail() const {

    int iIdxTail = m_iCurrent - (m_Buffer.size() - 1);

    return iIdxTail >= 0 ? m_Buffer[iIdxTail] : m_Buffer[iIdxTail + m_Buffer.size()];
}


double CCircularScalarBuffer::sumProduct(double *pdWeights, int iSize) const {

    double dSumProd = 0.0;
    int iHead = m_iCurrent;

    for (int i = 0; i != iSize; ++i) {

        dSumProd += m_Buffer[iHead] * pdWeights[i];

        iHead = iHead == 0 ? m_Buffer.size() - 1 : iHead - 1;
    }

    return dSumProd;
}


double CCircularScalarBuffer::sumProduct(const boost::numeric::ublas::matrix<double> &objWeights) const {

    double dSumProd = 0.0;
    int iHead = m_iCurrent;

    for (unsigned int i = 0; i != objWeights.size2(); ++i) {

        double dWtg = objWeights(0, i);

        dSumProd += m_Buffer[iHead] * dWtg;

        iHead = iHead == 0 ? m_Buffer.size() - 1 : iHead - 1;
    }

    return dSumProd;
}


bool CCircularScalarBuffer::isFull() const {

    return m_bFilled;
}


unsigned int CCircularScalarBuffer::getSize() const {

    return m_Buffer.size();
}


double CCircularScalarBuffer::getItem(int iIndex) const {

    return m_iCurrent + iIndex < m_Buffer.size() ? m_Buffer[m_iCurrent + iIndex] : m_Buffer[m_iCurrent + iIndex -
                                                                                            m_Buffer.size()];
}


// index = 0 returns current item, index = 1 returns previous etc.
double CCircularScalarBuffer::getItemOffsetFromHead(int iIndex) const {

    return static_cast<int>(m_iCurrent) - iIndex >= 0 ? m_Buffer[m_iCurrent - iIndex] : m_Buffer[m_iCurrent - iIndex +
                                                                                                 m_Buffer.size()];
}


void CCircularScalarBuffer::getMinMax(double &dMin,
        double &dMax) const {

    dMin = 0.0;
    dMax = 0.0;

    for (size_t i = 0; i != m_Buffer.size(); ++i) {

        if (i == 0 || m_Buffer[i] < dMin)
            dMin = m_Buffer[i];

        if (i == 0 || m_Buffer[i] > dMax)
            dMax = m_Buffer[i];
    }
}


double CCircularScalarBuffer::getMin() const {

    double dMin = 0.0;
    double dMax = 0.0;

    getMinMax(dMin, dMax);

    return dMin;
}


double CCircularScalarBuffer::getMax() const {

    double dMin = 0.0;
    double dMax = 0.0;

    getMinMax(dMin, dMax);

    return dMax;
}


double CCircularScalarBuffer::getSum() const {

    double dSum = 0.0;

    for (size_t i = 0; i != m_Buffer.size(); ++i) {

        dSum += m_Buffer[i];
    }

    return dSum;
}


double CCircularScalarBuffer::getMean() const {

    double dSum = 0.0;

    for (size_t i = 0; i != m_Buffer.size(); ++i) {

        dSum += m_Buffer[i];
    }

    return dSum / m_Buffer.size();
}


double CCircularScalarBuffer::getStd() const {

    double dMean = getMean();
    double dSum = 0.0;

    for (size_t i = 0; i != m_Buffer.size(); ++i) {

        dSum += pow(m_Buffer[i] - dMean, 2);
    }

    dSum /= (m_Buffer.size() - 1);

    double dStd = sqrt(dSum);

    return dStd;
}


double CCircularScalarBuffer::getZScore() const {

    double dMean = getMean();
    double dSum = 0.0;

    for (size_t i = 0; i != m_Buffer.size(); ++i) {

        dSum += pow(m_Buffer[i] - dMean, 2);
    }

    dSum /= (m_Buffer.size() - 1);

    double dStd = sqrt(dSum);

    return (dStd > 0 ? (getHead() - dMean) / dStd : 0.0);
}


double CCircularScalarBuffer::getEWMA() const {

    return m_dEWMA;
}


void CCircularScalarBuffer::reset() {

    m_bFilled = false;
    m_iHead = m_Buffer.size() - 1;
    m_iCurrent = m_iHead;
    m_iTail = 0;
}


std::string CCircularScalarBuffer::toString() const {

    char szBuffer[256];

    int iHead = m_iCurrent;

    for (unsigned int i = 0; i != m_Buffer.size(); ++i) {

        if (i == 0)
            sprintf(szBuffer, "%.2f", 10000 * m_Buffer[iHead]);
        else
            sprintf(szBuffer, "%s %.2f", szBuffer, 10000 * m_Buffer[iHead]);

        iHead = iHead == 0 ? m_Buffer.size() - 1 : iHead - 1;
    }

    return szBuffer;
}


double CCircularScalarBuffer::sumCumProduct(const boost::numeric::ublas::matrix<double> &objWeights) const {

    double dSumProd = 0.0;
    int iHead = m_iCurrent;

    double dCumSum = 0.0;

    for (unsigned int i = 0; i != objWeights.size2(); ++i) {

        dCumSum += m_Buffer[i];
    }

    for (unsigned int i = 0; i != objWeights.size2(); ++i) {

        double dWtg = objWeights(0, i);

        dSumProd += dCumSum * dWtg;

        dCumSum -= m_Buffer[iHead];

        iHead = iHead == 0 ? m_Buffer.size() - 1 : iHead - 1;
    }

    return dSumProd;
}


std::tuple<double, double> CCircularScalarBuffer::getMinMax() const {

    double dMin = 0.0;
    double dMax = 0.0;

    getMinMax(dMin, dMax);

    return std::make_tuple(dMin, dMax);
}





