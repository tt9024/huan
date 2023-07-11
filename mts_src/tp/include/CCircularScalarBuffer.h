#ifndef CCIRCULARSCALARBUFFER_HEADER

#define CCIRCULARSCALARBUFFER_HEADER

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include "CDateTime.h"

namespace Mts {
    namespace Indicator {
        class CCircularScalarBuffer
        {
        public:
            CCircularScalarBuffer(unsigned int iSize,
                    unsigned int iIntervalMSec,
                    bool bDummy);

            CCircularScalarBuffer(unsigned int iSize = 100,
                    unsigned int iIntervalSec = 0);

            void preload(const Mts::Core::CDateTime &dtNow,
                    double dItem);

            void update(const Mts::Core::CDateTime &dtNow,
                    double dItem);

            void append(const Mts::Core::CDateTime &dtNow,
                    double dItem);

            void append(const Mts::Core::CDateTime &dtNow,
                    double dItem,
                    bool bPreload);

            double getHead() const;

            double getTail() const;

            double getPrevious() const;

            double sumProduct(double *pdWeights, int iSize) const;

            double sumProduct(const boost::numeric::ublas::matrix<double> &objWeights) const;

            double sumCumProduct(const boost::numeric::ublas::matrix<double> &objWeights) const;

            bool isFull() const;

            unsigned int getSize() const;

            double getItem(int iIndex) const;

            double getItemOffsetFromHead(int iIndex) const;

            std::tuple<double, double> getMinMax() const;

            void getMinMax(double &dMin,
                    double &dMax) const;

            double getMin() const;

            double getMax() const;

            double getSum() const;

            double getMean() const;

            double getStd() const;

            double getZScore() const;

            double getEWMA() const;

            void reset();

            std::string toString() const;

        private:
            typedef std::vector<double> Buffer;

            // buffer will be appended to every n milliseconds
            unsigned int m_iIntervalMSec;

            Mts::Core::CDateTime m_dtLastUpdateTime;

            Buffer m_Buffer;
            unsigned int m_iHead;
            unsigned int m_iCurrent;
            unsigned int m_iTail;

            // approximate average of items in the buffer using alpha = 1.5 * (1 / n)
            double m_dAlpha;
            double m_dEWMA;

            // true if the buffer is fully filled
            bool m_bFilled;
        };
    }
}

#endif


