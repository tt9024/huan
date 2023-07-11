#ifndef CCIRCULARBUFFER_HEADER

#define CCIRCULARBUFFER_HEADER

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include "CDateTime.h"

namespace Mts {
    namespace Indicator {
        template<class T>
        class CCircularBuffer
        {
        public:
            CCircularBuffer(unsigned int iSize = 100,
                    unsigned int iIntervalSec = 0);

            void update(const Mts::Core::CDateTime &dtNow,
                    const T &objItem);

            void append(const Mts::Core::CDateTime &dtNow,
                    const T &objItem);

            T getHead() const;

            T getTail() const;

            T getPrevious() const;

            bool isFull() const;

            unsigned int getSize() const;

            T getItem(int iIndex) const;

            T getItemOffsetFromHead(int iIndex) const;

            void reset();

        private:
            typedef std::vector<T> Buffer;

            // buffer will be appended to every n milliseconds
            unsigned int m_iIntervalSec;

            Mts::Core::CDateTime m_dtLastUpdateTime;

            Buffer m_Buffer;
            unsigned int m_iHead;
            unsigned int m_iCurrent;
            unsigned int m_iTail;

            // true if the buffer is fully filled
            bool m_bFilled;
        };
    }
}

#include "CCircularBuffer.hxx"

#endif


