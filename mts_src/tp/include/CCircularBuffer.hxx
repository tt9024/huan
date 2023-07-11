using namespace Mts::Indicator;


template<class T>
CCircularBuffer<T>::CCircularBuffer(unsigned int iSize,
        unsigned int iIntervalSec)
        : m_Buffer(iSize),
          m_iIntervalSec(iIntervalSec),
          m_bFilled(false) {

    m_iHead = m_Buffer.size() - 1;
    m_iCurrent = m_iHead;
    m_iTail = 0;
}


template<class T>
void CCircularBuffer<T>::update(const Mts::Core::CDateTime &dtNow,
        const T &objItem) {

    m_Buffer[m_iHead] = objItem;
}


template<class T>
void CCircularBuffer<T>::append(const Mts::Core::CDateTime &dtNow,
        const T &objItem) {

    m_Buffer[m_iHead] = objItem;
    m_iCurrent = m_iHead;

    if (dtNow.getValue() - m_dtLastUpdateTime.getValue() >= static_cast<double>(m_iIntervalSec) / (24 * 3600)) {

        ++m_iHead;
        ++m_iTail;

        if (m_iHead == m_Buffer.size())
            m_iHead = 0;

        if (m_iTail == m_Buffer.size()) {
            m_bFilled = true;
            m_iTail = 0;
        }

        m_dtLastUpdateTime = dtNow;
    }
}


template<class T>
T CCircularBuffer<T>::getHead() const {

    return m_Buffer[m_iCurrent];
}


template<class T>
T CCircularBuffer<T>::getTail() const {

    int iIdxTail = m_iCurrent - (m_Buffer.size() - 1);

    return iIdxTail >= 0 ? m_Buffer[iIdxTail] : m_Buffer[iIdxTail + m_Buffer.size()];
}


template<class T>
T CCircularBuffer<T>::getPrevious() const {

    return m_iCurrent == 0 ? m_Buffer[m_Buffer.size() - 1] : m_Buffer[m_iCurrent - 1];
}


template<class T>
bool CCircularBuffer<T>::isFull() const {

    return m_bFilled;
}


template<class T>
unsigned int CCircularBuffer<T>::getSize() const {

    return m_Buffer.size();
}


template<class T>
T CCircularBuffer<T>::getItem(int iIndex) const {

    return m_iCurrent + iIndex < m_Buffer.size() ? m_Buffer[m_iCurrent + iIndex] : m_Buffer[m_iCurrent + iIndex -
                                                                                            m_Buffer.size()];
}


// index = 0 returns current item, index = 1 returns previous etc.
template<class T>
T CCircularBuffer<T>::getItemOffsetFromHead(int iIndex) const {

    return static_cast<int>(m_iCurrent) - iIndex >= 0 ? m_Buffer[m_iCurrent - iIndex] : m_Buffer[m_iCurrent - iIndex +
                                                                                                 m_Buffer.size()];
}


template<class T>
void CCircularBuffer<T>::reset() {

    m_bFilled = false;
    m_iHead = m_Buffer.size() - 1;
    m_iCurrent = m_iHead;
    m_iTail = 0;

    m_Buffer.clear();
}

