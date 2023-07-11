#include <exception>
#include <string.h>
#include "CMtsException.h"


using namespace Mts::Core;


template<typename T>
inline bool CCircCharBuffer::push(const T &objData) {

    const char *pszData = reinterpret_cast<const char *>(&objData);
    unsigned int iNumBytesToWrite = sizeof(objData);
    return push(pszData, iNumBytesToWrite);
}


inline bool CCircCharBuffer::push(const char *pszData, unsigned int iNumBytesToWrite) {

    bool bOverRun = false;

    // no wrap around
    if (m_pszHead + iNumBytesToWrite < m_pszBufferEnd) {

        bOverRun = m_pszHead < m_pszTail;
        char *m_pszNewHead = m_pszHead + iNumBytesToWrite;
        bOverRun = bOverRun && m_pszNewHead > m_pszTail;

        if (bOverRun == false) {
            memcpy(m_pszHead, pszData, iNumBytesToWrite);
            m_pszHead = m_pszNewHead;
        }
    } else {

        // write wraps around end of buffer so requires two memcpy's
        size_t iFirstWriteSizeBytes = m_pszBufferEnd - m_pszHead;
        size_t iSecondWriteSizeBytes = iNumBytesToWrite - iFirstWriteSizeBytes;

        bOverRun = m_pszHead < m_pszTail;
        char *m_pszNewHead = m_pszBufferStart + iSecondWriteSizeBytes;
        bOverRun = bOverRun || m_pszNewHead > m_pszTail;

        if (bOverRun == false) {
            memcpy(m_pszHead, pszData, iFirstWriteSizeBytes);
            memcpy(m_pszBufferStart, pszData + iFirstWriteSizeBytes, iSecondWriteSizeBytes);
            m_pszHead = m_pszNewHead;
        }
    }

    if (bOverRun == true) {
        return false;
    }

    ++m_iNumItems;

    return true;
}


template<typename T>
inline void CCircCharBuffer::pop(T &objData, bool bPeeking) {

    char *pszData = reinterpret_cast<char *>(&objData);
    unsigned int iNumBytesToRead = sizeof(objData);
    pop(pszData, iNumBytesToRead, bPeeking);
}


inline void CCircCharBuffer::pop(char *pszData, unsigned int iNumBytesToRead, bool bPeeking) {

    if (bPeeking == false)
        --m_iNumItems;

    // no wrap around
    if (m_pszTail + iNumBytesToRead < m_pszBufferEnd) {
        memcpy(pszData, m_pszTail, iNumBytesToRead);

        if (bPeeking == false)
            m_pszTail += iNumBytesToRead;

        return;
    }

    // read wraps around end of buffer so requires two memcpy's
    size_t iFirstReadSizeBytes = m_pszBufferEnd - m_pszTail;
    size_t iSecondReadSizeBytes = iNumBytesToRead - iFirstReadSizeBytes;

    memcpy(pszData, m_pszTail, iFirstReadSizeBytes);
    memcpy(pszData + iFirstReadSizeBytes, m_pszBufferStart, iSecondReadSizeBytes);

    if (bPeeking == false)
        m_pszTail = m_pszBufferStart + iSecondReadSizeBytes;
}


template<typename T>
inline void CCircCharBuffer::pop(T &objData) {

    pop(objData, false);
}


template<typename T>
inline void CCircCharBuffer::peek(T &objData) {

    pop(objData, true);
}


inline bool CCircCharBuffer::isEmpty() const {

    return m_pszHead == m_pszTail;
}


inline unsigned int CCircCharBuffer::getNumItems() const {

    return m_iNumItems;
}


inline bool CCircCharBuffer::checkForBufferOverRun(unsigned int iNumBytesToWrite) {

    bool bOverRun = false;

    // no wrap around
    if (m_pszHead + iNumBytesToWrite < m_pszBufferEnd) {

        bOverRun = m_pszHead < m_pszTail;
        char *m_pszNewHead = m_pszHead + iNumBytesToWrite;
        bOverRun = bOverRun && m_pszNewHead > m_pszTail;
    } else {

        // write wraps around end of buffer so requires two memcpy's
        size_t iFirstWriteSizeBytes = m_pszBufferEnd - m_pszHead;
        size_t iSecondWriteSizeBytes = iNumBytesToWrite - iFirstWriteSizeBytes;

        bOverRun = m_pszHead < m_pszTail;
        char *m_pszNewHead = m_pszBufferStart + iSecondWriteSizeBytes;
        bOverRun = bOverRun || m_pszNewHead > m_pszTail;
    }

    return bOverRun;
}


template<typename T>
inline void CCircCharBuffer::peek(unsigned int iIndex,
        T &objData) {

    pop(objData, true);
}


