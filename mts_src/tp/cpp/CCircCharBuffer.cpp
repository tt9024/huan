#include "CCircCharBuffer.h"


using namespace Mts::Core;


CCircCharBuffer::CCircCharBuffer(unsigned int iBufferSizeBytes)
        : m_iBufferSizeBytes(iBufferSizeBytes) {

    initialize();
}


CCircCharBuffer::CCircCharBuffer(const CCircCharBuffer &objRhs) {

    m_iBufferSizeBytes = objRhs.m_iBufferSizeBytes;

    m_pszBufferStart = new char[m_iBufferSizeBytes];
    memcpy(m_pszBufferStart, objRhs.m_pszBufferStart, m_iBufferSizeBytes);

    m_pszBufferEnd = m_pszBufferStart + m_iBufferSizeBytes;
    m_pszHead = m_pszBufferStart + (objRhs.m_pszHead - objRhs.m_pszBufferStart);
    m_pszTail = m_pszBufferStart + (objRhs.m_pszTail - objRhs.m_pszBufferStart);
    m_iNumItems = objRhs.m_iNumItems;
}


CCircCharBuffer &CCircCharBuffer::operator=(const CCircCharBuffer &objRhs) {

    if (this == &objRhs)
        return *this;

    if (m_pszBufferStart != nullptr)
        delete[] m_pszBufferStart;

    m_iBufferSizeBytes = objRhs.m_iBufferSizeBytes;

    m_pszBufferStart = new char[m_iBufferSizeBytes];
    memcpy(m_pszBufferStart, objRhs.m_pszBufferStart, m_iBufferSizeBytes);

    m_pszBufferEnd = m_pszBufferStart + m_iBufferSizeBytes;
    m_pszHead = m_pszBufferStart + (objRhs.m_pszHead - objRhs.m_pszBufferStart);
    m_pszTail = m_pszBufferStart + (objRhs.m_pszTail - objRhs.m_pszBufferStart);
    m_iNumItems = objRhs.m_iNumItems;

    return *this;
}


CCircCharBuffer::~CCircCharBuffer() {

    delete[] m_pszBufferStart;
}


void CCircCharBuffer::initialize() {

    m_pszBufferStart = new char[m_iBufferSizeBytes];
    m_pszBufferEnd = m_pszBufferStart + m_iBufferSizeBytes;
    m_pszHead = m_pszBufferStart;
    m_pszTail = m_pszBufferStart;
    m_iNumItems = 0;
}
