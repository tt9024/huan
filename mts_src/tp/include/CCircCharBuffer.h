#ifndef CCIRCCHARBUFFER_HEADER

#define CCIRCCHARBUFFER_HEADER

namespace Mts {
    namespace Core {
        class CCircCharBuffer
        {
        public:
            CCircCharBuffer(unsigned int iBufferSizeBytes);

            ~CCircCharBuffer();

            void initialize();

            bool checkForBufferOverRun(unsigned int iNumBytesToWrite);

            template<typename T>
            bool push(const T &objData);

            template<typename T>
            void pop(T &objData, bool bPeeking);

            template<typename T>
            void pop(T &objData);

            template<typename T>
            void peek(T &objData);

            template<typename T>
            void peek(unsigned int iIndex,
                    T &objData);

            bool push(const char *pszData,
                    unsigned int iNumBytesToWrite);

            void pop(char *pszData,
                    unsigned int iNumBytesToRead,
                    bool bPeeking);

            bool isEmpty() const;

            unsigned int getNumItems() const;

            CCircCharBuffer(const CCircCharBuffer &objRhs);

            CCircCharBuffer &operator=(const CCircCharBuffer &objRhs);

        private:
            char *m_pszBufferStart;
            char *m_pszBufferEnd;

            // pointer to address of next read
            char *m_pszHead;

            // pointer to address of next write
            char *m_pszTail;

            // number of items in buffer
            unsigned int m_iNumItems;

            // buffer size
            unsigned int m_iBufferSizeBytes;
        };
    }
}

#include "CCircCharBuffer.hxx"

#endif

