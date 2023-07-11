#include "CApplicationLog.h"


using namespace Mts::Log;


template<typename T>
void CLogBinary::writeToFile(const T &objDataItem) {

    try {

        unsigned int iNumBytesToWrite = sizeof(objDataItem);
        fwrite(reinterpret_cast<const void *>(&iNumBytesToWrite), sizeof(iNumBytesToWrite), 1, m_ptrFile);
        fwrite(reinterpret_cast<const void *>(&objDataItem), sizeof(objDataItem), 1, m_ptrFile);
        fflush(m_ptrFile);
    }
    catch (std::exception &e) {

        AppLogError(e.what());
    }
}


// file is expected to contain objects of the same type but each can differ in size
template<typename T>
void CLogBinary::readFromFile(std::vector<T> &dataItems) {

    //std::vector<T> dataItems;

    try {

        long iFilePos = 0;
        long iFileSize = 0;

        fseek(m_ptrFile, 0, SEEK_END);
        iFileSize = ftell(m_ptrFile);
        rewind(m_ptrFile);

        while (iFilePos < iFileSize) {
            unsigned int iNumBytesToRead = 0;

            fseek(m_ptrFile, iFilePos, SEEK_SET);
            fread(reinterpret_cast<void *>(&iNumBytesToRead), sizeof(unsigned int), 1, m_ptrFile);

            iFilePos += sizeof(unsigned int);
            T objDataItem;

            fseek(m_ptrFile, iFilePos, SEEK_SET);
            fread(reinterpret_cast<void *>(&objDataItem), sizeof(T), 1, m_ptrFile);

            dataItems.push_back(objDataItem);
            iFilePos += sizeof(objDataItem);
        }
    }
    catch (std::exception &e) {

        AppLogError(e.what());
    }
}


// all records expected to be same size
template<typename T>
void CLogBinary::writeToFileHomogenous(const T &objDataItem) {

    try {

        fwrite(reinterpret_cast<const void *>(&objDataItem), sizeof(objDataItem), 1, m_ptrFile);
        fflush(m_ptrFile);
    }
    catch (std::exception &e) {

    }
}


// all records expected to be same size
template<typename T>
void CLogBinary::readFromFileHomogenous(T &objDataItem,
        long iRecNum) {

    try {

        long iFilePos = sizeof(objDataItem) * iRecNum;

        fseek(m_ptrFile, iFilePos, SEEK_SET);
        fread(reinterpret_cast<void *>(&objDataItem), sizeof(T), 1, m_ptrFile);
    }
    catch (std::exception &e) {

    }
}


