using namespace Mts::Log;


template<typename T>
void CLogBinaryBuffered::log(const T &objData) {

    try {

        boost::mutex::scoped_lock lock(m_Mutex);
        unsigned int iNumBytes = sizeof(objData);

        m_ptrCurrentBuffer->push(reinterpret_cast<const char *>(&iNumBytes), sizeof(iNumBytes));
        m_ptrCurrentBuffer->push(reinterpret_cast<const char *>(&objData), iNumBytes);
    }
    catch (std::exception &e) {
        throw e;
    }
}


// file is expected to contain objects of the same type but each can differ in size
template<typename T>
void CLogBinaryBuffered::readFromFile(std::vector<T> dataItems) {

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

