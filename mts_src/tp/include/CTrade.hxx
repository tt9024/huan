using namespace Mts::OrderBook;


inline unsigned int CTrade::getSymbolID() const {

    return m_iSymbolID;
}


inline unsigned int CTrade::getProviderID() const {

    return m_iProviderID;
}


inline const Mts::Core::CDateTime &CTrade::getMtsTimestamp() const {

    return m_dtMtsTimestamp;
}


inline const Mts::Core::CDateTime &CTrade::getExcTimestamp() const {

    return m_dtExcTimestamp;
}


inline double CTrade::getPrice() const {

    return m_dPrice;
}


inline unsigned int CTrade::getSize() const {

    return m_iSize;
}


inline std::string CTrade::getExch() const {

    return m_szExch;
}


inline std::string CTrade::getExDest() const {

    return m_szExDest;
}


inline void CTrade::setSymbolID(unsigned int iSymbolID) {

    m_iSymbolID = iSymbolID;
}


inline void CTrade::setProviderID(unsigned int iProviderID) {

    m_iProviderID = iProviderID;
}


inline void CTrade::setMtsTimestamp(const Mts::Core::CDateTime &dtMtsTimestamp) {

    m_dtMtsTimestamp = dtMtsTimestamp;
}


inline void CTrade::setExcTimestamp(const Mts::Core::CDateTime &dtExcTimestamp) {

    m_dtExcTimestamp = dtExcTimestamp;
}


inline void CTrade::setPrice(double dPrice) {

    m_dPrice = dPrice;
}


inline void CTrade::setSize(unsigned int iSize) {

    m_iSize = iSize;
}


inline void CTrade::setExch(const std::string &strExch) {

    strcpy(m_szExch, strExch.c_str());
}


inline void CTrade::setExDest(const std::string &strExDest) {

    strcpy(m_szExDest, strExDest.c_str());
}

