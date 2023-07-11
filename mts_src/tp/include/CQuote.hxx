using namespace Mts::OrderBook;


inline unsigned int CQuote::getSymbolID() const {

    return m_iSymbolID;
}


inline unsigned int CQuote::getProviderID() const {

    return m_iProviderID;
}


inline const Mts::Core::CDateTime &CQuote::getMtsTimestamp() const {

    return m_dtMtsTimestamp;
}


inline const Mts::Core::CDateTime &CQuote::getExcTimestamp() const {

    return m_dtExcTimestamp;
}


inline CQuote::Side CQuote::getSide() const {

    return m_iSide;
}


inline double CQuote::getPrice() const {

    return m_dPrice;
}


inline unsigned int CQuote::getSize() const {

    return m_iSize;
}


inline std::string CQuote::getExch() const {

    return m_szExch;
}


inline std::string CQuote::getExDest() const {

    return m_szExDest;
}


inline void CQuote::setSymbolID(unsigned int iSymbolID) {

    m_iSymbolID = iSymbolID;
}


inline void CQuote::setProviderID(unsigned int iProviderID) {

    m_iProviderID = iProviderID;
}


inline void CQuote::setMtsTimestamp(const Mts::Core::CDateTime &dtMtsTimestamp) {

    m_dtMtsTimestamp = dtMtsTimestamp;
}


inline void CQuote::setExcTimestamp(const Mts::Core::CDateTime &dtExcTimestamp) {

    m_dtExcTimestamp = dtExcTimestamp;
}


inline void CQuote::setSide(Side iSide) {

    m_iSide = iSide;
}


inline void CQuote::setPrice(double dPrice) {

    m_dPrice = dPrice;
}


inline void CQuote::setSize(unsigned int iSize) {

    m_iSize = iSize;
}


inline void CQuote::setExch(const std::string &strExch) {

    strcpy(m_szExch, strExch.c_str());
}


inline void CQuote::setExDest(const std::string &strExDest) {

    strcpy(m_szExDest, strExDest.c_str());
}

