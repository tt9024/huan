using namespace Mts::Core;


inline unsigned int CSymbol::getSymbolID() const {

    return m_iSymbolID;
}


inline std::string CSymbol::getSymbol() const {

    return m_strSymbol;
}


inline const CCurrncy &CSymbol::getBaseCcy() const {

    return m_objBaseCcy;
}


inline const CCurrncy &CSymbol::getRefCcy() const {

    return m_objRefCcy;
}


inline double CSymbol::getPointValue() const {

    return m_dPointValue;
}


inline double CSymbol::getTickSize() const {

    return m_dTickSize;
}


inline CSymbol::AssetClass CSymbol::getAssetClass() const {

    return m_iAssetClass;
}


inline std::string CSymbol::getExchSymbol() const {

    return m_strExchSymbol;
}


inline std::string CSymbol::getTTExchange() const {

    return m_strTTExchange;
}

inline std::string CSymbol::getExchange() const {

    return m_strExchange;
}


inline std::string CSymbol::getSecurityType() const {

    return m_strSecurityType;
}


inline std::tuple<std::string, std::string> CSymbol::getCcy() const {

    return std::make_tuple(m_objBaseCcy.getCcy(), m_objRefCcy.getCcy());
}


inline std::string CSymbol::getTTSymbol() const {

    return m_strTTSymbol;
}


inline double CSymbol::getTTScaleMultiplier() const {

    return m_dTTScaleMultiplier;
}


inline std::string CSymbol::getTDSymbol() const {

    return m_strTDSymbol;
}


inline double CSymbol::getTDScaleMultiplier() const {

    return m_dTDScaleMultiplier;
}


inline unsigned int CSymbol::getLotsPerMin() const {

    return m_iLotsPerMin;
}


inline unsigned int CSymbol::getTimeSliceSecs() const {

    return m_iTimeSliceSecs;
}


inline std::string CSymbol::getTTSecID() const {

    return m_strTTSecID;
}

inline void CSymbol::setTTSecID(const std::string &strTTSecID) {

    m_strTTSecID = strTTSecID;
}

inline std::string CSymbol::getContractExchSymbol() const {
    return m_strContractExchSymbol;
}

inline std::string CSymbol::getContractTicker() const {
    return m_strContractTicker;
}

inline std::string CSymbol::getContractMonth() const {
    return m_strContractMonth;
}

inline std::string CSymbol::toString() const {
    char buf[512];
    snprintf(buf, sizeof(buf), "sym=%s,exsym=%s,exch=%s,sectype=%s, assetcls=%d",
             getSymbol().c_str(),
             getExchSymbol().c_str(),
             getExchange().c_str(),
             getSecurityType().c_str(),
             (int) getAssetClass());
    return std::string(buf);
}


