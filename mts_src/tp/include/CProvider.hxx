using namespace Mts::Core;


inline unsigned int CProvider::getProviderID() const {

    return m_iProviderID;
}


inline std::string CProvider::getName() const {

    return m_strName;
}


inline std::string CProvider::getShortName() const {

    return m_strShortName;
}


inline double CProvider::getRTTMSec() const {

    return m_dRTTMSec;
}


inline double CProvider::getRTTDayFrac() const {

    return m_dRTTDayFrac;
}


inline double CProvider::getTicketFeeUSD() const {

    return m_dTicketFeeUSD;
}


inline double CProvider::getTicketFeeBps() const {

    return 10000.0 * m_dTicketFeeUSD / 1000000.0;
}

