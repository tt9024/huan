using namespace Mts::Core;


inline const Mts::Core::CDateTime &CDateTimeEvent::getTimestamp() const {

    return m_dtTimestamp;
}


inline std::string CDateTimeEvent::toString() const {

    return m_dtTimestamp.toString();
}

