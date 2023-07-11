#include "CKeyValue.h"


using namespace Mts::OrderBook;


CKeyValue::CKeyValue() {

}


CKeyValue::CKeyValue(const Mts::Core::CDateTime & dtTimestamp,
										 const std::string &					strKey,
										 double												dValue)
: CEvent(KEY_VALUE),
	m_dtTimestamp(dtTimestamp),
	m_dValue(dValue) {

	strcpy(m_szKey, strKey.c_str());
}


const Mts::Core::CDateTime & CKeyValue::getTimestamp() const {

	return m_dtTimestamp;
}


const char * CKeyValue::getKey() const {

	return m_szKey;
}


double CKeyValue::getValue() const {

	return m_dValue;
}


std::string CKeyValue::toString() const {

	char szBuffer[1024];

	sprintf(szBuffer, "%llu %s %0.5f", m_dtTimestamp.getCMTime(), m_szKey, m_dValue);

	return szBuffer;
}



