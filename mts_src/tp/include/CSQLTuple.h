#ifndef CSQLTUPLE_HEADER

#define CSQLTUPLE_HEADER

#include "CDateTime.h"

namespace Mts {
    namespace SQL {
        class CSQLTuple
        {
        public:
            CSQLTuple()
                    : m_dtTimestamp(0),
                      m_strKey(""),
                      m_dValue(0.0) {}

            CSQLTuple(const Mts::Core::CDateTime &dtTimestamp,
                    const std::string &strKey,
                    double dValue)
                    : m_dtTimestamp(dtTimestamp),
                      m_strKey(strKey),
                      m_dValue(dValue) {}

            Mts::Core::CDateTime getTimestamp() const { return m_dtTimestamp; }

            std::string getKey() const { return m_strKey; }

            double getValue() const { return m_dValue; }

        private:
            Mts::Core::CDateTime m_dtTimestamp;
            std::string m_strKey;
            double m_dValue;
        };
    }
}

#endif

