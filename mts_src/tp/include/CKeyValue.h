#ifndef CKEYVALUE_HEADER

#define CKEYVALUE_HEADER

#include <cstring>
#include "CEvent.h"
#include "CDateTime.h"

namespace Mts {
    namespace OrderBook {
        class CKeyValue : public Mts::Event::CEvent
        {
        public:
            CKeyValue();

            CKeyValue(const Mts::Core::CDateTime &dtTimestamp,
                    const std::string &strKey,
                    double dValue);

            const Mts::Core::CDateTime &getTimestamp() const;

            const char *getKey() const;

            double getValue() const;

            std::string toString() const;

        private:
            Mts::Core::CDateTime m_dtTimestamp;
            char m_szKey[32];
            double m_dValue;
        };
    }
}

#endif

