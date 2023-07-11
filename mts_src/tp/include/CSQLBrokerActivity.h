#pragma once

#include <string>

namespace Mts {
    namespace SQL {
        class CSQLBrokerActivity
        {
        public:
            CSQLBrokerActivity(const std::string &strBroker,
                    unsigned int iCount)
                    : m_strBroker(strBroker),
                      m_iCount(iCount) {}

            std::string getBroker() const { return m_strBroker; }

            unsigned int getCount() const { return m_iCount; }

        private:
            std::string m_strBroker;
            unsigned int m_iCount;
        };
    }
}
