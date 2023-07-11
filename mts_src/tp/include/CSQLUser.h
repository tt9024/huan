#ifndef CSQLUSER_HEADER

#define CSQLUSER_HEADER

#include <string>

namespace Mts {
    namespace SQL {
        class CSQLUser
        {
        public:
            CSQLUser(const std::string &strUsername,
                    const std::string &strUID,
                    const std::string &strDefaultCcys,
                    const std::string &strAccessRights)
                    : m_strUsername(strUsername),
                      m_strUID(strUID),
                      m_strDefaultCcys(strDefaultCcys),
                      m_strAccessRights(strAccessRights) {}

            const std::string &getUsername() const { return m_strUsername; }

            const std::string &getUID() const { return m_strUID; }

            const std::string &getDefaultCcys() const { return m_strDefaultCcys; }

            const std::string &getAccessRights() const { return m_strAccessRights; }

        private:
            std::string m_strUsername;
            std::string m_strUID;
            std::string m_strDefaultCcys;
            std::string m_strAccessRights;
        };
    }
}

#endif

