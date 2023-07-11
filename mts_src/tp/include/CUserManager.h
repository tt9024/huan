#ifndef CUSERMANAGER_HEADER

#define CUSERMANAGER_HEADER

#include <vector>
#include <boost/unordered_map.hpp>
#include "CUser.h"

namespace Mts {
    namespace User {
        class CUserManager
        {
        public:
            static CUserManager &getInstance();

            static void setModeSQL(bool bUseSQL);

            CUserManager(const CUserManager &objRhs) = delete;

            void initialize();

            Mts::User::CUser getUser(const std::string &strUID) const;

            bool isUser(const std::string &strUID) const;

        private:
            CUserManager();

        private:
            typedef boost::unordered_map<std::string, Mts::User::CUser> UID2UserMap;
            typedef std::vector<Mts::User::CUser> UserList;

            UID2UserMap m_UserMap;
            UserList m_UserList;

            static bool m_bUseSQL;
        };
    }
}

#endif


