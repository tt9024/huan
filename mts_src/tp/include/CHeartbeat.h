#ifndef CHEARTBEAT_HEADER

#define CHEARTBEAT_HEADER

#include "CProvider.h"
#include "CDateTime.h"

namespace Mts {
    namespace LifeCycle {
        class CHeartbeat
        {
        public:
            CHeartbeat();

            CHeartbeat(unsigned int iProviderID,
                    bool bFeedFlag,
                    double dJulianDateTime);

            unsigned int getProviderID() const;

            void setProviderID(unsigned int iProviderID);

            bool isFeed() const;

            void setIsFeed(bool bFeedFlag);

            double getJulianDateTime() const;

            void setJulianDateTime(double dJulianDateTime);

            bool isActive() const;

            void setActiveFlag(bool bActiveFlag);

        private:
            unsigned int m_iProviderID;
            bool m_bFeedFlag;
            double m_dJulianDateTime;
            bool m_bActiveFlag;
        };
    }
}

#endif

