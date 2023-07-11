#ifndef CPROVIDER_HEADER

#define CPROVIDER_HEADER

#include <string>
#include <vector>
#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>

namespace Mts {
    namespace Core {
        class CProvider
        {
        public:
            static bool load(const std::string &strXML);

            static const CProvider &getProvider(const std::string &strName);

            static const CProvider &getProvider(unsigned int iProviderID);

            static std::vector<boost::shared_ptr<const CProvider> > getProviders();

            static bool isProvider(unsigned int iProviderID);

            static bool isProvider(std::string strName);

            CProvider(unsigned int iProviderID,
                    const std::string &strName,
                    const std::string &strShortName,
                    double dRTTMSec,
                    double dTicketFeeUSD);

            unsigned int getProviderID() const;

            std::string getName() const;

            std::string getShortName() const;

            double getRTTMSec() const;

            double getRTTDayFrac() const;

            double getTicketFeeUSD() const;

            double getTicketFeeBps() const;

        private:
            typedef boost::unordered_map<std::string, boost::shared_ptr<CProvider> > ProviderMap;
            typedef boost::unordered_map<unsigned int, boost::shared_ptr<CProvider> > ProviderMapByID;

            static ProviderMap m_Providers;
            static ProviderMapByID m_ProvidersByID;

            unsigned int m_iProviderID;
            std::string m_strName;
            std::string m_strShortName;

            // round trip latency in millis and day fraction
            double m_dRTTMSec;
            double m_dRTTDayFrac;

            // per million ticket cost
            double m_dTicketFeeUSD;
        };
    }
}

#include "CProvider.hxx"

#endif

