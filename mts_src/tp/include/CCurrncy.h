#ifndef CCURRNCY_HEADER

#define CCURRNCY_HEADER

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

namespace Mts {
    namespace Core {
        class CCurrncy
        {
        public:
            static bool load(const std::string &strXML);

            static const CCurrncy &getCcy(const std::string &strCcy);

            static const CCurrncy &getCcy(unsigned int iCcyID);

            static std::vector<boost::shared_ptr<const CCurrncy> > getCcys();

            static bool isCcy(const std::string &strCcy);

            static bool isCcy(unsigned int iCcyID);

            CCurrncy();

            CCurrncy(unsigned int iCcyID,
                    const std::string &strCcy,
                    double dCcy2USDExcRate);

            unsigned int getCcyID() const;

            std::string getCcy() const;

            double getCcy2USDExcRate() const;

        private:
            // stored in duplicate with different indices for faster lookup
            typedef boost::unordered_map<std::string, boost::shared_ptr<CCurrncy> > CcyMapByName;
            typedef boost::unordered_map<unsigned int, boost::shared_ptr<CCurrncy> > CcyMapByID;

            static CcyMapByName m_CcysByName;
            static CcyMapByID m_CcysByID;

            unsigned int m_iCcyID;
            std::string m_strCcy;
            double m_dCcy2USDExcRate;
        };
    }
}

#endif


