#ifndef CFEEDMANAGER_HEADER

#define CFEEDMANAGER_HEADER

#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>
#include "CFeed.h"

namespace Mts {
    namespace Feed {
        class CFeedManager
        {
        public:

            static CFeedManager &getInstance();

            void addFeed(const std::string &strFeedName,
                    boost::shared_ptr<Mts::Feed::CFeed> ptrFeed);

            boost::shared_ptr<Mts::Feed::CFeed> getFeed(const std::string &strFeedName) const;

        private:
            CFeedManager();

            CFeedManager(const CFeedManager &objRhs);

        private:
            typedef boost::unordered_map<std::string, boost::shared_ptr<Mts::Feed::CFeed> > FeedName2FeedMap;

            FeedName2FeedMap m_FeedName2FeedMap;
        };
    }
}

#endif

