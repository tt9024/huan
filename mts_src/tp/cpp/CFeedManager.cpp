#include "CFeedManager.h"


using namespace Mts::Feed;


CFeedManager &CFeedManager::getInstance() {

    static CFeedManager theInstance;
    return theInstance;
}


CFeedManager::CFeedManager() {

}


void CFeedManager::addFeed(const std::string &strFeedName,
        boost::shared_ptr<Mts::Feed::CFeed> ptrFeed) {

    m_FeedName2FeedMap.insert(std::pair<std::string, boost::shared_ptr<Mts::Feed::CFeed> >(strFeedName, ptrFeed));
}


boost::shared_ptr<Mts::Feed::CFeed> CFeedManager::getFeed(const std::string &strFeedName) const {

    FeedName2FeedMap::const_iterator iter = m_FeedName2FeedMap.find(strFeedName);

    if (iter == m_FeedName2FeedMap.end())
        return boost::shared_ptr<Mts::Feed::CFeed>();
    else
        return iter->second;
}



