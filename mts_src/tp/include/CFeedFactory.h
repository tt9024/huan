#ifndef CFEEDFACTORY_HEADER

#define CFEEDFACTORY_HEADER

#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>
#include "CFeed.h"

namespace Mts
{
	namespace Feed
	{
		class CFeedFactory
		{
		public:
			static CFeedFactory & getInstance();

			CFeedFactory() = default;
			CFeedFactory(const CFeedFactory & objRhs) = delete;
			
			boost::shared_ptr<CFeed> createFeed(const std::string & strFeedDefXML);

		private:
			boost::shared_ptr<CFeed> createFeedEnginePersistedTOB(const boost::property_tree::ptree & objPTree);
			boost::shared_ptr<CFeed> createFeedTickDataTOB(const boost::property_tree::ptree & objPTree);
			boost::shared_ptr<CFeed> createFeedFIX_TT(const boost::property_tree::ptree & objPTree);
			//boost::shared_ptr<CFeed> createFeedPollDB(const boost::property_tree::ptree & objPTree);
		};
	}
}

#endif

