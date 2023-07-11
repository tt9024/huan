#ifndef CFILEUTIL_HEADER

#define CFILEUTIL_HEADER

#include <vector>
#include <string>
#include <boost/filesystem.hpp>

namespace Mts
{
	namespace Core
	{
		class CFileUtil
		{
		public:

			CFileUtil();
			std::vector<std::string> getFiles(const std::string & strDirectory) const;
			std::string getFileNameFromPath(const std::string & strFileNameIncPath) const;
			std::vector<std::string> readFileByLine(const std::string & strFileNameIncPath) const;
			void renameFile(const std::string & strFrom, const std::string & strTo) const;

		private:
		};
	}
}

#endif

