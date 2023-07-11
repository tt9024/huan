#include <fstream>
#include <sstream>
#include <string>
#include "CFileUtil.h"


using namespace Mts::Core;


CFileUtil::CFileUtil() {

}


void CFileUtil::renameFile(const std::string & strFrom, const std::string & strTo) const {

	boost::filesystem::rename(strFrom, strTo);
}


std::vector<std::string> CFileUtil::getFiles(const std::string & strDirectory) const {

	std::vector<std::string> files;

	boost::filesystem::path objDir(strDirectory);
	boost::filesystem::directory_iterator endIter;

	if (boost::filesystem::exists(objDir) && boost::filesystem::is_directory(objDir)) {

		for (boost::filesystem::directory_iterator iter(objDir); iter != endIter; ++iter) {
			if (boost::filesystem::is_regular_file(iter->status()))
				files.push_back(iter->path().string());
		}
	}

	return files;
}


std::string CFileUtil::getFileNameFromPath(const std::string & strFileNameIncPath) const {

	if (strFileNameIncPath.find_last_of('\\') != std::string::npos)
		return std::string(strFileNameIncPath, strFileNameIncPath.find_last_of('\\')+1);
	else
		return std::string(strFileNameIncPath, strFileNameIncPath.find_last_of('/')+1);
}


std::vector<std::string> CFileUtil::readFileByLine(const std::string & strFileNameIncPath) const {

	std::ifstream infile(strFileNameIncPath.c_str());

	std::string strLine;
	std::vector<std::string> strLines;

	while (std::getline(infile, strLine)) {

		strLines.push_back(strLine);
	}

	return strLines;
}




