#include <vector>
#include <sstream>
#include "CStringTokenizer.h"


using namespace Mts::Core;


std::vector<std::string> CStringTokenizer::splitString(const std::string &strDelimitedData,
        const std::string &strDelimiter) const {

    std::istringstream ss(strDelimitedData);
    std::string strToken;
    std::vector<std::string> objTokens;

    while (std::getline(ss, strToken, strDelimiter[0])) {
        objTokens.push_back(strToken);
    }

    return objTokens;
}

