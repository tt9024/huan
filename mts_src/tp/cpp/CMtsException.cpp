#include <cstring>
#include "CMtsException.h"


using namespace Mts::Exception;


CMtsException::CMtsException(const std::string &strErrorMsg)
        : m_strErrorMsg(strErrorMsg) {

}


CMtsException::~CMtsException() throw() {

}


const char *CMtsException::what() const throw() {

    return m_strErrorMsg.c_str();
}

