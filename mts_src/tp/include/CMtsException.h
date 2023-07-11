#ifndef CMtsEXCEPTION_HEADER

#define CMtsEXCEPTION_HEADER

#include <string>
#include <stdexcept>
#include "CExcept.h"

namespace Mts {
    namespace Exception {
        class CMtsException : public std::exception
        {
        public:
            CMtsException(const std::string &strErrorMsg);

            ~CMtsException() CNOEXCEPT;

            virtual const char *what() const CNOEXCEPT;

        private:
            std::string m_strErrorMsg;
        };
    }
}

#endif
