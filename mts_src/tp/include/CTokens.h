#ifndef CTOKENS_HEADER

#define CTOKENS_HEADER

#include <cstring>

namespace Mts {
    namespace Core {
        struct CTokens : std::ctype<char>
        {
            CTokens() : std::ctype<char>(get_table()) {}

            static std::ctype_base::mask const *get_table() {
                typedef std::ctype<char> cctype;
                static const cctype::mask *const_rc = cctype::classic_table();

                static cctype::mask rc[cctype::table_size];
                std::memcpy(rc, const_rc, cctype::table_size * sizeof(cctype::mask));

                rc[','] = std::ctype_base::space;
                rc[' '] = std::ctype_base::space;
                return &rc[0];
            }
        };
    }
}

#endif




