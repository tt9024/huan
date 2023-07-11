#ifndef CDATETIMELESSTHAN_HEADER

#define CDATETIMELESSTHAN_HEADER

#include <functional>
#include "CDateTime.h"

namespace Mts {
    namespace Core {
        class CDateTimeLessThan : public std::binary_function<CDateTime, CDateTime, bool>
        {
        public:
            CDateTimeLessThan() {}

            bool operator()(const Mts::Core::CDateTime &dLhs,
                    const Mts::Core::CDateTime &dRhs) const {

                return dLhs.getCMTime() < dRhs.getCMTime();
            }
        };
    }
}

#endif

