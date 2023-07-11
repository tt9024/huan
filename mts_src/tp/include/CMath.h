#ifndef CMATH_HEADER

#define CMATH_HEADER

#include <string>

namespace Mts {
    namespace Math {
        class CMath
        {
        public:
            template<typename T>
            static T min(const T &objLhs,
                    const T &objRhs);

            template<typename T>
            static T max(const T &objLhs,
                    const T &objRhs);

            template<typename T>
            static int sign(const T &objVar);

            static int atoi(const std::string &s);

            static double atof(const std::string &s);

            static int atoi(const char *p);

            static double atof(const char *p);

            static int atoi_1(const std::string &s);

            static int atoi_2(const std::string &s);

            static int atoi_3(const std::string &s);

            static int atoi_4(const std::string &s);

            static int atoi_1(const char *p);

            static int atoi_2(const char *p);

            static int atoi_3(const char *p);

            static int atoi_4(const char *p);
        };
    }
}

#include "CMath.hxx"

#endif


