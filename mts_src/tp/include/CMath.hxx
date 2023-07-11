#include <math.h>


using namespace Mts::Math;


template<typename T>
inline T CMath::min(const T &objLhs,
        const T &objRhs) {

    if (objLhs < objRhs)
        return objLhs;
    else
        return objRhs;
}


template<typename T>
inline T CMath::max(const T &objLhs,
        const T &objRhs) {

    if (objLhs > objRhs)
        return objLhs;
    else
        return objRhs;
}


template<typename T>
inline int CMath::sign(const T &objVar) {

    if (objVar > 0)
        return 1;

    if (objVar < 0)
        return -1;

    return 0;
}


inline int CMath::atoi(const std::string &s) {

    return atoi(s.c_str());
}


inline double CMath::atof(const std::string &s) {

    return atof(s.c_str());
}


inline int CMath::atoi(const char *p) {

    int r = 0;
    bool neg = false;

    if (*p == '-') {
        neg = true;
        ++p;
    }

    while (*p >= '0' && *p <= '9') {
        r = (r * 10) + (*p - '0');
        ++p;
    }

    if (neg) {
        r = -r;
    }

    return r;
}


inline double CMath::atof(const char *p) {

    double r = 0.0;
    bool neg = false;

    if (*p == '-') {
        neg = true;
        ++p;
    }

    while (*p >= '0' && *p <= '9') {
        r = (r * 10.0) + (*p - '0');
        ++p;
    }

    if (*p == '.') {
        double f = 0.0;
        int n = 0;
        ++p;
        while (*p >= '0' && *p <= '9') {
            f = (f * 10.0) + (*p - '0');
            ++p;
            ++n;
        }
        r += f / pow(10.0, n);
    }

    if (neg) {
        r = -r;
    }

    return r;
}


inline int CMath::atoi_1(const std::string &s) {

    return atoi_1(s.c_str());
}


inline int CMath::atoi_2(const std::string &s) {

    return atoi_2(s.c_str());
}


inline int CMath::atoi_3(const std::string &s) {

    return atoi_3(s.c_str());
}


inline int CMath::atoi_4(const std::string &s) {

    return atoi_4(s.c_str());
}


inline int CMath::atoi_1(const char *p) {

    return *p - '0';
}


inline int CMath::atoi_2(const char *p) {

    return 10 * (*p - '0') + (*(p + 1) - '0');
}


inline int CMath::atoi_3(const char *p) {

    return 100 * (*p - '0') + 10 * (*(p + 1) - '0') + (*(p + 2) - '0');
}


inline int CMath::atoi_4(const char *p) {

    return 1000 * (*p - '0') + 100 * (*(p + 1) - '0') + 10 * (*(p + 2) - '0') + (*(p + 3) - '0');
}

