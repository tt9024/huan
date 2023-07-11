#include "PLCC.hpp"

#define ConfigFile "config.txt"

using namespace utils;

int main() {
    char buf[1024*8];
    memset(buf, 64, sizeof(buf));
    buf[sizeof(buf)-1] = 0;
    logInfo("long value: %s", buf);

    buf[1] = 0;
    for (int i=0; i<10000; i++) {
        logInfo("burst msg %d: %s - should only see 100", i, buf);
    }
    return 0;
}
