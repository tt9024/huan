#include "async_io.h"
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <memory.h>
#include <string.h>

int main() {
    char line[16];
    while (true) {
        size_t len;
        printf("Checking...\n");
        while (utils::getline_nb(line, 16, &len)) {
            printf("Got (%d) line: %s", (int)len, line);
        }
        sleep(2);
    };
    return 0;
}
