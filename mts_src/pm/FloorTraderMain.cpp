#include "FloorTrader.h"
#include <csignal>
#include <stdio.h>
#include <atomic>

namespace
{
    pm::FloorTrader* __ftrd = nullptr;
}

void signal_handler(int signal)
{
    if (!__ftrd) return;
    __ftrd->stop();
    std::atomic_signal_fence(std::memory_order_release);
    logInfo("received signal %d, floor manager stop() called...\ndump_state\n%s", 
            signal, __ftrd->toString().c_str());
    __ftrd=nullptr;
}

int main(int argc, char**argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    if (argc != 2) {
        printf("Usage: %s instance_name\n", argv[0]);
        return 1;
    };
    pm::FloorTrader ftrd (argv[1]);
    __ftrd = &ftrd;
    logInfo("Trader %s starting at %s", argv[1], utils::TimeUtil::frac_UTC_to_string(0,0).c_str());
    ftrd.start();
    logInfo("Trader %s exit at %s", argv[1], utils::TimeUtil::frac_UTC_to_string(0,0).c_str());
    return 0;
}
