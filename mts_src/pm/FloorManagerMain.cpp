#include "FloorManager.h"
#include <csignal>
#include <atomic>

namespace
{
    pm::FloorManager& fmgr = pm::FloorManager::get();
}

void signal_handler(int signal)
{
    fmgr.stop();
    std::atomic_signal_fence(std::memory_order_release);
    logInfo("received signal %d, floor manager stop() called...\ndump_state\n%s", 
            signal, fmgr.toString().c_str());
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    logInfo("starting at %s", utils::TimeUtil::frac_UTC_to_string(0,0).c_str());
    fmgr.start();
    logInfo("exit at %s", utils::TimeUtil::frac_UTC_to_string(0,0).c_str());
    return 0;
}
