#include "StratSim.h"
#include <memory>
#include <csignal>
#include "plcc/PLCC.hpp"

namespace {
    std::shared_ptr<algo::StratSim> g_thread;
}

void signal_handler(int signal)
{
    if (g_thread) {
        logInfo("received signal %d, stopping all!", signal);
        g_thread->stop();
        std::atomic_signal_fence(std::memory_order_release);
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("Usage: %s trade_day [cfg_file, default config/main.cfg]\n", argv[0]);
        return -1;
    }

    const char* trade_day = argv[1];
    const char* cfg_file = "config/main.cfg";
    if (argc == 3) {
        cfg_file = argv[2];
    }

    unsigned long long cur_micro = (utils::TimeUtil::string_to_frac_UTC(trade_day, 0, "%Y%m%d")-6*3600)*1000000ULL;
    // setup current time for time util
    utils::TimeUtil::set_cur_time_micro(cur_micro);

    logInfo("Running simulation on %s, config file: %s\n",  trade_day, cfg_file );
    g_thread = std::make_shared<algo::StratSim>(trade_day, cfg_file);

    // run on the main thread
    g_thread->run();

    return 0;
}
