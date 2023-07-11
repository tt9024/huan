#include "AlgoThread.h"
#include <memory>
#include <csignal>
#include "plcc/PLCC.hpp"

namespace {
    std::shared_ptr<algo::AlgoThread> g_thread;
}

void signal_handler(int signal)
{
    if (g_thread) {
        g_thread->stopAll();
        std::atomic_signal_fence(std::memory_order_release);
        logInfo("received signal %d, stop all!", signal);
    }
}

int main(int argc, char** argv) {
    const char* instance = "strat";
    if (argc == 2) {
        instance = argv[1];
    }

    const std::string cfg_file = plcc_getString("Strat");
    g_thread = std::make_shared<algo::AlgoThread>(instance, cfg_file);

    // run on the main thread
    g_thread->run();

    logInfo("%s done!", argv[1]);
    return 0;
}
