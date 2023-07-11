#include <iostream>
#include "bbtp.h"
#include <csignal>

tp::bpipe::BPIPE_Thread objApp;

void signal_handler(int signal)
{
    printf("received signal %d, stopping!", signal);
    objApp.kill();
    std::atomic_signal_fence(std::memory_order_release);
}

int main(int argc, char* argv[])
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    try {
        objApp.run();
    }
    catch(std::exception & e) {
        std::cout << "Exception thrown" << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}

