#include <iostream>
#include "CApplication.h"
#include <csignal>

#include <Version.h>

namespace
{
    //Mts::Application::CApplication objApp(MTS_VERSION_HASH);
    Mts::Application::CApplication objApp(std::string{MTS_VERSION});
}

void signal_handler(int signal)
{
    objApp.stop();
    std::atomic_signal_fence(std::memory_order_release);
}

int main(int argc, char* argv[])
{
    // Example of simple Version dump
    std::cout << "Git branch: " << GIT_BRANCH
        << "\nGit hash: " << GIT_COMMIT_HASH
        << "\nDate/Time: " << BUILD_DATE << "/" << BUILD_TIME
        << "\nTag: " << GIT_TAG << '\n';

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    try {
        objApp.run(".");
    }
    catch(std::exception & e) {
        std::cout << "Exception thrown" << std::endl;
        std::cout << e.what() << std::endl;
    }

    return 0;
}

