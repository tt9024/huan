#include <csignal>
#include <stdio.h>
#include <atomic>

#include "md_bar.h"
#include "time_util.h"

namespace
{
    volatile bool should_run;
}

void signal_handler(int signal)
{
    should_run = false;
    logInfo("Signal %d received, stopping bar read", signal);
}

void printBar(const md::BarPrice& bp, int barsec) {
    printf("Bar %d update@ %s - %s\n", barsec, 
            utils::TimeUtil::frac_UTC_to_string(0, 6).c_str(),
            bp.toCSVLine().c_str());
}

int main(int argc, char**argv ) {

    if (argc != 4) {
        printf("Usage: %s venue symbol L1/L2\nExample: %s CME CLF1 L1\n", argv[0], argv[0]);
        return 0;
    }
    std::signal(SIGINT, signal_handler); 
    std::signal(SIGTERM, signal_handler);

    md::BookConfig bcfg(argv[1], argv[2], argv[3]);

    // create reader for all bar period defined
    auto bsv = bcfg.barsec_vec();
    std::vector<md::BarReader> br;
    for (const auto& bs : bsv) {
        br.emplace_back(bcfg, bs);
    }

    md::BarPrice bp;
    should_run = true;
    while (should_run) {

        for (auto& b : br) {
            if (b.read(bp)) {
                printBar(bp, b.barsec);
            }
        }
        utils::TimeUtil::micro_sleep(5000);
    }

    return 0;
}

