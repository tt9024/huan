#include "md_bar.h"
#include <csignal>
#include <stdio.h>
#include <atomic>

#include "time_util.h"
#include "symbol_map.h"

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

int main() {
    std::signal(SIGINT, signal_handler); 
    std::signal(SIGTERM, signal_handler);

    using BookOutputType = md::BookQ<1>::Writer;
    using BookWriterType = md::BookWriter<BookOutputType, 1>;

    // pub to WTI_N1, SPX_N1
    const auto* ti (utils::SymbolMapReader::get().getByMtsSymbol("WTI_N1"));
    md::BookConfig bcfg(ti->_venue, ti->_tradable, "L1");
    md::BookQ<1> bq(bcfg, false);
    auto wti_out = bq.theWriter();
    BookWriterType bw_wti(wti_out, bcfg);

    const auto* tispx (utils::SymbolMapReader::get().getByMtsSymbol("SPX_N1"));
    md::BookConfig bcfgspx(tispx->_venue, tispx->_tradable, "L1");
    md::BookQ<1> bq_spx(bcfgspx, false);
    auto spx_out = bq_spx.theWriter();
    BookWriterType bw_spx(spx_out, bcfg);

    should_run = true;

    //sleep for 10 second before start
    //to test the empty book cases
    //utils::TimeUtil::micro_sleep(10*1000*1000);

    double midpx = 40.5;
    while (should_run) {
        // generate tick
        uint64_t cur_micro = utils::TimeUtil::cur_micro();
        bool is_bid = (cur_micro%2);
        double price = midpx + ((double)(cur_micro%3+1)/100.0) * (is_bid?-1:1);
        double spx_price = (double)(((int)(((int)(price*100.0 + 0.5))/4.0*100+0.5))/100.0+2000.0);
        midpx = price;
        if (midpx < 20) midpx = 20.0;
        if (midpx > 60) midpx = 60.0;

        double size = 1;

        bool is_trade = ((cur_micro%5) == 0);
        if (is_trade) {
            bw_wti.updTrade(price, size);
            bw_spx.updTrade(spx_price, size);
        } else {
            bw_wti.updBBO(price, size, is_bid, cur_micro);
            bw_spx.updBBO(spx_price, size, is_bid, cur_micro);
        }

        // wait for a random time
        uint64_t sleep_micro = cur_micro%100 + 1;
        utils::TimeUtil::micro_sleep( sleep_micro*sleep_micro*100 );
    }

    return 0;
}

