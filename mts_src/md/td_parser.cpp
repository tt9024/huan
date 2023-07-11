#include "td_parser.h"

int main(int argc, char** argv) {
    if (argc < 8) {
        printf("Usage: %s quote_file trade_file utc_start utc_end barsec \"bar|tick\" out_file [tick_size]\n",
                argv[0]);
        return 0;
    }

    std::string quote_file = argv[1];
    std::string trade_file = argv[2];
    int utc_start=atoi(argv[3]);
    int utc_end  =atoi(argv[4]);
    int barsec =  atoi(argv[5]);
    bool dump_bar = (strcmp(argv[6], "bar")==0);
    std::string out_file = argv[7];
    double tick_size = 0;
    if (argc>8) {
        tick_size = atof(argv[8]);
    };
    md::TickData2Bar tp(quote_file, trade_file, utc_start, utc_end, barsec, tick_size);
    if (dump_bar) {
        tp.parse(out_file);
    } else {
        tp.tickDump(out_file);
    }
    return 0;        
}

