#include "md_bar.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Usage: %s tick_file barsec out_file tick_size\n",
                argv[0]);
        return 0;
    }

    int barsec = atoi(argv[2]);
    double tick_size = atof(argv[4]);
    //bool use_dir = atoi(argv[5]) == 1; // 0: don't use, 1: use

    auto bar = md::BarPrice();
    bar.set_tick_size(tick_size);
    bar.set_write_optional(true);
    FILE* fp=fopen(argv[3],"wb");

    const auto& lines(utils::CSVUtil::read_file(argv[1]));
    time_t due = (time_t) (atoll(lines[0][0].c_str())/1000000LL+barsec);
    md::BookDepotLevel<1> bk;
    for (const auto& line: lines) {
        // debug
        time_t cur_sec = (time_t) (atoll(line[0].c_str())/1000000LL);
        if (__builtin_expect(cur_sec >= due,0)) {
            while(true) {
                const auto bar_line = bar.writeAndRoll(due);
                fprintf(fp, "%s\n",bar_line.c_str());
                fflush(fp);
                due += barsec;
                if (__builtin_expect(cur_sec < due,1)) {
                    break;
                }
            }
        }

        bk.updateFrom(line);
        bar.update(bk);
#ifdef TD_PARSER_TRACE
        if (cur_sec == 1685714489LL) {
        FILE* fp=fopen("/tmp/tick_parser_trace.csv","at");
        fprintf(fp, "%s,%s\n",utils::CSVUtil::write_line_to_string(line).c_str(), bar.toString().c_str());
        fclose(fp);
        }
#endif
    }
    return 0;
}

