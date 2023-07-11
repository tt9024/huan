#include <md_snap.h>

#include <sstream>
#include <string>
#include <iostream>
#include <cctype>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

using namespace md;
using namespace utils;
using namespace std;

typedef BookQ<1> BookQType;
volatile bool user_stopped = false;

void sig_handler(int signo)
{
  if (signo == SIGINT) {
    printf("Received SIGINT, exiting...\n");
  }
  user_stopped = true;
}

int main(int argc, char**argv) {
    if (argc < 3) {
        printf("Usage: %s venue/symbol L1|L2 -o out_csv_file\n", argv[0]);
        printf("\n");
        return 0;
    }
    if (signal(SIGINT, sig_handler) == SIG_ERR)
    {
            printf("\ncan't catch SIGINT\n");
            return -1;
    }
    utils::PLCC::instance("booktap");
    BookConfig bcfg(argv[1], argv[2]);
    bool trade_only=false;
    bool dump_all = false;
    std::string out_csv;
    FILE* fp = nullptr;
    if (argc>3 && strcmp(argv[3], "-o")==0) {
        out_csv = argv[4];
        printf("dumping to csv file %s\n", out_csv.c_str());
        dump_all=true;
        fp = fopen(out_csv.c_str(), "at");
    }

    BookQType bq(bcfg, true);
    auto book_reader = bq.newReader();
    BookDepotLevel<1> myBook;

    //uint64_t start_tm = utils::TimeUtil::cur_time_micro();
    user_stopped = false;
    long long cnt = 0;
    while (!user_stopped) {
        if (dump_all) {
            if (book_reader->getNextUpdate(myBook)) {
                fprintf(fp, "%s\n", myBook.toCSV(1).c_str());
                if (++cnt % 10000 == 0) {
                    fflush(fp);
                }
            } else {
                usleep(1000);
            }
        } else {
            if (book_reader->getLatestUpdateAndAdvance(myBook))
            {
                if (!trade_only || myBook.update_type==2)
                printf("%s\n", myBook.prettyPrint().c_str());
            } else {
                usleep(100*1000);
            }
        }
    }
    if (fp) fclose(fp);
    printf("Done.\n");
    return 0;
}
