#include "floor.h"
#include "time_util.h"

const int QLen = 1024*1024*64; 
using QType = utils::MwQueue<QLen, utils::ShmCircularBuffer>;

void pokeQ(QType::Writer* q, const char* name) {
    const  char* buf = "Hello!";
    int bytes = 7;
    utils::QPos pos = q->put(buf, bytes);
    fprintf(stderr, "%s wrote at pos %lld, dump: %s\n", name, pos, q->dump().c_str());
};

int main(int argc, char**argv) {
    if (argc!=2) {
        printf("Usage %s Y|N\nY: init the queue to zero, N: don't init to zero\n", argv[0]);
        return 0;
    }
    bool init_to_zero = false;
    if (argv[1][0] == 'Y') {
        init_to_zero = true;
    }

    QType qin("fq1", false, init_to_zero);
    QType qout("fq2", false, init_to_zero);
    auto qin_writer = new QType::Writer(qin);
    auto qout_writer = new QType::Writer(qout);

    fprintf(stderr, "fd1 dump: %s\n", qin_writer->dump().c_str());
    fprintf(stderr, "fd2 dump: %s\n", qout_writer->dump().c_str());


    while (true) {
        pokeQ(qin_writer, "qin");
        pokeQ(qout_writer, "qout");
        utils::TimeUtil::micro_sleep(1000000);
    }
    return 0;
}

