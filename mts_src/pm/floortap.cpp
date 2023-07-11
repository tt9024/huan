#include "floor.h"
#include "time_util.h"

const int QLen = 1024*1024*64; 
using QType = utils::MwQueue<QLen, utils::ShmCircularBuffer>;

void outputBuf(char*buf, int size) {
    fprintf(stderr, "\n");
    for (int i=0;i<size;++i) {
        if (buf[i] < 32 || buf[i] > 127)
            fprintf(stderr, " 0x%x ", buf[i]&0xff);
        else 
            fprintf(stderr, "%c", buf[i]);
    }
    fprintf(stderr, "\n");
}

void checkQ(QType::Reader* q, const char* name) {
    volatile char* buf;
    int bytes;
    utils::QStatus status = q->takeNextPtr(buf, bytes);
    if (status == utils::QStat_OK) {
        fprintf(stderr, "%s got %d", name, bytes);
        outputBuf((char*)buf, bytes);
        q->advance(bytes);
    } else {
        if (status != utils::QStat_EAGAIN) {
            fprintf(stderr, "%s got status %d\n", name, (int)status);
            q->syncPos();
        }
    }
};

int main() {
    QType qin("fq1", false, false);
    QType qout("fq2", false, false);
    auto qin_reader = new QType::Reader(qin);
    auto qout_reader = new QType::Reader(qout);

    fprintf(stderr, "fd1 read_pos: %lld, ready_pos: %lld\n", 
            qin_reader->getReadPos(), 
            qin_reader->getReadyPos());

    fprintf(stderr, "fd2 read_pos: %lld, ready_pos: %lld\n", 
            qout_reader->getReadPos(), 
            qout_reader->getReadyPos());

    while (true) {
        checkQ(qin_reader, "fd1");
        checkQ(qout_reader, "fd2");
        utils::TimeUtil::micro_sleep(10000);
    }
    return 0;
}

