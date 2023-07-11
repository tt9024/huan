#include "qtest.h"

typedef SwQueue<QLEN, DATALEN, CircularBuffer> QType;

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s Num_Readers, writer_wait(micro)\n", argv[0]);
        return 0;
    }
    int num_readers = atoi(argv[1]);
    int wait_micro = atoi(argv[2]);

    QType queue;

    int itemCount = ITEMCNT;

    for (int i=0; i<num_readers; ++i)
    {
        QueueReader< QType::Reader >* reader =
                new QueueReader< QType::Reader >(i, *queue.newReader());
        ThreadWrapper< QueueReader< QType::Reader > >* readerThread =
                new ThreadWrapper< QueueReader< QType::Reader > >(*reader);

        readerThread->run(&itemCount);
    }

    QueueWriter< QType::Writer > writer (0, queue.theWriter(), wait_micro);
    ThreadWrapper< QueueWriter< QType::Writer > > writerThread(writer);
    writerThread.run(&itemCount);
    writerThread.join();

    return 0;
}
