#include "qtest.h"

#define QItems (1024*64)

typedef SwQueue<QLEN, DATALEN, ShmCircularBuffer> SQType;
typedef MwQueue2<QLEN, DATALEN, ShmCircularBuffer> MQType;
typedef MwQueue<QLEN, ShmCircularBuffer> MWQType;

template<typename QType>
void runTest(QType& queue, bool is_read, int num_threads, int itemCount, int wait_micro, int writer_start_id) {
    if (is_read) {
        for (int i=0; i<num_threads; ++i)
        {
            QueueReader< typename QType::Reader >* reader =
                    new QueueReader< typename QType::Reader >(i, *queue.newReader());
            ThreadWrapper< QueueReader< typename QType::Reader > >* readerThread =
                    new ThreadWrapper< QueueReader< typename QType::Reader > >(*reader);

            readerThread->run(&itemCount);
        }
        while (1) {
            sleep (1000000);
        }
        return;
    }

    for (int i=0; i<num_threads; ++i)
    {
        QueueWriter< typename QType::Writer > *writer =
                new QueueWriter< typename QType::Writer >(writer_start_id+i, queue.theWriter(), wait_micro);
        ThreadWrapper< QueueWriter< typename QType::Writer > >* writerThread =
                new ThreadWrapper< QueueWriter< typename QType::Writer > >(*writer);
        writerThread->run(&itemCount);
    }
    while (1) {
        sleep (1000000);
    }
}

template<typename QType>
void runTestVarSize(QType& queue, bool is_read, int num_threads, int itemCount, int wait_micro, int writer_start_id) {
    if (is_read) {
        for (int i=0; i<num_threads; ++i)
        {
            QueueReaderVarSize< typename QType::Reader >* reader =
                    new QueueReaderVarSize< typename QType::Reader >(i, *queue.newReader());
            ThreadWrapper< QueueReaderVarSize< typename QType::Reader > >* readerThread =
                    new ThreadWrapper< QueueReaderVarSize< typename QType::Reader > >(*reader);

            readerThread->run(&itemCount);
        }
        while (1) {
            sleep (1000000);
        }
        return;
    }

    for (int i=0; i<num_threads; ++i)
    {
        QueueWriterVarSize< typename QType::Writer > *writer =
                new QueueWriterVarSize< typename QType::Writer >(writer_start_id+i, queue.theWriter(), wait_micro);
        ThreadWrapper< QueueWriterVarSize< typename QType::Writer > >* writerThread =
                new ThreadWrapper< QueueWriterVarSize< typename QType::Writer > >(*writer);
        writerThread->run(&itemCount);
    }
    while (1) {
        sleep (1000000);
    }

}

int main(int argc, char** argv) {
    if (argc < 7) {
        printf("Usage: %s read[0/1], num_threads writer_wait(micro), init_to_zero[0/1], qtype[s/m/w] writer_start_id\n", argv[0]);
        return 0;
    }

    bool is_read = (atoi(argv[1]) == 1);
    int num_readers = atoi(argv[2]);
    int wait_micro = atoi(argv[3]);
    bool init_to_zero = (atoi(argv[4]) == 1);
    char qtchar = argv[5][0];
    int writer_start_id = atoi(argv[6]);

    int itemCount = ITEMCNT;

    switch(qtchar) {
    case 's' : {
        SQType queue("qtest", is_read, init_to_zero);
        runTest(queue, is_read, num_readers, itemCount, wait_micro, writer_start_id);
        break;
    }
    case 'm' : {
        MQType queue("qtestm", is_read, init_to_zero);
        runTest(queue, is_read, num_readers, itemCount, wait_micro, writer_start_id);
        break;
    }
    case 'w' : {
        MWQType queue("qtestw", is_read, init_to_zero);
        runTestVarSize(queue, is_read, num_readers, itemCount, wait_micro, writer_start_id);
        break;
    }
    }
    return 0;
}
