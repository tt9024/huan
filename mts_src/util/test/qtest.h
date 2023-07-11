#pragma once

#include "queue.h"
#include "thread_utils.h"
#include "time_util.h"
#include <vector>

using namespace utils;

struct ItemType {
    uint64_t _seq;
    uint64_t _ts;
    uint64_t _id;
};

template<typename WriterType>
class QueueWriter {
public:
    QueueWriter(int id, WriterType& wrt, int wait_micro) :
        _id(id), _num(0), _writer(wrt), _wait_micro(wait_micro), _should_run(false) {};

    void run(void* data_len_ptr) {
        int len = *((int*)data_len_ptr);
        ItemType* buf = (ItemType*)malloc(len*sizeof(ItemType));
        _should_run = true;
        while (_should_run) {
            uint64_t now = TimeUtil::cur_micro();
            ++_num;
            for (int i=0; i<len; ++i) {
                buf[i]._seq = _num;
                buf[i]._ts = now;
                buf[i]._id = _id;
            }
            _writer.put((char*)buf);
            if ((_num & 0xffff) == 0 ) {
                printf("writer %d wrote %llu\n", _id, (unsigned long long)_num);
            }
            now += _wait_micro;
            while (TimeUtil::cur_micro() < now);
        }
        printf("writer %d stopped.", _id);
    }

    void stop() {
        _should_run = false;
    }

private:
    const int _id;
    uint64_t _num;
    WriterType& _writer;
    int _wait_micro;
    volatile bool _should_run;
};

template<typename WriterType>
class QueueWriterVarSize {
public:
    QueueWriterVarSize(int id, WriterType& wrt, int wait_micro) :
        _id(id), _num(0), _writer(wrt), _wait_micro(wait_micro), _should_run(false) {};

    void run(void* data_len_ptr) {
        int len = *((int*)data_len_ptr);
        ItemType* buf = (ItemType*)malloc(len*sizeof(ItemType));
        _should_run = true;

        fprintf(stderr, "starting dump: %s\n", _writer.dump().c_str());
        while (_should_run) {
            uint64_t now = TimeUtil::cur_micro();
            ++_num;
            for (int i=0; i<len; ++i) {
                buf[i]._seq = _num;
                buf[i]._ts = now;
                buf[i]._id = _id;
            }
            _writer.put((char*)buf, sizeof(ItemType)*(_num%len +1));
            if ((_num & 0xffff) == 0 ) {
                printf("writer %d wrote %llu\n", _id, (unsigned long long)_num);
            }
            now += _wait_micro;
            while (TimeUtil::cur_micro() < now);
        }
        printf("writer %d stopped.", _id);
    }

    void stop() {
        _should_run = false;
    }

private:
    const int _id;
    uint64_t _num;
    WriterType& _writer;
    int _wait_micro;
    volatile bool _should_run;
};

template<typename ReaderType>
class QueueReader {
public:
    QueueReader(int id, ReaderType& rdr) :
        _id(id), _reader(rdr), _should_run(false) {};

    struct WriterInfo {
        uint64_t seq;
        uint64_t lat;
        uint64_t cnt;
        uint64_t id;
        WriterInfo() {
            memset(this, 0, sizeof(WriterInfo));
        }
    };

    void run(void* data_len_ptr) {
        int len = *((int*)data_len_ptr);
        ItemType* buf = (ItemType*)malloc(len*sizeof(ItemType));
        _should_run = true;
        //uint64_t num = 0;
        while (_should_run) {
            utils::QStatus qstatus;
            qstatus = _reader.copyNextIn((char*)buf);

            if (qstatus == utils::QStat_OK) {
                if (!checkContent(buf, len)) {
                    printf("reader %d found error at %llu!\n", _id, (unsigned long long) (ItemType *)buf->_seq);
                } else {
                    //ItemType *iptr = buf + (++num%len);
                    ItemType *iptr = buf;
                    processWriterInfo(iptr);
                }
                _reader.advance();
            } else {
                if (qstatus != QStat_EAGAIN) {
                    printf("reader %d got qstatus %s.\n", _id, QStatStr(qstatus));
                    _reader.seekToTop();
                }
            }
        }
        printf("reader %d Stopped.\n", _id);
    }

    void stop() {

        _should_run = false;
    }

protected:
    const int _id;
    ReaderType& _reader;
    bool checkContent(ItemType* buf, int len) {
        for (int i=1; i<len; ++i) {
            if (memcmp(buf+i, buf, sizeof(ItemType))) {
                printf ("check failed at %d(%d) buf(%p): [%llu %llu %llu] vs [%llu %llu %llu]\n",
                        i, len, (char*)buf, (unsigned long long)buf->_id, (unsigned long long)buf->_seq,
                        (unsigned long long)buf->_ts, (unsigned long long)buf[i]._id,
                        (unsigned long long)buf[i]._seq, (unsigned long long)buf[i]._ts);
                return false;
            }
        }
        return true;
    }

    void processWriterInfo(ItemType* iptr) {

        WriterInfo& wr(_writerInfo[iptr->_id]);
        uint64_t& _lat (wr.lat);
        uint64_t& _cnt (wr.cnt);
        uint64_t& num (wr.seq);

        // keeping the stats for that content
        uint64_t new_num = iptr->_seq;
        if (num)
        {
            if (num != new_num-1) {
                printf("Failure! Reader %d found gap of %lld, (mine: %llu, writer%d: %llu), catching up\n",
                        _id,
                        (long long) (new_num - num - 1),
                        (unsigned long long) num,
                        (int) iptr->_id,
                        (unsigned long long) new_num);
            }
        } else {
            printf("reader %d started with %llu", _id, (unsigned long long) new_num);
        }
        num = new_num;
        uint64_t now = TimeUtil::cur_micro();
        _lat += (now - iptr->_ts);
        _cnt += 1;

        if ((_cnt & 0xffff) == 0) {
            printf("reader %d latency from writer %d: %llu (micro)\n", _id, (int)iptr->_id, (unsigned long long) _lat/_cnt);
            _lat = 0;
            _cnt = 0;
        }
    }

    bool _should_run;
    static const int MaxWriter = 16;
    WriterInfo _writerInfo[MaxWriter];
};


template<typename ReaderType>
class QueueReaderVarSize : public QueueReader<ReaderType> {
public:
    QueueReaderVarSize(int id, ReaderType& rdr) :
        QueueReader<ReaderType>(id, rdr) {};

    void run(void* data_len_ptr) {
        int len = *((int*)data_len_ptr);

#ifdef CopyBuffer
        ItemType* buf = (ItemType*)malloc(len*sizeof(ItemType));
        printf("copying ptr\n");
#else
        ItemType* buf;
        printf("taking ptr\n");
#endif
        this->_should_run = true;
        //uint64_t num = 0;

        fprintf(stderr, "starting dump: %s\n", this->_reader.dump_state().c_str());
        while (this->_should_run) {
            int bytes;
            utils::QStatus qstatus;

#ifdef CopyBuffer
            qstatus = this->_reader.copyNextIn((char*)buf, bytes);
#else
            qstatus = this->_reader.takeNextPtr((volatile char*&)buf, bytes);
#endif
            len = bytes/(int)sizeof(ItemType);

            if (qstatus == utils::QStat_OK) {
                if (!this->checkContent(buf, len)) {
                    printf("reader %d found error at %llu!\n", this->_id, (unsigned long long) (ItemType *)buf->_seq);
                } else {
                    //ItemType *iptr = buf + (++num%len);
                    ItemType *iptr = buf;
                    this->processWriterInfo(iptr);
                }
                this->_reader.advance(bytes);
            } else {
                if (qstatus != QStat_EAGAIN) {
                    printf("reader %d got qstatus %s.\n", this->_id, QStatStr(qstatus));
                    this->_reader.seekToTop();
                }
            }
        }
        printf("reader %d Stopped.\n", this->_id);
    }
};

#define ITEMLEN sizeof(ItemType)  /* seq + timestamp */
#define ITEMCNT 16
#define DATALEN (ITEMLEN*ITEMCNT) /* this represents the size of content, i.e. dob snapshot */
#define QLEN (64*1024*32*DATALEN)

