#pragma once
#include <memory>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

//  compound assignment with ‘volatile’-qualified left operand is deprecated [-Werror=volatile]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvolatile"

#include "circular_buffer.h"

/** This is for multi-threaded environment.
 * Goal is to achieve multi-write, multi-read
 * lock-free read/write. Each put/take is
 * also wait-free except in unlikely situation
 * where writer can spin on bursty concurrent
 * write operations.
 */

namespace utils {

    // This is a fixed size, lossy single writer multiple reader queue.
    // The writer does not check the overflow on readers, and it always
    // writes to queue without blocking. Overflow can be detected at
    // the reader side and can be somewhat avoided by increasing buffer size.
    // The reader has to check queue status on each read.
    template<int QLen, int DataLen, template<int, int> class BufferType = CircularBuffer>
    class SwQueue
    {
    public:
        class Writer;
        class Reader;
        // this is for multi-threaded environment using heap
        SwQueue() : m_name(""), m_writer(new Writer(*this)) {};

        // this is for multi-process environment using shm
        explicit SwQueue(const char* shm_name, bool read_only = true, bool init_to_zero = true) :
                m_name(shm_name), m_buffer(shm_name, read_only, init_to_zero),
                m_writer(read_only?NULL:new Writer(*this, init_to_zero)) {};

        ~SwQueue() {
            if (m_writer) {
                delete m_writer;
                m_writer = NULL;
            };
        };

        // the only writer
        Writer& theWriter() {
            if (!m_writer) {
                throw std::runtime_error("SwQueue writer instance NULL!");
            }
            return *m_writer;
        };

        // caller is responsible to delete this object
        Reader* newReader() {
            return new Reader(*this);
        }

    private:
        const std::string m_name;
        static const int HeaderLen = 64;   // one 64-bit counter for next writer position
        BufferType<QLen, HeaderLen> m_buffer;
        Writer* m_writer;
        // in case the compiler uses c++03
        friend class Reader;
        friend class Writer;
        volatile QPos *getPtrReadyBytes() const { return (volatile QPos*) m_buffer.getHeaderStart(); } ;

    public:
        // only one writer will have shared access to the queue
        class Writer {
        public:
            // just put the content to the queue, slow readers
            // could get overflow.  C++ ensures the order of operation on
            // two volatile pointers are honored.
            // returns the writer position before the put
            QPos putNoSpin(const char* content) {
                const QPos ready_bytes = *m_ready_bytes;
                m_buffer->template copyBytesNoCross<true>(ready_bytes, content, DataLen);
                asm volatile("" ::: "memory");
                *m_ready_bytes += DataLen;
                return ready_bytes;
            }

            // for interface with the spin writer
            // return the writer position before the put
            QPos put(const char* content) {
                return putNoSpin(content);
            }

            // for in context writers. It's for performance
            // reasons: the QLen has to be multiple of Datalen
            // you need to getNextWritePtr(), update buffer
            // and when done, advanceWritePtr(), in sequence
            volatile char* getNextWritePtr() const {
                return m_buffer->getBufferPtr(*m_ready_bytes);
            }

            volatile char* advanceWritePtr() volatile {
                return (*m_ready_bytes += DataLen);
            }

            static const int data_len = DataLen;
            static const int q_len = QLen;

            inline QPos getWritePos() const {
                return *m_ready_bytes;
            }

        private:
            explicit Writer(SwQueue<QLen, DataLen, BufferType>& queue, bool init_to_zero=true)
            : m_buffer(&queue.m_buffer),
              m_ready_bytes(queue.getPtrReadyBytes())
            {
                if (init_to_zero)
                    *m_ready_bytes = 0;

                // the head format is 
                // ready_bytes(8), item_size(8), total_items(8)
                m_ready_bytes[1] = DataLen;
                m_ready_bytes[2] = QLen / DataLen; //total items
            };

            explicit Writer(const Writer& writer);
            volatile BufferType<QLen, SwQueue<QLen, DataLen, BufferType>::HeaderLen>* const m_buffer;
            volatile QPos* const m_ready_bytes;
            // for private constructor access
            friend class SwQueue<QLen, DataLen, BufferType>;
        };


        // Readers share access to the queue,
        // has it's own read position
        class Reader {
        public:
            QStatus copyNextIn(char* buffer) {
                QPos pos = *m_ready_bytes;
                long long bytes = (long long) (pos - m_pos);

                if (__builtin_expect((bytes == 0),0)) {
                    return QStat_EAGAIN;
                }
                while (__builtin_expect((bytes < 0),0)) {
                    // queue was restarted with a lower position
                    // set it to bottom and start there
                    seekToBottom();
                    pos = *m_ready_bytes;
                    bytes = (long long) (pos - m_pos);
                }
                if (__builtin_expect((bytes > QLen - DataLen), 0)) {
                    return QStat_OVERFLOW;
                }
                if (__builtin_expect((bytes < 0), 0)) {
                    // writer restart detected
                    seekToTop();
                    return copyNextIn(buffer);
                }
                m_buffer->template copyBytesNoCross<false>(m_pos, buffer, DataLen);
                // check overflow after read
                if (__builtin_expect(((*m_ready_bytes - m_pos) > QLen - DataLen), 0)) {
                    return QStat_OVERFLOW;
                }
                return QStat_OK;
            }

            QStatus copyTopIn(char* buffer) {
                QPos pos = *m_ready_bytes;
                if (__builtin_expect( (pos==0),0)) {
                    return QStat_EAGAIN;
                }

                pos -= DataLen;
                m_buffer->template copyBytesNoCross<false>(pos, buffer, DataLen);
                // check overflow after read
                if (__builtin_expect(((*m_ready_bytes-pos)>QLen-DataLen),0)){
                    // try one more time and giev up
                    pos = *m_ready_bytes - DataLen;
                    m_buffer->template copyBytesNoCross<false>(pos, buffer, DataLen);
                    if (((*m_ready_bytes-pos)>QLen-DataLen),0){ 
                        return QStat_OVERFLOW;
                    }
                }
                return QStat_OK;
            }

            QStatus copyPosInRandomAccess(char* buffer, QPos pos) {
                // the return value from put() could be stored 
                // in associate with the data. 
                // Usage scenario: out-of-order message stoers
                if (__builtin_expect((long long)pos < 0, 0)) {
                    return QStat_ERROR;
                }
                QPos qpos = *m_ready_bytes;
                if (__builtin_expect(qpos <= pos,0)) {
                    return QStat_EAGAIN;
                }
                if (__builtin_expect((qpos - pos >= QLen), 0)) {
                    return QStat_OVERFLOW;
                }
                m_buffer->template CopyBytesFromBufferNoCross(pos, buffer, DataLen);

                // check after copy
                if (__builtin_expect((*m_ready_bytes - pos >= QLen), 0)) {
                    return QStat_OVERFLOW;
                }
                return QStat_OK;
            }

            // this doesn't copy the bytes
            // returns the position for next read
            QStatus takeNextPtr(volatile char*& buffer, QPos& pos) const {
                long long bytes = (long long)(*m_ready_bytes - m_pos);
                if (bytes <= 0) {
                    return QStat_EAGAIN;
                }
                if (__builtin_expect((bytes > QLen - DataLen), 0)) {
                    return QStat_OVERFLOW;
                }
                buffer = m_buffer->getBufferPtr(m_pos);
                pos = m_pos + DataLen;
                return QStat_OK;
            }

            // takes the pointer to top item, set pos_top to be after the item
            QStatus takeTopPtr(volatile char*& buffer, QPos& pos_top) const {
                QPos pos = *m_ready_bytes;
                if (__builtin_expect((pos == 0), 0)) {
                    return QStat_EAGAIN;
                }
                pos_top = pos;
                buffer = m_buffer->getBufferPtr(pos_top - DataLen);
                return QStat_OK;
            }

            bool verifyPosValid(const QPos pos) const {
                long long bytes = (long long) (*m_ready_bytes - pos);
                if (__builtin_expect((bytes > QLen - DataLen) || (bytes<0), 0)) {
                    return false;
                }
                return true;
            }

            bool verifyPosTop(const QPos pos) const {
                return (*m_ready_bytes == pos) ;
            }

            // set the next read to be at
            // the latest item in the queue
            void seekToTop() {
                QPos pos = *m_ready_bytes;
                if (__builtin_expect((pos == 0), 0)) {
                    m_pos = 0;
                    return ;
                }
                m_pos = pos - DataLen;
            }

            // set the next read to be at a valid
            // position, noop if it is already.
            // Used for catch up from overflow.
            // Returns number of items lost
            int catchUp() {
                QPos pos = *m_ready_bytes;
                QPos prev_pos = m_pos;
                while (pos - m_pos > QLen - DataLen) {
                    m_pos += DataLen;
                }
                return (m_pos - prev_pos)/DataLen;
            }

            // Same as seekToTop, except when
            // no updates since last read, 
            // seekToTop() reduces m_pos to allow next read
            // of the latest item.  advanceToTop() 
            // doesn't reduce the m_pos, so next read will
            // return false, useful if caller doesn't want 
            // to read same latest item multiple times, such
            // as bootap
            bool advanceToTop() {
            	QPos pos = m_pos;
            	seekToTop();
                QPos newPos = m_pos + DataLen;

                if (__builtin_expect(pos > newPos, 0)) {
                    // queue restarted since last read, reset
                    // m_pos to reflect the new top read pos
                    pos = m_pos;
                }
                if (pos > m_pos) {
                    // no update case
                    m_pos = pos;
                    return false;
                }
                if (__builtin_expect(*m_ready_bytes == 0, 0)) {
                    // empty queue
                    m_pos = 0;
                    return false;
                };
                return true;
            }

            void seekToBottom() {
                QPos pos = *m_ready_bytes;
                m_pos = (pos > (QLen - DataLen))? (pos - QLen - DataLen) : 0;
            }

            void advance() { m_pos += DataLen; };
            void syncPos() { m_pos = *m_ready_bytes;};
            QPos getPos() const { return m_pos; };
            QPos getWritePos() const { return *m_ready_bytes;}
            //void setPos(QPos pos) { m_pos = pos; };

        private:
            explicit Reader(SwQueue<QLen, DataLen, BufferType>& queue)
            : m_buffer(&queue.m_buffer),
              m_ready_bytes(queue.getPtrReadyBytes()),
              m_pos(*m_ready_bytes)
            {
                //seekToBottom();
                //seekToTop();
                // initialize m_pos with the existing ready bytes and
                // go from there
                syncPos();
            }
            Reader(const Reader&);
            volatile BufferType<QLen, SwQueue<QLen, DataLen, BufferType>::HeaderLen>* const m_buffer;
            const volatile QPos* const m_ready_bytes;
            QPos m_pos;
            // for private constructor access
            friend class SwQueue<QLen, DataLen, BufferType>;
        };
    };

    // FIXME  ---
    //     ADD READER POSITION TO THE CIRCULAR QUEUE
    //     TO SUPPORT MULTI-PROCESS SHARED MEMROY
    // ------
    //
    // This is a fixed size, lossless single writer multiple reader queue.
    // The writer keeps track of all readers and spin on any overflow.
    // So readers don't need to check for overflow.
    // Both QLen and DataLen has to be power of 2.
    template<int QLen, int DataLen>
    class SwQueueLossless
    {
    public:
        class Writer;
        class Reader;
        SwQueueLossless() : m_writer(*this), m_numReaders(0) {};
        ~SwQueueLossless() {};
        Reader* newReader() {
            // caller is responsible to delete this object
            if (m_numReaders >= MaxReaders)
                throw std::runtime_error("too many Reader created!");
            if (!(m_readers[m_numReaders++] = new Reader(*this)))
                throw std::runtime_error("Reader failed to create!");
            return m_readers[m_numReaders-1];
        }
        // the only writer
        Writer& theWriter() {
            return m_writer;
        };

        Writer* newWriter() {
            return &m_writer;
        }

    private:
        volatile QPos *getPtrReadyBytes() const { return (volatile QPos*) m_buffer.getHeaderStart(); } ;
        static const int HeaderLen = 64;   // one 64-bit counter for next writer position
        static const int MaxReaders = 16;
        CircularBuffer<QLen, HeaderLen> m_buffer;
        Writer m_writer;
        volatile Reader* m_readers[MaxReaders];
        int m_numReaders;
        // in case the compiler uses c++03
        friend class Reader;
        friend class Writer;

    public:
        // only one writer will have shared access to the queue
        class Writer {
        public:
            void putSpin(const char* content) {
                QPos pos = *m_ready_bytes;
                // spin to wait for the slow reader
                for (int i=0; i<m_queue.m_numReaders; ++i) {
                    while (__builtin_expect((pos - m_readers[i]->getPos() > QLen - DataLen), 0)) {}
                }
                m_buffer->template copyBytes<true>(*m_ready_bytes, content, DataLen);
                asm volatile("" ::: "memory");
                *m_ready_bytes += DataLen;
            }

            void put(const char* content) {
                putSpin(content);
            }

        private:
            explicit Writer(SwQueueLossless<QLen, DataLen>& queue)
            : m_queue(queue),
              m_buffer(&queue.m_buffer),
              m_ready_bytes(queue.getPtrReadyBytes())
            {
                *m_ready_bytes = 0;
            };
            explicit Writer(const Writer& writer);
            SwQueueLossless<QLen, DataLen>& m_queue;
            volatile CircularBuffer<QLen, SwQueueLossless<QLen, DataLen>::HeaderLen>* const m_buffer;
            volatile QPos* const m_ready_bytes;
            // for private constructor access
            friend class SwQueueLossless<QLen, DataLen>;
        };

        // Readers share access to the queue,
        // has it's own read position, made visible to the writer
        // No overflow check is performed as the writer will wait
        // on slowest reader for queue full
        class Reader {
        public:
            QStatus copyNextIn(char* buffer) {
                QPos wPos = *m_ready_bytes;
                QPos rPos = m_pos;
                if (wPos <= rPos) {
                    return QStat_EAGAIN;
                }
                m_buffer->template copyBytes<false>(rPos, buffer, DataLen);
                return QStat_OK;
            }

            // this doesn't copy the bytes
            QStatus takeNextPtr(char*& buffer) {
                QPos wPos = *m_ready_bytes;
                QPos rPos = m_pos;
                if (wPos <= rPos) {
                    return QStat_EAGAIN;
                }
                if (__builtin_expect(m_buffer->wouldCrossBoundary(rPos, DataLen, buffer), 0))
                {
                    // we cannot just return the pointer to the content since the
                    // content would cross circular boundary.
                    // Copy the bytes locally and return;
                    m_buffer->template copyBytes<false>(rPos, m_localBuffer, DataLen);
                    buffer = m_localBuffer;
                }
                return QStat_OK;
            }

            void seekToTop() {
                QPos pos = *m_ready_bytes;
                if (__builtin_expect((pos == 0), 0))
                    m_pos = 0;
                m_pos = pos - DataLen;
            }

            void seekToBottom() {
                QPos pos = *m_ready_bytes;
                QPos pos_lowest = (pos > QLen)? (pos - QLen) : 0;
                int ready_bytes = pos - pos_lowest;
                while (ready_bytes >= DataLen) {
                    ready_bytes -= DataLen;
                }
                m_pos = pos_lowest + ready_bytes;
            }

            void advance() { m_pos += DataLen; };
            QPos getPos() const volatile { return m_pos; };
            QPos getPosVolatile() const volatile { return m_pos; };
            void setPos(QPos pos) { m_pos = pos; };

        private:
            explicit Reader(SwQueueLossless<QLen, DataLen>& queue)
            : m_buffer(&queue.m_buffer),
              m_ready_bytes(queue.getPtrReadyBytes),
              m_pos(0) { seekToBottom(); }
            Reader(const Reader&);
            volatile CircularBuffer<QLen, SwQueueLossless<QLen, DataLen>::HeaderLen>* const m_buffer;
            const volatile QPos* const m_ready_bytes;
            volatile QPos m_pos;
            char m_localBuffer [DataLen];
            // for private constructor access
            friend class SwQueueLossless<QLen, DataLen>;
        };
    };

    // the multi-writer queue, data can be uneuqal length
    // the three counters, write_pos, sync_pos and dirty_count
    // are stored in circular buffer as well
    // the MaxBurstSize is the maximum number of bytes that
    // can be dirty but not sync'ed to be readable
    template<int QLen, template<int, int> class BufferType = CircularBuffer>
    class MwQueue
    {
    public:
        class Reader;
        class Writer;

        // this is for multi-threaded environment using heap
        explicit MwQueue(const char* queue_name = "ThreadMwQueue") : m_name(queue_name) {
        };

        // this is for multi-process environment using shm
        // be careful about the init to zero option. If set,
        // all existing contents are lost.  Usually we should
        // leave it to be false here.
        MwQueue(const char* shm_name, bool read_only, bool init_to_zero = false) :
                m_name(shm_name), m_buffer(shm_name, read_only, init_to_zero) {
            if (!init_to_zero) {
                //auto ptr_write_pos = getPtrPosWrite();
                //auto ptr_dirty_pos = getPtrPosDirty();
                //auto ptr_ready_pos = getPtrPosReady();
                if (!read_only) {
                    // resetState();
                    //*ptr_dirty_pos = *ptr_write_pos;
                    //*ptr_ready_pos = *ptr_write_pos;
                }
            }
        };

        ~MwQueue() {
            for (size_t i = 0; i<_writersToDelete.size(); ++i) {
                delete (Writer*) (_writersToDelete[i]);
            }
            _writersToDelete.clear();
        };
        Reader* newReader() {
            // the caller responsible for deleting the instance
            return new Reader(*this);
        }
        Writer* newWriter() {
            // the caller responsible for deleting the instance
            return new Writer(*this);
        }

        Writer& theWriter() {
            Writer* wtr = new Writer(*this);
            _writersToDelete.push_back(wtr);
            return *wtr;
        }

        std::vector<Writer*> _writersToDelete;
        static const int HeaderLen = 128;   // three 64-bit counters, occupying 2 cache lines
                                            // one for writers - write + dirty, one for read
                                            // also aligning to the cache line

        volatile QPos *getPtrPosWrite() const { return (volatile QPos*) m_buffer.getHeaderStart() ; };
        volatile QPos *getPtrPosDirty() const { return (volatile QPos*) (m_buffer.getHeaderStart() + sizeof(QPos)) ; };
        volatile QPos *getPtrPosReady() const { return (volatile QPos*) (m_buffer.getHeaderStart() + 64); } ;
        volatile QPos *getPtrPosReadyPrev() const { return (volatile QPos*) (m_buffer.getHeaderStart() + 64 + sizeof(QPos)) ; } ;

    private:
        const std::string m_name;
        BufferType<QLen, HeaderLen> m_buffer;
        using QType = MwQueue<QLen, BufferType>;


    public:
        // each reader will have have shared access to the queue,
        // has it's own read position
        class Reader {
        public:
            explicit Reader(QType& queue)
            : m_buffer(queue.m_buffer),
              m_pos_write(queue.getPtrPosWrite()),
              m_ready_bytes(queue.getPtrPosReady()),
              m_ready_bytes_prev(queue.getPtrPosReadyPrev()),
              m_pos(0),
              m_localBuffer(NULL),
              m_localBufferSize(0) {
                  syncPos();
              };

            QStatus copyNextIn(char* buffer, int& bytes);

            // this doesn't copy the bytes in most cases, except the content
            // go across the circular buffer boundary
            QStatus takeNextPtr(volatile char*& buffer, int& bytes);

            // following 2 reads the latest update in the queue,
            // if the queue is not empty, using m_ready_bytes_prev
   	    QStatus copyLatestIn(char*buffer, int& bytes);
            QStatus takeLatestPtr(volatile char*& buffer, int& bytes);

            // this reads from a specific location without checking ready_bytes
   	    QStatus copyPosInRandomAccess(char*buffer, int& bytes, QPos pos);

            QStatus seekToTop();
            void advance(int bytes) { m_pos += (bytes+sizeof(int));};
            void syncPos() {m_pos = *m_ready_bytes;};
            std::string dump_state() const;
            QPos getReadPos() const { return m_pos; };
            QPos getReadyPos() const { return *m_ready_bytes; };

            ~Reader() {
                if (m_localBuffer) {
                    free(m_localBuffer); 
                    m_localBuffer = NULL ;
                };
            };
        private:
            const BufferType<QLen, QType::HeaderLen>& m_buffer;
            const volatile QPos* const m_pos_write; // readonly volatile pointer to a ever changing memory location
            const volatile QPos* const m_ready_bytes;
            const volatile QPos* const m_ready_bytes_prev;
            QPos m_pos;
            mutable char* m_localBuffer;
            mutable int  m_localBufferSize;
        };

        // each writer will have shared access to the queue
        class Writer {
        public:
            explicit Writer(QType & queue)
            : m_buffer(queue.m_buffer),
              m_pos_write(queue.getPtrPosWrite()),
              m_pos_dirty(queue.getPtrPosDirty()),
              m_ready_bytes(queue.getPtrPosReady()),
              m_ready_bytes_prev(queue.getPtrPosReadyPrev()) {
                //reset();
            };

            // be very careful calling this
            //void reset() { *m_pos_write = *m_ready_bytes; *m_pos_dirty = 0};
            QPos put(const char* content, int bytes);
            QPos put(const char* content1, int bytes1, const char* content2, int bytes2); // as one message
            std::string dump() const {
            	char buf[128];
            	snprintf(buf, sizeof(buf), "Writer - %lld, %lld, %lld",
            			(long long) *m_pos_write,
						(long long) *m_pos_dirty,
						(long long) *m_ready_bytes);
            	return std::string(buf);
            }
            QPos writePos() const { return *m_pos_write;};
        private:
            BufferType<QLen, MwQueue::HeaderLen>& m_buffer;
            volatile QPos* const m_pos_write;
            volatile QPos* const m_pos_dirty;
            volatile QPos* const m_ready_bytes;
            volatile QPos* const m_ready_bytes_prev;
            QPos getWritePos(int bytes);
            void finalizeWrite(int bytes);
            static const QPos SpinThreshold = QLen/2;
        };
    };

    // it first get the write pos, write a size, write the content
    // and check if it detects a synchronous point and update read
    template<int QLen, template<int, int> class BufferType>
    inline
    QPos MwQueue<QLen, BufferType>::Writer::put(const char* content, int bytes) {
        int total_bytes = bytes + sizeof(int);
        QPos pos = getWritePos(total_bytes);
        m_buffer.template copyBytes<true>(pos, (char*) &bytes, sizeof(int));
        m_buffer.template copyBytes<true>(pos + sizeof(int), (char*) content, bytes);
        asm volatile("" ::: "memory");
        finalizeWrite(total_bytes);
        return pos;
    }

    // it first get the write pos, write a size, write the content
    // and check if it detects a synchronous point and update read
    template<int QLen, template<int, int> class BufferType>
    inline
    QPos MwQueue<QLen, BufferType>::Writer::put(const char* content1, int bytes1, const char* content2, int bytes2) {
        int total_bytes = bytes1 + bytes2 + sizeof(int);
        QPos pos = getWritePos(total_bytes);
        total_bytes-=sizeof(int);
        m_buffer.template copyBytes<true>(pos,                    (char*) &total_bytes, sizeof(int));
        m_buffer.template copyBytes<true>(pos+sizeof(int),        (char*) content1,     bytes1);
        m_buffer.template copyBytes<true>(pos+sizeof(int)+bytes1, (char*) content2,     bytes2);
        asm volatile("" ::: "memory");
        finalizeWrite(total_bytes+sizeof(int));
        return pos;
    }

    template<int QLen, template<int, int> class BufferType>
    inline
    QPos MwQueue<QLen, BufferType>::Writer::getWritePos(int bytes) {
        QPos pos;
        // get a time count and throw if wait for too long


        /*
        {
            // debug
            QPos posw = *m_pos_write;
            QPos posr = *m_ready_bytes;
            if (posw>posr) {
                printf("%lld-%lld\n",posw, posw-posr);
            }
        }
        */

        while (1) {
            pos = *m_pos_write;
            if (__builtin_expect(((QPos)(pos - *m_ready_bytes) < SpinThreshold), 1)) {
                QPos newPos = pos + bytes;
                if (compareAndSwap(m_pos_write, pos, newPos) == pos) {
                    return pos;
                }
            }
            // TODO - queue state validation (low priority)
            // there is a rare possiblity that some writers got killed 
            // during the write. This would make everyone after it spinning.
            // To Fix it, demand the write to finish pushing the dirty. 
            // Otherwise, it just spins.  A spin state is needed  
            // when spining more than a threshold, say 1 second, in which case writer
            // try to set the "spin state", with CAS. If it gets it
            // then it waits for a short while, say 100 milli, 
            // (for trainsient writes before spin set, in which case it will spin)
            // reads the current write pos, and then then writes the
            // total size and zero everything in dirty. It then writes
            // its own, finalize, and clear spining, and then return.
            // Additional safety check, when finalize, check if the ready bytes
            // already more than ready bytes.  This for transient writes (see above)
            // 
            // This safety check slows the write a little
            // but make it safer.
            // Consider the probability of gone bad during the two memcopy,
            // it is therefore a low priority fix.
        }
    }

    template<int QLen, template<int, int> class BufferType>
    inline
    void MwQueue<QLen, BufferType>::Writer::finalizeWrite(int bytes) {
        QPos dirty = AddAndfetch(m_pos_dirty, bytes);
        if (__builtin_expect((dirty == *m_pos_write), 1)) {
            // detected a sync point
            // ready bytes are at least dirty bytes
            // try to update the read position
            QPos prev_ready_bytes = *m_ready_bytes;
            const QPos prev_ready_bytes_save = prev_ready_bytes;
            while (prev_ready_bytes < dirty) {
                // multiple writers may be updating ready_bytes
                // take the hight dirty value
                prev_ready_bytes = compareAndSwap(m_ready_bytes, prev_ready_bytes, dirty);
            }

            // update the previous ready bytes
            // note this might not be the latest, but is a valid
            // previous position, reader could use it to seek
            if (__builtin_expect(prev_ready_bytes_save < prev_ready_bytes,1)) {
                *m_ready_bytes_prev = prev_ready_bytes_save;
            }
        }
    }

    template<int QLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue<QLen, BufferType>::Reader::takeNextPtr(volatile char*& buffer, int& bytes) {
        bytes = 0;
        if (__builtin_expect((m_pos > *m_pos_write),0)) {
            // queue restarted, return failure
            fprintf(stderr, "takeNextPtr error: %s\n",dump_state().c_str());
            syncPos();
            return QStat_ERROR;
        }
        if (__builtin_expect((*m_pos_write - m_pos >= QLen), 0)) {
            return QStat_OVERFLOW;
        }

        QPos unread_bytes = *m_ready_bytes - m_pos;
        if (unread_bytes == 0)
            return QStat_EAGAIN;

        // get size
        m_buffer.template copyBytes<false>(m_pos, (char*)&bytes, sizeof(int));
        QPos data_pos = m_pos + sizeof(int);
        if (__builtin_expect(m_buffer.wouldCrossBoundary(data_pos, bytes, buffer), 0))
        {
            // we cannot just return the pointer to the content since the
            // content would cross circular boundary.
            // Copy the bytes locally and return;
            //
            if (__builtin_expect((!m_localBuffer || (m_localBufferSize<bytes)), 0)) {
                if (m_localBuffer) {
                    free( m_localBuffer );
                }
                m_localBuffer = (char*) malloc(bytes);
                m_localBufferSize = bytes;
            }
            //printf("bytes = %d, local_size = %d, buffer = %p\n", (int) bytes, m_localBufferSize, m_localBuffer);

            m_buffer.template copyBytes<false>(data_pos, m_localBuffer, bytes);
            buffer = m_localBuffer;
        }
        return QStat_OK;
    };

    template<int QLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue<QLen, BufferType>::Reader::copyNextIn(char* buffer, int& bytes) {
        bytes = 0;
        if (__builtin_expect((m_pos > *m_pos_write),0)) {
            // queue restarted, return failure
            syncPos(); 
            return QStat_ERROR;
        }
        if (__builtin_expect((*m_pos_write - m_pos >= QLen-sizeof(int)), 0)) {
            return QStat_OVERFLOW;
        }
        QPos unread_bytes = *m_ready_bytes - m_pos;
        if (__builtin_expect((unread_bytes < sizeof(int)), 0)) {
            return QStat_EAGAIN;
        };
        // get size
        m_buffer.template copyBytes<false>(m_pos, (char*)&bytes, sizeof(int));
        // this shouldn't happen as we update the update as whole
        if (__builtin_expect(unread_bytes - sizeof(int) < bytes,0)) {
            syncPos();
            return QStat_ERROR;
        }
        m_buffer.template copyBytes<false>(m_pos + sizeof(int), buffer, bytes);
        if (__builtin_expect((*m_pos_write - m_pos >= QLen-(bytes+sizeof(int))), 0)) {
            return QStat_OVERFLOW;
        }
        return QStat_OK;
    }

    template<int QLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue<QLen, BufferType>::Reader::copyLatestIn(char*buffer, int& bytes) {
        const auto prev_m_pos = m_pos;
        m_pos = *m_ready_bytes_prev;
        QStatus qs = seekToTop();
        if (__builtin_expect(qs == QStat_OK, 1)) {
            qs = copyNextIn(buffer, bytes);
        }
        m_pos = prev_m_pos;
        return qs;
    }

    template<int QLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue<QLen, BufferType>::Reader::takeLatestPtr(volatile char*& buffer, int& bytes) {
        const auto prev_m_pos = m_pos;
        m_pos = *m_ready_bytes_prev;
        QStatus qs = seekToTop();
        if (__builtin_expect(qs == QStat_OK, 1)) {
            qs = takeNextPtr(buffer, bytes);
        }
        m_pos = prev_m_pos;
        return qs;
    }

    template<int QLen, template<int, int> class BufferType>
    inline
    std::string MwQueue<QLen, BufferType>::Reader::dump_state() const {
    	char buf[128];
    	snprintf(buf, sizeof(buf), "MwQueue Reader %lld %lld %lld",
    			(long long) *m_pos_write,
				(long long) *m_ready_bytes,
				(long long) m_pos);
    	return std::string(buf);
    }

    template<int QLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue<QLen,BufferType>::Reader::copyPosInRandomAccess(char*buffer, int& bytes, QPos pos) {
        bytes = 0;
        if (__builtin_expect((pos > *m_pos_write),0)) {
            // queue restarted, return failure
            return QStat_ERROR;
        }
    	if (__builtin_expect((*m_pos_write - pos >=QLen-sizeof(int)),0)) {
    		return QStat_OVERFLOW;
    	}
        m_buffer.template copyBytes<false>(pos, (char*)&bytes, sizeof(int));
        m_buffer.template copyBytes<false>(pos + sizeof(int), buffer, bytes);
        if (__builtin_expect((*m_pos_write - pos >= QLen-(bytes+sizeof(int))), 0)) {
            return QStat_OVERFLOW;
        }
    	return QStat_OK;
    }

    // it traverse all the way to top, and get the position of top update
    // to be ready for the next read. 
    // The next read could return EAGAIN if nothing is unread, (EAGAIN)
    // or queue was restarted (ERROR)
    // or too many unread, (OVERFLOW)
    // For all above three cases, read position is set to the end, 
    // as if syncPos()
    template<int QLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue<QLen, BufferType>::Reader::seekToTop() {
        if (__builtin_expect((m_pos > *m_pos_write), 0)) {
            // the queue restart case, this could be solved 
            // by syncPos() to go forward
            syncPos();
            return QStat_ERROR;
        }
        if (__builtin_expect(*m_ready_bytes == 0, 0)) {
            m_pos = 0;
            return QStat_EAGAIN;
        }
        if (__builtin_expect((*m_pos_write - m_pos >= QLen), 0)) {
            syncPos();
            return QStat_OVERFLOW;
        }
        long long unread_bytes = *m_ready_bytes - m_pos;
        if (unread_bytes == 0)
            return QStat_EAGAIN;

        if (__builtin_expect((unread_bytes < QLen), 1)) {
            QPos pos = m_pos;
            int bytes;
            while (unread_bytes > 0) {
                m_pos = pos;
                m_buffer.template copyBytes<false>(pos, (char*)&bytes, sizeof(int));
                pos += (sizeof(int) + bytes);
                unread_bytes -= (sizeof(int) + bytes);
            }
            return QStat_OK;
        }
        return QStat_OVERFLOW;
    }

    // an equal length version
    template<int QLen, int DataLen, template<int, int> class BufferType = CircularBuffer>
    class MwQueue2
    {
    public:
        class Reader;
        class Writer;

        // this is for multi-threaded environment using heap
        explicit MwQueue2(const char* queue_name = "ThreadMwQueue2") : m_name(queue_name) {
            static_assert(QLen == (QLen/DataLen*DataLen), "QLen has to be a multiple of DataLen");
        };

        // this is for multi-process environment using shm
        // be careful about the init to zero option. If set,
        // all existing contents are lost.  Usually we should
        // leave it to be false here.
        MwQueue2(const char* shm_name, bool read_only, bool init_to_zero = false) :
                m_name(shm_name), m_buffer(shm_name, read_only, init_to_zero) {

            static_assert(QLen == (QLen/DataLen*DataLen), "QLen has to be a multiple of DataLen");
            if (!init_to_zero) {
                //auto ptr_write_pos = getPtrPosWrite();
                //auto ptr_dirty_pos = getPtrPosDirty();
                //auto ptr_ready_pos = getPtrPosReady();
                if (!read_only) {
                    //*ptr_dirty_pos = 0;
                    //*ptr_ready_pos = *ptr_write_pos;
                }
            }
        };

        ~MwQueue2() {
            for (size_t i = 0; i<_writersToDelete.size(); ++i) {
                delete (Writer*) (_writersToDelete[i]);
            }
            _writersToDelete.clear();
        };
        Reader* newReader() {
            // the caller responsible for deleting the instance
            return new Reader(*this);
        }
        Writer* newWriter() {
            // the caller responsible for deleting the instance
            return new Writer(*this);
        }

        Writer& theWriter() {
            Writer* wtr = new Writer(*this);
            _writersToDelete.push_back(wtr);
            return *wtr;
        }

        volatile QPos *getPtrPosWrite() const { return (volatile QPos*) m_buffer.getHeaderStart() ; };
        volatile QPos *getPtrPosDirty() const { return (volatile QPos*) (m_buffer.getHeaderStart() + sizeof(QPos)) ; };
        volatile QPos *getPtrPosReady() const { return (volatile QPos*) (m_buffer.getHeaderStart() + 64); } ;
        std::vector<Writer*> _writersToDelete;
        static const int HeaderLen = 128;   // three 64-bit counters, occupying 2 cache lines
                                            // one for writers - write + dirty, one for read
                                            // also aligning to the cache line
    private:
        const std::string m_name;
        static const int ItemCount = (QLen / DataLen);
        BufferType<QLen, HeaderLen> m_buffer;

    public:
        // each reader will have have shared access to the queue,
        // has it's own read position
        class Reader {
        public:
            typedef MwQueue2<QLen, DataLen, BufferType> QType;
            explicit Reader(QType& queue)
            : m_buffer(queue.m_buffer),
              m_pos_write(queue.getPtrPosWrite()),
              m_ready_bytes(queue.getPtrPosReady()),
              m_pos(0)
              {
                syncPos();
              };

            QStatus copyNextIn(char* buffer) ;
            QStatus copyPosIn(char* buffer, QPos pos);

            // this doesn't check if the earlier writes
            // is also available.
            // It could be the case when this position is
            // ready for read but pos - 1 is not.
            // In this case this random access version will still
            // read pos, but the previous safe version will
            // wait until all writes below pos is ready. i.e.
            // m_ready_bytes >= pos.
            // Should be fine with Quite id, but not good
            // with sequential access of the multiple writers'
            // content such as log lines
            // This random access version still checks for over flows.
            // It should be sufficient for QuoteIDStore
            QStatus copyPosInRandomAccess(char* buffer, QPos pos);


            // this doesn't copy the bytes in most cases,
            // because QLen is a multiple of Datalen,
            // it should never go across the circular buffer boundary
            QStatus takeNextPtr(volatile char*& buffer) ;
            QStatus seekToTop();

            // this just set the reader position to the
            // existing writer position
            void syncPos() { m_pos = *m_ready_bytes; };
            void advance() { m_pos += DataLen; };
            void reset()   { m_pos = 0; };
            QPos getWritePos() const { return *m_ready_bytes;};
            ~Reader() {};
            std::string dump_state() const;
        private:
            const BufferType<QLen, QType::HeaderLen>& m_buffer;
            // readonly volatile pointer to a ever changing memory location
            const volatile QPos* const m_pos_write;
            const volatile QPos* const m_ready_bytes;
            QPos m_pos;
        };

        // each writer will have shared access to the queue
        class Writer {
        public:
            typedef MwQueue2<QLen, DataLen, BufferType> QType;
            explicit Writer(QType& queue)
            : m_buffer(queue.m_buffer),
              m_pos_write(queue.getPtrPosWrite()),
              m_pos_dirty(queue.getPtrPosDirty()),
              m_ready_bytes(queue.getPtrPosReady())
            {
            	// Just continue to write on the position
            	// of the existing positions
                // reset();
            };

            // be very careful calling this, content from all writers lost
            // void reset() { *m_pos_write = 0; *m_pos_dirty = 0; *m_ready_bytes = 0;};
            QPos put(const char* content);

            // this will fail (and return false) if
            // the immediate write is not possible,
            // when there were too many concurrent writes, 
            // the "dirty bytes" would exceed SpinThreashold
            bool putNoSpin(const char* content);
            long long get_pos_write() const {
            	return *m_pos_write;
            }
            std::string dump() const {
            	char buf[128];
            	snprintf(buf, sizeof(buf), "%lld, %lld, %lld",
            			(long long) *m_pos_write,
						(long long) *m_pos_dirty,
						(long long) *m_ready_bytes);
            	return std::string(buf);
            }

        private:
            BufferType<QLen, QType::HeaderLen>& m_buffer;
            volatile QPos* const m_pos_write;
            volatile QPos* const m_pos_dirty;
            volatile QPos* const m_ready_bytes;
            QPos getWritePos();
            void finalizeWrite();
            static const QPos SpinThreshold = QLen/4;
        };
    };

    // it first get the write pos, write a size, write the content
    // and check if it detects a synchronous point and update read
    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    QPos MwQueue2<QLen, DataLen, BufferType>::Writer::put(const char* content) {
        QPos pos = getWritePos();
        m_buffer.template copyBytes<true>(pos, (char*) content, DataLen);
        asm volatile("" ::: "memory");
        finalizeWrite();
        return pos;
    }

    // it first get the write pos, write a size, write the content
    // and check if it detects a synchronous point and update read
    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    bool MwQueue2<QLen, DataLen, BufferType>::Writer::putNoSpin(const char* content) {
        if (__builtin_expect(((QPos)(*m_pos_write - *m_ready_bytes) >= SpinThreshold), 0)) {
            return false;
        }
        put(content);
        return true;
    }

    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    QPos MwQueue2<QLen, DataLen, BufferType>::Writer::getWritePos() {
        /*
        {
            // debug
            QPos posw = *m_pos_write;
            QPos posr = *m_ready_bytes;
            if (posw>posr) {
                printf("%lld-%lld\n",posw, posw-posr);
            }
        }
        */
        QPos pos;
        // get a time count and throw if wait for too long
        while (1) {
            pos = *m_pos_write;
            if (__builtin_expect(((QPos)(pos - *m_ready_bytes) < SpinThreshold), 1)) {
                QPos newPos = pos + DataLen;
                if (compareAndSwap(m_pos_write, pos, newPos) == pos) {
                    return pos;
                }
            }
            // TODO -
            // there is slight possiblity that some writers got killed 
            // during the write. This would make everyone after it spinning.
            // Maybe a safer way is to add a check at finalize in case
            // of not being able to commit. 
            // Each writer after that would demand the finalization be
            // finished, if not after a period, then collectively
            // reset write_pos to ready_bytes and retry everything.
            // However when resetting the write_pos, there maybe a writer
            // using stale pos in writing.  The way to guard against that
            // is to make the spin threshold very small, or even to be 1
            // But that took away the performance. A similar approach 
            // is to turn off the signal during the write, again they are
            // system calls to install handlers on per-write basis.
            // At minimum should let it fail if waiting for too long.
            // So the system could reset.
        }
    }

    // m_pos_dirty is the total bytes written so far
    // m_pos_write is the current write starting position
    // m_ready_bytes is the current point upto where content is readable
    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    void MwQueue2<QLen, DataLen, BufferType>::Writer::finalizeWrite() {
        QPos dirty = AddAndfetch(m_pos_dirty, DataLen);
        QPos pos = *m_pos_write;

        // if the slow writer guard is turned on
        /*
        if (__builtin_expect(dirty > pos, 0)) {
        	while (*m_pos_dirty > pos) {
        	    AddAndfetch(m_pos_dirty, -DataLen);
        	    dirty -= DataLen;
        	}
        }
        */

        if (__builtin_expect( dirty == pos, 1)) {
            // detected a sync point
            // ready bytes are at least dirty bytes
            // try to update the read position
            QPos prev_ready_bytes = *m_ready_bytes;
            while (prev_ready_bytes < dirty) {
                // multiple writers may be updating ready_bytes
                // take the highest dirty value
                prev_ready_bytes = compareAndSwap(m_ready_bytes, prev_ready_bytes, dirty);
            }
        }

        // this is to guard against a slow write
        // blocking in the writing stage
        // Intended for heavily loaded faulty systems
        /*
        const int MAX_CPU_CORES = 128;
        if (__builtin_expect( (pos-dirty)/DataLen > MAX_CPU_CORES ,0)) {
        	AddAndFetch(m_pos_dirty, DataLen);
        	return;
        }
        */
    }

    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue2<QLen, DataLen, BufferType>::Reader::takeNextPtr(volatile char*& buffer) {
        if (__builtin_expect((*m_pos_write - m_pos >= QLen), 0)) {
            return QStat_OVERFLOW;
        }
        QPos unread_bytes = *m_ready_bytes - m_pos;
        if (unread_bytes <= 0) {
            if (__builtin_expect((unread_bytes < 0),0)) {
                syncPos();
            }
            return QStat_EAGAIN;
        }
        // get size
        buffer = m_buffer.getBufferPtr(m_pos);
        return QStat_OK;
    };

    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue2<QLen, DataLen, BufferType>::Reader::copyNextIn(char* buffer) {
        if (__builtin_expect((*m_pos_write - m_pos >= QLen), 0)) {
            return QStat_OVERFLOW;
        }
        QPos unread_bytes = *m_ready_bytes - m_pos;
        if (unread_bytes <= 0) {
            if (__builtin_expect((unread_bytes < 0),0)) {
                syncPos();
            }
            return QStat_EAGAIN;
        }
        // get size
        m_buffer.template CopyBytesFromBuffer(m_pos, buffer, DataLen);
        // check after copy
        if (__builtin_expect((*m_pos_write - m_pos >= QLen), 0)) {
            return QStat_OVERFLOW;
        }
        return QStat_OK;
    }

    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    std::string MwQueue2<QLen, DataLen, BufferType>::Reader::dump_state() const {
    	char buf[128];
    	snprintf(buf, sizeof(buf), "%lld %lld %lld",
    			(long long) *m_pos_write,
				(long long) *m_ready_bytes,
				(long long) m_pos);
    	return std::string(buf);
    }

    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue2<QLen, DataLen, BufferType>::Reader::copyPosIn(char*buffer, QPos pos) {
    	if (__builtin_expect((*m_pos_write - pos >=QLen),0)) {
    		return QStat_OVERFLOW;
    	}
    	QPos unread_bytes = *m_ready_bytes - pos;
    	if (__builtin_expect( (unread_bytes <= 0 ),0)) {
    		// spin for a short time of not too far
    		if (unread_bytes / DataLen > -10) {
    			int spin = 10000;
    			while (--spin > 0) {
    				unread_bytes = *m_ready_bytes - pos;
    				if (unread_bytes > 0)
    					break;
    			}
    			if (spin <= 0) {
    				return QStat_EAGAIN;
    			}
    		} else {
    		    return QStat_EAGAIN;
    		}
    	}
    	m_buffer.template CopyBytesFromBuffer(pos, buffer, DataLen);
    	// check after copy
    	if (__builtin_expect((*m_pos_write - pos >=QLen),0)) {
    		return QStat_OVERFLOW;
    	}
    	return QStat_OK;
    }

    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
	QStatus MwQueue2<QLen,DataLen,BufferType>::Reader::copyPosInRandomAccess(char*buffer,QPos pos) {
    	if (__builtin_expect((*m_pos_write - pos >=QLen),0)) {
    		return QStat_OVERFLOW;
    	}
    	m_buffer.template CopyBytesFromBuffer(pos, buffer,DataLen);
    	// check after copy
		if (__builtin_expect((*m_pos_write - pos >=QLen),0)) {
			return QStat_OVERFLOW;
		}
    	return QStat_OK;
    }

    // it traverse all the way to top, and get the position of top update,
    template<int QLen, int DataLen, template<int, int> class BufferType>
    inline
    QStatus MwQueue2<QLen, DataLen, BufferType>::Reader::seekToTop() {
        int unread_bytes = *m_ready_bytes - m_pos;
        if (unread_bytes == 0)
            return QStat_EAGAIN;

        m_pos = *m_ready_bytes - DataLen;
        return QStat_OK;
    }

}
#pragma GCC diagnostic pop /* "-Werror=volatile" */
