#pragma once

#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

/*
 * This is the multi-process version of lock-free queue
 * Shm is used, processes need to configure a shm file name as
 * the queue name, same for writers and readers.
 */
namespace utils {

    enum QStatus {
        QStat_OK,
        QStat_EAGAIN,
        QStat_OVERFLOW,
        QStat_ERROR
    };

    inline
    const char* QStatStr(QStatus stat) {
        switch (stat) {
        case QStat_OK: return "QStat_OK";
        case QStat_EAGAIN: return "QStat_EAGAIN";
        case QStat_OVERFLOW: return "QStat_OVERFLOW";
        case QStat_ERROR: return "QStat_ERROR";
        default:
            return "UNKnown";
        }
    }

    typedef long long QPos;

    inline QPos fetchAndAdd(volatile QPos* pos, int val) {
        return __sync_fetch_and_add(pos, val);
    }
    inline QPos AddAndfetch(volatile QPos* pos, int val) {
        return __sync_add_and_fetch(pos, val);
    }
    inline QPos compareAndSwap(volatile QPos* pos, QPos old_pos, QPos new_pos) {
        return __sync_val_compare_and_swap(pos, old_pos, new_pos);
    }

    template<bool isInbound>
    inline void CopyBytes(char* buf1, char* buf2, int bytes) {};
    template<>
    inline void CopyBytes<true>(char* buf1, char* buf2, int bytes) {
            memcpy(buf1, buf2, bytes);
    }
    template<>
    inline void CopyBytes<false>(char* buf1, char* buf2, int bytes) {
            memcpy(buf2, buf1, bytes);
    }

    // The circular buffer stores the content in the first QLen bytes
    // and the header information at the end
    template<int QLen, int HeaderLen>
    class CircularBuffer {
    public:
        CircularBuffer() : m_buffer((char*)malloc(QLen + HeaderLen)), m_buffer_given(false) {
            // removing this restriction.  Not a bid deal to use modulo QLen vs. & QMask
            //static_assert(((QLen-1)&(QLen)) == 0, "CircularBuffer: qlen not power of 2");
            if (!m_buffer)
                throw std::runtime_error("CircularBuffer: malloc failed");
            // zero out everything
            memset((void*)m_buffer, 0, QLen + HeaderLen);
        }

        explicit CircularBuffer(char* buf) : m_buffer(buf), m_buffer_given(true) {
            // removing this restriction
            // static_assert(((QLen-1)&(QLen)) == 0, "CircularBuffer: qlen not power of 2");
        }

        // adding this for the shm interface
        explicit CircularBuffer(const char*, bool, bool) :
                m_buffer((char*)malloc(QLen + HeaderLen)), m_buffer_given(false) {

            // shouldn't allow this?
            if (!m_buffer)
                throw std::runtime_error("CircularBuffer: malloc failed");
            // zero out everything
            memset((void*)m_buffer, 0, QLen + HeaderLen);
        }

        ~CircularBuffer() {
            if (!m_buffer_given)
            {
                free((void*)m_buffer);
                m_buffer = NULL;
            }
        }
        volatile char* getHeaderStart() const { return m_buffer + QLen; };
        volatile char* getBufferStart() const { return m_buffer; };

        // this will break if msg len are larger than QLen
        // this should be assured by the caller. Otherwise
        // behavior is unspecified
        // it returns the pointer of the next write location
        // within the circular buffer
        //
        //
        // We have three sets of copyBtes interfaces
        // copyBytes<ifInbound>(char*, char*, len) is the lowest memcopy
        // copyBytes<ifInboud>(start_pos, *content, bytes) mainly used by write (maybe a change?)
        // copyBytesFromBuffer(start_pos, *content, bytes) used by reader only
        //
        template<bool ifInbound>
        void copyBytes(QPos start_pos, const char* content, int bytes) volatile const {
            int s_pos = start_pos % QLen;
            int e_pos = (start_pos + bytes) % QLen;
            if (__builtin_expect((e_pos < s_pos),0)) {
                int len1 = QLen - s_pos;
                CopyBytes<ifInbound>((char*) (m_buffer + s_pos), (char*) content, len1);
                content += len1;
                bytes -= len1;
                s_pos = 0;
            }
            CopyBytes<ifInbound>((char*) (m_buffer + s_pos), (char*) content, bytes);
        }

        template<bool ifInbound>
        void copyBytesNoCross(QPos start_pos, const char* content, int bytes) volatile const {
            int s_pos = start_pos % QLen;
            CopyBytes<ifInbound>((char*) (m_buffer + s_pos), (char*) content, bytes);
        }

        inline void CopyBytesFromBuffer(QPos start_pos, const char* content, int bytes) volatile const {
        	int s_pos = start_pos % QLen;
        	int e_pos = (start_pos + bytes) % QLen;
        	if (__builtin_expect((e_pos < s_pos),0)) {
        		int len1 = QLen - s_pos;
        		CopyBytes<false>((char*) (m_buffer+s_pos),(char*) content, len1);
        		content += len1;
        		bytes -=len1;
        		s_pos = 0;
        	}
        	CopyBytes<false>((char*) (m_buffer + s_pos),(char*)content,bytes);
        }

        inline void CopyBytesFromBufferNoCross(QPos start_pos, const char* content, int bytes) volatile const {
        	int s_pos = start_pos % QLen;
        	CopyBytes<false>((char*) (m_buffer + s_pos),(char*)content,bytes);
        }

        // check if the contents of bytes starting from start_pos would
        // cross the boundary.  If not, buffer will be the starting pointer
        bool wouldCrossBoundary(const QPos start_pos, int bytes, volatile char*& buffer) const volatile {
            QPos s_pos = (start_pos % QLen);
            buffer = m_buffer + s_pos;
            return s_pos > ((start_pos + bytes) % QLen);
        }

        volatile char* getBufferPtr(QPos start_pos) const {
            return m_buffer + (start_pos % QLen) ;
        }

    protected:
        void setBuffer(volatile char* buf) {
            if ( !m_buffer_given ) {
                delete m_buffer;
                m_buffer_given = true;
            }
            m_buffer = buf;
        }

    private:
        volatile char* m_buffer;
        //static const int QMask = QLen - 1;
        bool m_buffer_given;
    };

    template<int QLen, int HeaderLen>
    class ShmCircularBuffer : public CircularBuffer<QLen, HeaderLen>  {
    public:
        ShmCircularBuffer(const char* shm_name, bool is_read, bool init_to_zero):
        CircularBuffer<QLen, HeaderLen>(NULL), m_shm_name(shm_name), m_is_read(is_read), m_shm_size(QLen + HeaderLen),
        m_shm_ptr(NULL), m_shm_fd(-1)
        {
        	if (!shm_name)
        		// this is the case where an empty buffer is created
        		return;

            if (is_read) {
                m_shm_fd = shm_open(m_shm_name.c_str(), O_RDONLY, S_IRUSR);
                if (m_shm_fd == -1) {
                    // could this because the file does not exist?
                    m_shm_fd = shm_open(m_shm_name.c_str(), O_RDWR | O_CREAT, S_IRUSR| S_IWUSR);
                    if (m_shm_fd != -1) {
                        struct stat st;
                        if ((fstat(m_shm_fd, &st)!=0) || ((int) st.st_size != m_shm_size))
                        {
                            if (ftruncate(m_shm_fd, m_shm_size) == -1) {
                                throw std::runtime_error("Failed to truncate shm file");
                            };
                        }
                        ::close(m_shm_fd);
                        m_shm_fd = shm_open(m_shm_name.c_str(), O_RDONLY, S_IRUSR);
                    }
                }
            } else {
                m_shm_fd = shm_open(m_shm_name.c_str(), O_RDWR | O_CREAT, S_IRUSR| S_IWUSR);
            }
            if (m_shm_fd == -1) {
                std::string errstr(m_shm_name +
                        std::string(": ShmCircularBuffer: shm_open failed") +
                        std::string(strerror(errno)));
                throw  std::runtime_error(errstr.c_str());
            };

            // truncate if needed
            struct stat st;
            if ((fstat(m_shm_fd, &st)!=0) || ((int) st.st_size != m_shm_size))
            {
                if (ftruncate(m_shm_fd, m_shm_size) == -1) {
                    throw std::runtime_error("Failed to truncate shm file");
                };
            }

            // mmap
            if (is_read) {
                m_shm_ptr = mmap(NULL, m_shm_size, PROT_READ, MAP_SHARED, m_shm_fd, 0);
            } else {
                m_shm_ptr = mmap(NULL, m_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_shm_fd, 0);
            }

            if (m_shm_ptr == MAP_FAILED) {
                std::string errstr(m_shm_name +
                        std::string(": ShmCircularBuffer: mmap failed") +
                        std::string(strerror(errno)));
                throw  std::runtime_error(errstr.c_str());
            }

            if ((!is_read) && init_to_zero) {
                memset((uint8_t*)m_shm_ptr, 0, m_shm_size);
            }

            this->setBuffer((char*)m_shm_ptr);
        }

        ~ShmCircularBuffer() {
            if (m_shm_fd != -1) {
                munmap((void*)m_shm_ptr, m_shm_size);
                //if (!m_is_read) {
                //    shm_unlink(m_shm_name.c_str());
                //}
                ::close(m_shm_fd);
                m_shm_fd = -1;
            }
        }

    private:
        const std::string m_shm_name;
        const bool m_is_read;
        const int m_shm_size;
        void* m_shm_ptr;
        int m_shm_fd;
    };
}
