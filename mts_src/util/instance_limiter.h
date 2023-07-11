#pragma once
#include "time_util.h"
#include "queue.h"
#include "csv_util.h"

#include <stdexcept>
#include <exception>

#include <unordered_set>
#include <memory>


/*
 * This is a runtime tool to ensure there is only one
 * instance running with the name given at constructor.
 * It uses a heartbeat queue to detect multiple updates,
 * user can periodically check the uniqueness whenever needed.
 */

namespace utils {
    class OnlyMe {
    public:
        enum ConfigType {
            MAX_NAME_SIZE = 256,
            DEFAULT_BUF_SIZE = 1024,
        };

        // singleton instance
        static OnlyMe& get();

        // populate the hashmap and update the _buf as csv line of names
        // prepare the _buf and the _buf_size to reflect the state
        // return true if insertion/removal took place
        bool add_name(const std::string& name);
        bool remove_name(const std::string& name);
        void remove_all();

        // check if the instances in _name exist, without updating
        // the queue.  So the existing instance won't detect duplication.
        bool check_only();

        // same behavior as check_only, in addition, regardless of 
        // the result, it announces the instances in _names so others are aware
        bool check();

        ~OnlyMe();
    private:
        static const int QLen = 1024*64; // 256KBytes
        using QType = utils::MwQueue<QLen, utils::ShmCircularBuffer>;
        QType _heartbeat_q;
        QType::Writer _qw;
        QType::Reader _qr;
        QPos _qpos;
        char*  _buf;
        size_t _buf_size;
        size_t _buf_alloc_size;
        uint64_t* _last_micro;
        std::unordered_set<std::string> _names;

        OnlyMe(const char* queue_name);
        OnlyMe(const OnlyMe& onlyme) = delete;
        void operator=(const OnlyMe& onlyme) = delete;
        QPos update();
        bool check_line(char* line) const;
        size_t make_line();
    };

    inline OnlyMe& OnlyMe::get() {
        static OnlyMe onlyme = OnlyMe("onlyme.q");
        return onlyme;
    }

    bool OnlyMe::add_name(const std::string& name) {
        if (name.size() >= MAX_NAME_SIZE) {
            throw std::runtime_error(std::string("name is too long! ") + name);
        }
        bool inserted = _names.insert(name).second;
        if (inserted) {
            make_line();
        } else {
            printf("%s already added\n", name.c_str());
        }
        return inserted;
    }

    bool OnlyMe::remove_name(const std::string& name) {
        bool removed = _names.erase(name);
        if (removed) {
            make_line();
        } else {
            printf("%s not added yet\n", name.c_str());
        }
        return removed;
    }

    void OnlyMe::remove_all() {
        _names.clear();
        make_line();
    }

    bool OnlyMe::check_only() {
        volatile char* data;
        int data_len = 0;
        // make sure we read past the _qpos
        // _qpos is the position immediately after this instance's previous update
        // and read each data to match the instance name
        while (_qr.getReadPos() < _qpos) {
            QStatus ret = _qr.takeNextPtr(data, data_len);
            if (ret == QStat_OK) {
                asm volatile("" ::: "memory");
                _qr.advance(data_len);
                continue;
            }
            if (ret == QStat_EAGAIN) {
                // the previous write hasn't been finalized yet
                //printf("previous update not finalized yet, skip the check for %d names\n", (int)_names.size());
                return true;
            }
            printf("error in queue read, syncing and skip the check for %d names\n", (int)_names.size());
            _qr.syncPos();
            return true;
        }

        bool check_result = true;
        // this would copy if data go across boundry of circular buffer
        while (true) {
            QStatus ret = _qr.takeNextPtr(data, data_len);
            if (ret == QStat_OK) {
                bool check_ok_;
                try {
                    check_ok_ = check_line((char*) data+sizeof(long long));
                } catch (const std::exception& e) {
                    printf("exception caught while checking line: %s\n", e.what());
                    check_ok_ = false;
                }
                if (__builtin_expect(!check_ok_,0)) {
                    long long* mptr = (long long*)data;
                    printf("failed check on %d names, micro=%lld\n", (int)_names.size(), *mptr);
                    check_result = false;
                }
                asm volatile("" ::: "memory");
                _qr.advance(data_len);
                continue;
            }
            if (ret == QStat_EAGAIN) {
                break;
            }
            printf("onlyme problem reading at check() status %d, qr_read_pos %lld, qr_ready_pos %lld syncing...\n", (int) ret, (long long) _qr.getReadPos(), (long long) _qr.getReadyPos());
            _qr.syncPos();
            break;
        }
        _qpos = _qr.getReadPos();
        return check_result;
    }

    inline
    bool OnlyMe::check() {
        bool check_result = check_only();
        if (_names.size()>0) {
            _qpos = update();
        }
        return check_result;
    }

    inline
    OnlyMe::~OnlyMe() {
        if (_buf) {
            free (_buf);
            _buf=0;
        }
    }

    inline
    QPos OnlyMe::update() {
        *_last_micro = utils::TimeUtil::cur_micro();
        _qw.put((char*)_buf, _buf_size);
        return _qw.writePos();
    }

    OnlyMe::OnlyMe(const char* /* queue_name */) :
        _heartbeat_q("onlyme.q", false, false), 
        _qw(_heartbeat_q),
        _qr(_heartbeat_q),
        _qpos(0),
        _buf(0),
        _buf_size(0),
        _buf_alloc_size(DEFAULT_BUF_SIZE),
        _last_micro(0)
    {
        _buf = (char*) malloc(_buf_alloc_size);
        if (!_buf) {
            throw std::runtime_error("cannot allocate memory!");
        }
        memset((void*)_buf, 0, _buf_alloc_size);
        _last_micro = (uint64_t*)_buf;
        _qr.syncPos();  // this gets the m_ready_bytes
        _qpos = _qw.writePos();
    }

    bool OnlyMe::check_line(char* line) const {
        // get from line to an vector of names
        // check each one against the hashmap
        const auto& names_ (utils::CSVUtil::read_line(line, ','));
        bool check_result = true;
        for (const auto& n : names_) {
            if (__builtin_expect( _names.find(n) != _names.end(), 0)) {
                printf("Duplicate %s\n",n.c_str());
                check_result = false;
            }
        }
        return check_result;
    };

    size_t OnlyMe::make_line() {
        // prepare the _buf with the _names
        _buf_size = 0;
        if (_names.size() == 0) {
            return _buf_size;
        }
        for (const auto& name : _names) {
            _buf_size += name.size()+1;
        }
        _buf_size += sizeof(long long);
        if (_buf_alloc_size < _buf_size) {
            _buf_alloc_size = 2*_buf_size;
            if (_buf) {
                free(_buf);
                _buf = 0;
            }
            _buf = (char*) malloc(_buf_alloc_size);
            if (!_buf) {
                throw std::runtime_error(std::string("cannot allocate size of ") + std::to_string(_buf_alloc_size) + " for size of names " + std::to_string(_names.size()));
            }
            _last_micro = (uint64_t*)_buf;
        }
        char * ptr = _buf + sizeof(long long);
        size_t bytes = 0;
        for (const auto& name : _names) {
            bytes += sprintf(ptr+bytes, "%s,", name.c_str());
        }
        _buf[_buf_size-1] = 0;
        return _buf_size;
    }
}
