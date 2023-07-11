#include <memory>
#include <map>
#include <set>
#include <iostream>
#include <stdio.h>
#include <mutex>
#include "time_util.h"
#include "queue.h"
#include "plcc/PLCC.hpp"

namespace utils {

    // Event channel that handles message passing between
    // clients: i.e. algo, tp
    // servers: i.e. traders

    class Floor {
    public:
        // TODO - maybe protect the buf/buf_capacity
        // as private if needed in the future usage cases
        class Message {
        public:
            // a stateless message that doesn't demand response
            int type;
            mutable uint64_t ref;
            char* buf;
            size_t data_size;
        private: 
            size_t buf_capacity;

        public:
            //uint64_t id;
            Message() : type(0), ref(NOREF), buf(nullptr), data_size(0), buf_capacity(0) {}

            Message(int type_, const char* data_, size_t size_, uint64_t ref_=NOREF)
            : type(type_), ref(ref_), buf((char*)malloc(size_)), data_size(size_), buf_capacity(size_) 
            {
                if (!data_) 
                    data_size = 0;
                else {
                    if (data_size)
                        memcpy(buf, data_, size_);
                }
            }

            Message(const Message& msg)
            : type(msg.type), ref(msg.ref), buf(nullptr), data_size(0), buf_capacity(0)
            {
                if (msg.data_size) {
                    copyData(msg.buf, msg.data_size);
                }
            }

            void operator = (const Message& msg)
            {
                type = msg.type;
                ref = msg.ref;
                if (msg.data_size) {
                    copyData(msg.buf, msg.data_size);
                } else {
                    data_size = 0;
                }
            }

            ~Message() {
                if (buf && buf_capacity) {
                    buf_capacity=0;
                    free (buf);
                    buf=nullptr;
                }
            }

            void copyData(const char* data, const size_t data_size_) {
                reserve(data_size_);
                memcpy(buf, data, data_size_);
                data_size = data_size_;
            }

            void copyString(const std::string& str) {
                copyData(str.c_str(), str.size()+1);
            }

            std::string toString() const {
                char strbuf[128];
                snprintf(strbuf, sizeof(strbuf), 
                        "Floor Message (type: %d, ref: %llu, buf: %p, data_size: %d, buf_capacity: %d)",
                        (int) type, (unsigned long long) ref, buf, (int) data_size, (int) buf_capacity);
                return std::string(strbuf);
            }

            bool refSet() const {
                return ref != NOREF;
            }

            void removeRef() const {
                ref = NOREF;
            }

            void reserve(size_t data_size_) {
                if (buf_capacity<data_size_) {
                    if (buf) {
                        free (buf);
                    }
                    buf = (char*)malloc(data_size_*2);
                    buf_capacity = data_size_*2;
                }
            };

            static const uint64_t NOREF = static_cast<uint64_t>(-1);
        };

        static Floor& get() {
            static Floor flr(nullptr);
            return flr;
        };

        static std::shared_ptr<Floor> getByName(const char* flr_name) {
            static std::map<std::string, std::shared_ptr<Floor> > _floor_map;
            static std::mutex _lock;

            auto iter = _floor_map.find(flr_name);
            if (iter != _floor_map.end()) {
                return iter->second;
            }
            {
                std::lock_guard<std::mutex> gard(_lock);
                auto iter = _floor_map.find(flr_name);
                if (iter != _floor_map.end()) {
                    return iter->second;
                }
                std::shared_ptr<Floor> flr_ptr( new Floor(flr_name) );
                _floor_map.emplace(flr_name, flr_ptr);
                return flr_ptr;
            }
        }

        ~Floor() {};

        class Channel;
        std::unique_ptr<Channel> getClient() const {
            // create a channel for write_qin, read_qout
            return std::unique_ptr<Channel>(new Channel(_qout, _qin));
        }

        std::unique_ptr<Channel> getServer() const {
            // create a channel for read_qin, write_qout
            return std::unique_ptr<Channel>(new Channel(_qin, _qout));
        }


        /*
        std::shared_ptr<QType> get_qin() const {
            return _qin;
        }

        std::shared_ptr<QType> get_qout() const {
            return _qout;
        }
        */

        static const int QLen = 1024*1024*64; // 64M
        using QType = utils::MwQueue<QLen, utils::ShmCircularBuffer>;
    private:
        std::shared_ptr<QType> _qin, _qout;

        explicit Floor(const char* name)
        : _qin (std::make_shared<QType>( (name? (std::string(name)+"_fq1").c_str() : "fq1"), false, false)), 
          _qout(std::make_shared<QType>( (name? (std::string(name)+"_fq2").c_str() : "fq2"), false, false))
        {
            // createt the two queues read+write, without init to zero
        };

        Floor(const Floor&) = delete ;
        void operator=(const Floor&) = delete;

    public:
        class Channel {
        public:
            Channel(std::shared_ptr<QType> qread, std::shared_ptr<QType> qwrite)
            : _qin(qread), _qout(qwrite),
              _reader(std::make_shared<QType::Reader>(*_qin)),
              _writer(std::make_shared<QType::Writer>(*_qout))
            {}

            Channel() {}; // dummy channel

            bool request(const Message& req, Message& resp, int timeout_sec=1) {
                // request encode the type, msg and size
                // resp will be byte-wise copied from the queue, with resp_size set accordingly. 
                // timeout_sec is the longest time to wait after when it returns false.
                // Memory allocation wise, caller is responsible to allocate enough memory in the
                // resp message and set the size as available. Upon return, the resp byte-wise
                // copied and size updated. It returns false if size is not enough.
                auto reader (std::make_shared<QType::Reader>(*_qin));
                return sendSync(req, &resp, timeout_sec, reader);
            }

            bool requestWithReader(const Message& req, Message& resp, std::shared_ptr<QType::Reader>& reader, int timeout_sec=5) {
                // same as request(), also gives the reader and could be used in nextMessage monitoring
                if (!reader) {
                    reader = std::make_shared<QType::Reader>(*_qin); 
                }
                return sendSync(req, &resp, timeout_sec, reader);
            }

            uint64_t update(const Message& upd) {
                // send a update and return a reference number
                return sendMessage(upd.type, upd.buf, upd.data_size, upd.ref);
            }

            // this reads the next message from the read queue using the 
            // channel's _reader 
            bool nextMessage(Message& msg) {
                return nextMessage(&msg, _reader, true);
            }

            void addSubscription(const std::set<int>& type_set) {
                for (auto tp : type_set) {
                    _subscribed_types.insert(tp);
                }
            }

            void removeSubscription(const std::set<int>& type_set) {
                for (auto tp : type_set) {
                    _subscribed_types.erase(tp);
                }
            }

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

            // for simple cases when sending request that expects an acknowledgement (Ack)
            // it assumes resp is a string
            bool requestAndCheckAck(const Message& req, Message& resp, int timeout_sec, int ack_type) {
                resp.copyString("Channel Error");
                if (! request(req, resp, timeout_sec)) {
                    logError("Error sending request: %s", req.toString().c_str());
                    return false;
                }

                // debug dump
                //outputBuf(resp.buf, resp.data_size);

                if ( (resp.type != ack_type) || (strncmp(resp.buf, "Ack",3)!=0)) {
                    logError("Received message without Ack. resp_type = %d, expected_type=%d, msg=%s", 
                            (int) resp.type, (int) ack_type, resp.buf);
                    return false;
                }
                return true;
            }

            // sending acknowledgement for a request
            void updateAck(const Message& req, Message& resp, int ack_type, const std::string& ack_message = "Ack") {
                const std::string& ackstr = (ack_message.size()==0? "Ack":ack_message);
                resp.type = ack_type;
                resp.ref = req.ref;
                resp.copyString(ackstr);
                update(resp);
            }

            static Channel getDummyChannel() {
                std::shared_ptr<QType> qin, qout;
                return Channel(qin, qout);
            }

            // for monitoring purpose, this gets all message with ref from reader given
            // if ref is NOREF, then gets all msg.  
            // non-blocking, returns true if the msg is assigned, otherwise false
            bool nextMessageRef(Message* msg, std::shared_ptr<QType::Reader> reader=nullptr, uint64_t ref=utils::Floor::Message::NOREF) {
                volatile char* buf;
                int bytes;
                if (!reader) {
                    reader = _reader;
                }
                while (true) {
                    QStatus status = reader->takeNextPtr(buf, bytes);
                    if (status == utils::QStat_OK) {
                        msg->type = readMessage(buf, &(msg->ref));
                        reader->advance(bytes);
                        if ((ref==utils::Floor::Message::NOREF) || (msg->ref==ref)) {
                            msg->copyData((char*)buf, bytes-_hdrsize);
                            return true;
                        }
                        // try next one
                        continue;
                    } else {
                        if (status != utils::QStat_EAGAIN) {
                            // overflow or error, sync
                            logError("sendSync got read error from reader queue. %d, %s", (int)status, reader->dump_state().c_str());
                            reader->syncPos();
                        }
                    }
                    return false;
                }
            }

        private:
            std::shared_ptr<QType> _qin, _qout;
            std::shared_ptr<QType::Reader> _reader;
            std::shared_ptr<QType::Writer> _writer;
            std::set<int> _subscribed_types;
            static const int _hdrsize = sizeof(int) + sizeof(uint64_t);

            bool sendSync(const Message& req,  Message* resp, int timeout_sec, std::shared_ptr<QType::Reader>& reader) {
                // create a new reader (position sync'ed), send the request, record the sending 
                // position as reference, scan through the read queue for a message matching with referece
                // return true if found within timeout, otherwise, false
                req.removeRef();
                utils::QPos pos_ref = update(req);

                uint64_t timeout_micro = utils::TimeUtil::cur_micro() + (uint64_t)timeout_sec*1000000ULL;
                while (utils::TimeUtil::cur_micro() < timeout_micro) {
                    if ( nextMessage(resp, reader, false) ) {
                        if (resp->ref == (uint64_t)pos_ref) {
                            return true;
                        }
                        continue;
                    }
                    utils::TimeUtil::micro_sleep(10 * 1000);
                }
                logError("sendSync timeout while waiting for response. event_type: %d", (int) req.type);
                return false;
            }

            utils::QPos sendMessage(int et, char* msg_data, size_t msg_size, uint64_t ref= utils::Floor::Message::NOREF) {
                // header has format of type(int), ref(uint64_t)
                // returns the starting queue position where the message was written to. 
                // this is used at the reader side to assign reference to a request msg, i.e. a msg with ref=NOREF
                char buf[_hdrsize];
                memcpy(buf, &et, sizeof(int));
                memcpy(buf+sizeof(int), &ref, sizeof(uint64_t));
                return _writer->put(buf, _hdrsize, msg_data, msg_size);
            }

            int readMessage(volatile char*& buf, uint64_t* ref) {
                // given a raw message read from the queue as buf and bytes
                // parse the type and ref and adjust the buf and bytes for payload data
                
                int type = *(int*)buf;
                buf+=sizeof(int);
                *ref = *(uint64_t*)buf;
                buf+=sizeof(uint64_t);
                return type;
            }

            bool nextMessage(Message* msg, std::shared_ptr<QType::Reader> reader, bool filter_on) {
                // this reads next message from reader with filter.  
                // If the msg is a request, i.e. a msg with ref to be NOREF
                //     the msg ref assigned to be the reader position before reading the msg,
                //     this is the starting queue position of the message.
                //     the subsequent responses to this request will all have this ref set 
                // if the msg is a reply, 
                //     the ref already assigned to be the position of the request msg,
                //     it is used to match responses to requests
                // In case multiple reponse to a same request, i.e. multiple floor threads,
                // all responses have the same ref number of request.
                //
                volatile char* buf;
                int bytes;
                while (true) {
                    QStatus status = reader->takeNextPtr(buf, bytes);
                    if (status == utils::QStat_OK) {
                        msg->type = readMessage(buf, &(msg->ref));
                        if (! msg->refSet()) {
                            msg->ref = reader->getReadPos();
                        }

                        // debug
                        //logInfo("nextMessage received %d bytes, msg: %s, queue dump: %s", 
                        //        bytes, msg->toString().c_str(),
                        //        reader->dump_state().c_str());

                        asm volatile("" ::: "memory");
                        reader->advance(bytes);
                        if ( (!filter_on) ||
                             (_subscribed_types.find(msg->type) != _subscribed_types.end()) ) {
                            msg->copyData((char*)buf, bytes-_hdrsize);
                            return true;
                        }
                        // try next one
                        continue;
                    } else {
                        if (status != utils::QStat_EAGAIN) {
                            // overflow or error, sync
                            logError("sendSync got read error from reader queue. %d, %s", (int)status, reader->dump_state().c_str());
                            reader->syncPos();
                        }
                    }
                    return false;
                }
            }
        };
    };
}

