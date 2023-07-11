#include "ExecutionReport.h"
#include "TPClient.h"
#include "time_util.h"
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <csignal>
#include "symbol_map.h"

namespace pm {

    template<typename Timer>
    class OrderConnectionMock {
    public:
        OrderConnectionMock(const std::string& name, std::string existing_er_file)
        : _name(name), 
          _cur_micro( Timer::cur_micro() ),
          _cur_id(_cur_micro),
          _cur_exid(_cur_micro),
          _should_run(false),
          _floor(name, *this)
        {
            if (existing_er_file.size() > 0) {
                populate_history_fill(existing_er_file);
            }
        };

        void run() {
            _should_run = true;
            while (_should_run) {
                _cur_micro = Timer::cur_micro();
                const auto& iter = _to_publish.begin();
                if (iter != _to_publish.end() && iter->first < _cur_micro) {
                    // publish the next message
                    // if it's a fill, append to history_fill history
                    auto& msg = iter->second;
                    _floor.sendMsg(*msg);
                    if (msg->type == FloorBase::ExecutionReport) {
                        ExecutionReport* er = (ExecutionReport*)(msg->buf);
                        if (er->isFill()) {
                            _history_fill.push_back(msg);
                        }
                    }
                    _to_publish.erase(iter);
                    continue;
                }
                if (!_floor.run_one_loop(_floor)) // this populates the queueu
                {
                    Timer::micro_sleep(100000);
                }
            }
        }

        void stop() {
            logInfo("%s stopping...", _name.c_str());
            _should_run = false; 
        };

        std::string sendOrder(const char* data) {
            // order string is supposed to be the following
            // B/S algo, symbol, qty, px
            try {
                size_t size = strlen(data);
                if (size < 5) {
                    return  "unknown order string of size "+std::to_string(size);
                }
                int bs = (data[0]=='B'?1:-1);
                auto line = utils::CSVUtil::read_line(std::string(data+1));

                // new scheduled in 0.1 seconds
                ++_cur_id; ++_cur_exid;
                auto new_micro = _cur_micro + 100000;
                ExecutionReport er_new ( line[1], line[0], 
                        "cid"+std::to_string(_cur_id), "exid"+std::to_string(_cur_exid),
                        "0",  // tag39 = new
                        std::stoi(line[2])*bs,
                        std::stod(line[3]),
                        utils::TimeUtil::frac_UTC_to_string(new_micro/1000ULL, 3),
                        "",
                        new_micro,
                        0);
                enq(er_new);

                // fill scheduled in 0.5 second
                ++_cur_exid;
                auto fill_micro = _cur_micro + 500000;
                ExecutionReport er_fill ( line[1], line[0], 
                        "cid"+std::to_string(_cur_id), "exid"+std::to_string(_cur_exid),
                        "2",  // tag39 = fill 
                        std::stoi(line[2])*bs,
                        std::stod(line[3]),
                        utils::TimeUtil::frac_UTC_to_string(fill_micro/1000ULL, 3),
                        "",
                        fill_micro,
                        0);
                enq(er_fill);
            } catch (const std::exception& e) {
                return e.what();
            }
            return "";
        }

        std::string requestReplay(const char* data, int size, std::string& ready_file) {
            // data is in format of "from_utc, filepath"
            // write all the existing ers in the history fill (after from_utc) into the given file
            // schedule a message of done in 2 seconds
            // return errstr.
            // set the ready_file in case the replay is ready, and
            // therefore will trigger an additional ExecutionReplayDone
            try {
                auto tk = utils::CSVUtil::read_line(std::string(data));
                uint64_t from_micro = utils::TimeUtil::string_to_frac_UTC(tk[0].c_str(), 6);

                logInfo("received replay request to file %s from %s (%lld)",
                        tk[1].c_str(), tk[0].c_str(), (long long)from_micro);

                utils::CSVUtil::FileTokens lines;
                // _history_fill is maintained as all past fills published
                for (const auto& msg: _history_fill) {
                    const ExecutionReport* er = (ExecutionReport*)(msg->buf);
                    if (er->m_recv_micro > from_micro) {
                        lines.push_back(er->toCSVLine());
                    }
                }
                std::string fn(tk[1]);
                utils::CSVUtil::write_file(lines, fn, false);
                _to_publish.emplace( _cur_micro + 2000000,  
                                     std::make_shared<FloorBase::MsgType>(
                                         FloorBase::ExecutionReplayDone, 
                                         fn.c_str(), 
                                         fn.size()+1)
                                   );
            } catch (const std::exception& e) {
                return e.what();
            }
            return "";
        }

        std::string requestOpenOrder(const char* data, int size) {
            // schedule a message of one open order in 0.1 second
            // the open order will never be filled
            // return errstr

            std::string sym = utils::SymbolMapReader::get().getByMtsSymbol("WTI_N1")->_tradable;
            static std::string line = sym + ", algo1,cid1, eid1, 0,-10, 3.0, 20210124-18:33:02,, 1601850782023138";
            try {
                ExecutionReport er_oo = ExecutionReport::fromCSVLine(utils::CSVUtil::read_line(line));
                er_oo.m_recv_micro=_cur_micro+1000000;
                enq(er_oo);
            } catch (const std::exception& e) {
                return e.what();
            }
            return "";
        }

        std::string toString() const {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s: cur_micro: %s, hist_queue_size: %d, pub_queue_size: %d\n", _name.c_str(), utils::TimeUtil::frac_UTC_to_string(_cur_micro, 6).c_str(), (int)_history_fill.size(), (int)_to_publish.size());
            return std::string(buf);
        }

    private:
        const std::string _name;
        std::map<uint64_t, std::shared_ptr<FloorBase::MsgType> > _to_publish;
        std::vector<std::shared_ptr<FloorBase::MsgType> > _history_fill;
        uint64_t _cur_micro;
        uint64_t _cur_id, _cur_exid;
        volatile bool _should_run;
        FloorClientOrder<OrderConnectionMock<Timer> > _floor;

        void populate_history_fill(const std::string& csv_file) {
            auto er_lines = utils::CSVUtil::read_file(csv_file);
            for (const auto& line: er_lines) {
                const auto er = ExecutionReport::fromCSVLine(line);
                if (er.isFill()) {
                    _history_fill.emplace_back(
                            std::make_shared<FloorBase::MsgType>(
                                FloorBase::ExecutionReport,
                                (const char*)&er,
                                sizeof(ExecutionReport)));
                }
            }
        }

        void enq(const ExecutionReport& er) {
            _to_publish.emplace(er.m_recv_micro, 
                                std::make_shared<FloorBase::MsgType> (
                                    FloorBase::ExecutionReport,
                                    (const char*)&er,
                                    sizeof(ExecutionReport))
                               );
        }
    };
};

std::shared_ptr<pm::OrderConnectionMock<utils::TimeUtil> > ocm;

void signal_handler(int signal)
{
    ocm->stop();
    std::atomic_signal_fence(std::memory_order_release);
    logInfo("received signal %d, floor manager stop() called...\ndump state\n%s",
            signal, ocm->toString().c_str());
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::string previou_fill_file = "";
    std::string previous_fill_file;
    if (argc > 1) {
        previous_fill_file = std::string(argv[1]);
    }
    ocm = std::make_shared<pm::OrderConnectionMock<utils::TimeUtil> >("mock_exchange", previous_fill_file);
    ocm->run();
    return 0;
}

