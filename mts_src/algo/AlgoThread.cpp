#include "AlgoThread.h"
#include "strat/AR1.h"
#include "strat/idbo/idbo_tf.h"

namespace algo {
    AlgoThread::AlgoThread(const std::string& inst, const std::string& cfg)
    : FloorBase(inst, true), m_inst(inst), m_strat_cfg(cfg), 
      m_floor(inst+"_client", false), m_should_run(false) 
    { 
        logInfo("AlgoThread %s created!", inst.c_str());
    }

    AlgoThread::~AlgoThread() {
        logInfo("AlgoThread %s destructed!", m_inst.c_str());
    };

    void AlgoThread::init() {
        const auto& vc = utils::ConfigureReader(m_strat_cfg.c_str());
        const auto& al = vc.getArr<std::string>("RunList");
        for (const auto& a : al) {
            const auto ac = vc.getArr<std::string>(a.c_str());
            // expect [class_name, config_file]
            addAlgo(a, ac[0], ac[1]);
            logInfo("AlgoThread %s: added strategies %s - %s", m_inst.c_str(), ac[0].c_str(), ac[1].c_str());
        }
        logInfo("AlgoThread %s: %d strategies added", m_inst.c_str(), (int) al.size());
    }

    void AlgoThread::reload(const std::string& algo, const std::string& cfg) {
        logInfo("Reloading algo %s %s", algo.c_str(), ((cfg.size()==0)?"":(std::string("(cfg:")+cfg+")").c_str()));
        m_algo_map[algo]->onReload(TimerType::cur_micro(), cfg);
    }

    void AlgoThread::reload(const std::string& algo ) {
        reload(algo, "");
    }

    void AlgoThread::reloadAll() {
        for (auto& am : m_algo_map) {
            reload(am.first);
        }
    }

    void AlgoThread::stop(const std::string& algo) {
        if (algo == "*") {
            // stopping all
            logInfo("Stopping All!");
            for (auto iter = m_algo_map.begin(); iter!= m_algo_map.end(); ++iter) {
                stop(iter->first);
            }
            return;
        }
        logInfo("Stopping algo %s", algo.c_str());
        m_algo_map[algo]->onStop(TimerType::cur_micro());
        m_algo_map[algo]->setShouldRun(false);
    }

    void AlgoThread::stopAll() {
        for (auto& am : m_algo_map) {
            stop(am.first);
        }
    }

    void AlgoThread::start(const std::string& algo) {
        logInfo("Startng algo %s", algo.c_str());
        m_algo_map[algo]->setShouldRun(true);
        m_algo_map[algo]->onStart(TimerType::cur_micro());
    }

    void AlgoThread::startAll() {
        for (auto& am : m_algo_map) {
            start(am.first);
        }
    }

    void AlgoThread::run() {
        init();

        m_should_run = true;
        setSubscriptions();
        const int max_sleep_micro = 50000;
        uint64_t cur_micro = TimerType::cur_micro();
        uint64_t next_micro = (cur_micro / 1000000 + 2)*1000000;

        startAll();
        logInfo("AlgoThread %s is running", m_name.c_str());

        while (m_should_run) {
            while( this->run_one_loop(*this) ) {
            };
            // call onOneSecond 
            cur_micro = TimerType::cur_micro();
            int64_t diff_time = next_micro - cur_micro;
            if (diff_time < 500) {
                while (diff_time > 10) {
                    cur_micro = TimerType::cur_micro();
                    diff_time = next_micro - cur_micro;
                }
                runOneSecond(cur_micro);
                next_micro += 1000000ULL;
            } else {
                // sleep upto max_sleep_micro
                if (diff_time > max_sleep_micro) {
                    diff_time = max_sleep_micro;
                }
                TimerType::micro_sleep(diff_time);
            }
        }

        logInfo("AlgoThread %s is done, stopping all", m_name.c_str());
        stopAll();
    }

    void AlgoThread::runOneSecond(uint64_t cur_micro) {
        for (auto& am : m_algo_map) {
            auto& ap = am.second;
            if (ap->shouldRun()) {
                ap->onOneSecond(cur_micro);
            }
        }
    }

    void AlgoThread::handleMessage(const MsgType& msg_in) {
        // the algo command is expected to be
        // 'L': list loaded strategies
        // 'strat_name S' : start 
        // 'strat_name R config_file' : reload
        // 'strat_name E' : stop 
        // 'strat_name D' : dump

        std::string respstr("Ack");
        switch (msg_in.type) {
            case FloorBase::AlgoUserCommand :
            {
                const char* cmd = msg_in.buf;
                const auto tk = utils::CSVUtil::read_line(cmd, ' ');
                const auto& sn(tk[0]);
                if (strcmp(sn.c_str(), "L")==0) {
                    respstr = toString();
                    logInfo("List loaded: %s", respstr.c_str());
                    break;
                }
                if (m_algo_map.find(sn) == m_algo_map.end()) {
                    logInfo("Strategy %s not found!", sn.c_str());
                    //respstr = std::string("Strategy ") + sn + " not found!";
                    break;
                };
                if (tk.size() < 2) {
                    logError("No command given for Strategy %s!", sn.c_str());
                    respstr = std::string("No command given for Strategy ") + sn + " \nRun help to see a list of supported commands.";
                    break;
                }

                const auto& c(tk[1]);
                switch (c[0]) {
                case 'S':
                {
                    logInfo("Starting strategy %s", sn.c_str());
                    start(sn);
                    break;
                }
                case 'E':
                {
                    logInfo("Stopping strategy %s", sn.c_str());
                    stop(sn);
                    break;
                }
                case 'R':
                {
                    if (tk.size() < 3) {
                        respstr = std::string("Reload command error!\n@strat_name R config_file");
                        logError("%s", respstr.c_str());
                        break;
                    }
                    const auto& cfg_file(tk[2]);
                    logInfo("Reload strategy %s with config %s", 
                            sn.c_str(), cfg_file.c_str());
                    reload(sn, cfg_file);
                    break;
                }
                case 'D':
                {
                    auto dumpstr = m_algo_map[sn]->toString();
                    logInfo("Dump %s: %s", sn.c_str(), dumpstr.c_str());
                    respstr = dumpstr;
                    break;
                }
                default :
                    logError("Unknown strategy command: %s", cmd);
                    respstr = "Unknown strategy command: " + std::string(cmd);
                }
                break;
            }
            default :
            {
                logError("Unknown message type received: %s", msg_in.toString().c_str());
                respstr = "unknown message type!";
            }
        }
        m_msgout.type = FloorBase::AlgoUserCommandResp;
        m_msgout.ref = msg_in.ref;
        m_msgout.copyString(respstr);
        m_channel->update(m_msgout);
    }

    void AlgoThread::addAlgo(const std::string& name, const std::string& class_name, const std::string& cfg) {
        std::shared_ptr<AlgoBase> algp;
        
        // create algo based on object name
        if (class_name == "AR1") {
            algp = std::make_shared<AR1>(name, cfg, m_floor.m_channel, TimerType::cur_micro());
        } else if (class_name == "IDBO_TF") {
            algp = std::make_shared<IDBO_TF>(name, cfg, m_floor.m_channel, TimerType::cur_micro());
        } else {
            logError("Unknown object name (%s) when createing strategy %s", class_name.c_str(), name.c_str());
        }
        m_algo_map.emplace(name, algp);
    }

    void AlgoThread::setSubscriptions() {
        std::set<int> type_set;
        type_set.insert((int)FloorBase::AlgoUserCommand);
        subscribeMsgType(type_set);
    }

    std::string AlgoThread::toString() const {
        std::string ret = "Loaded=[ ";
        for (const auto& m : m_algo_map) {
            ret += "(";
            ret += m.second->toString(false);
            ret += ") ";
        }
        ret += "]";
        return ret;
    }

}
