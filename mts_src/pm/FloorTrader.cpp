#include "FloorTrader.h"
#include "plcc/ConfigureReader.hpp"
#include "instance_limiter.h"
#include "symbol_map.h"
#include <stdexcept>

// execution trades
#include "TWAPTrader.h"

namespace pm {
    FloorTrader::FloorTrader(const std::string& instance_name)
    : FloorCPR<FloorTrader>(instance_name) {
        // config file expected as config/floortrader_instance_name.cfg
        const auto cfg(utils::ConfigureReader((std::string("config/")+instance_name + ".cfg").c_str()));

        const auto trader_names (cfg.listKeys());
        if (trader_names.size() == 0) {
            logError("no traders defined for %s", instance_name.c_str());
            throw std::runtime_error(std::string("Floor Trader found no traders defined for ") + instance_name);
        }

        // fill in the m_symbol_set and populate the m_et
        m_et.resize((size_t)PositionInstruction::TOTAL_TYPES);
        m_symbol_set.resize((size_t)PositionInstruction::TOTAL_TYPES);
        const auto& all_ti(utils::SymbolMapReader::get().listAllTradable());
        std::set<std::string> all_mkt;  // no _N, WTI, Brent, etc
        for (const auto& ti: all_ti) {
            all_mkt.insert(ti->_symbol);
        }
        for (const auto& tn : trader_names) {
            // this could throw
            // get a type number from trader string from config
            int tp = (int) PositionInstruction::typeFromString(tn);
            if (m_et[tp]) {
                logError("Duplicate trader type %d (%s)", tp, tn.c_str());
                throw std::runtime_error(std::string("Duplicate trader type ") + tn);
            }
            const auto& trader_cfg (cfg.get<std::string>((tn + ".config").c_str()));
            m_et[tp] = createTrader(tp, trader_cfg); // this would throw
            const auto& symbol_arr(cfg.getArr<std::string>((tn + ".symbol").c_str()));
            for (const auto& symbol: symbol_arr) {
                // parse the symbol string, must have "_" 
                if (symbol.find("-") != std::string::npos) {
                    logError("trader %s cannot handle spread symbols :%s", tn.c_str(), symbol.c_str());
                    throw std::runtime_error(tn+std::string(" spread symbol cannot be handled" + symbol));
                }
                // expand string
                auto np = symbol.find("_N");
                if (np == std::string::npos) {
                    logError("symbol %s does not have _N", symbol.c_str());
                    throw std::runtime_error(tn+std::string(" no _N: " + symbol));
                }
                auto mkt = symbol.substr(0,np);
                auto nstr = symbol.substr(np);
                std::set<std::string> type_mkt;
                if (mkt != "ALL") {
                    if (all_mkt.find(mkt) == all_mkt.end()) {
                        logError("symbol %s not found in symbol map", mkt.c_str());
                        throw std::runtime_error(std::string("symbol not found in symbol map " ) + mkt);
                    }
                    type_mkt.insert(mkt);
                } else {
                    type_mkt = all_mkt;
                }
                for (const auto& m: type_mkt) {
                    const auto sym (m + nstr);
                    if (!m_symbol_set[tp].insert(sym).second) {
                        logError("trader %s has duplicate symbol %s", \
                                tn.c_str(), sym.c_str());
                        throw std::runtime_error(std::string("duplicate symbol for " + tn + " " + sym));
                    }
                }
            }
            logInfo("type %s loaded with %d symbols", tn.c_str(), (int) (m_symbol_set[tp].size()));
        }
    }

    FloorTrader::~FloorTrader() {};
    void FloorTrader::shutdown_derived() { 
        utils::OnlyMe::get().remove_all();
    };  // destroy ETs

    void FloorTrader::start_derived() {
        // m_onlyme is 'type::symbol' instance checker
        // follows through the m_symbol_set

        // m_symbol_set a vector, indexed by type
        utils::OnlyMe::get().remove_all();
        for (int tp=0; tp<(int)m_symbol_set.size(); ++tp) {
            const auto& sym_set(m_symbol_set[tp]);
            const std::string& tp_str(PositionInstruction::TypeString(tp));
            for (const auto& sym : sym_set) {
                const auto tp_sym = tp_str + "::" + sym;
                utils::OnlyMe::get().add_name(tp_sym);
            }
        }

        // do a check upon start
        checkOnlyMe();
        utils::TimeUtil::micro_sleep(1000*1000*1.5);
        checkOnlyMe();
        m_last_onlyme_second = utils::TimeUtil::cur_utc();
    }

    std::string FloorTrader::toString_derived() const {
        // dump the m_eiMap
        std::string resp = m_name + "States:\n";
        for (const auto& kv:m_eiMap){
            const auto& k(kv.first);
            const auto& ei(kv.second);
            resp += (std::string("[")+k+"-"+ (ei?ei->toString():std::string("NONE"))+"]\n");
        }
        resp += "Trader-Symbol:\n";

        // TODO dump the m_symbol_set
        for (int i=0; i<(int)m_symbol_set.size(); ++i) {
            const auto& sym_set (m_symbol_set[i]);
            if (sym_set.size()==0) continue;
            const std::string& tp_str(PositionInstruction::TypeString(i));
            resp += (tp_str + "(" + std::to_string(sym_set.size()) + ")[ ");
            for (const auto& sym : sym_set) {
                resp += (sym + " ");
            }
            resp += "]\n";
        }
        return resp;
    }

    void FloorTrader::addPositionSubscriptions_derived() {
        std::set<int> type_set;
        type_set.insert((int)FloorBase::SetPositionReq);
        subscribeMsgType(type_set);
    }

    void FloorTrader::run_loop_derived() {
        /*scan through m_eiMap and run bool onRun(etinfo)
          if all false, then check onlyme and sleep IdleSleepMicro
         */
        uint64_t cur_micro = utils::TimeUtil::cur_micro();
        // run everything that should be run
        for (auto& kv:m_eiMap) {
            auto& etinfo (kv.second);
            if (etinfo->_done) {
                continue;
            }
            // get the next run time
            if (etinfo->_next_micro<=cur_micro) {
                int tp_ = etinfo->_pi->type;
                m_et[tp_]->onRun(etinfo);
            }
        }

        // check  onlyme if needed
        time_t cur_second = utils::TimeUtil::cur_utc();
        if (__builtin_expect(cur_second > m_last_onlyme_second,0)) {
            m_last_onlyme_second = cur_second;
            checkOnlyMe();
        }

        // figure out this sleep time
        cur_micro = utils::TimeUtil::cur_micro();
        uint64_t min_next_micro = cur_micro + IdleSleepMicro;
        for (auto& kv:m_eiMap) {
            auto& etinfo (kv.second);
            if (etinfo->_done) {
                continue;
            }
            auto nm = etinfo->_next_micro;
            if (nm < cur_micro - 1000000) {
                logError("State(%s) out of _next_micro update for one second, put it to run next cycle", etinfo->toString().c_str());
                etinfo->_next_micro = cur_micro + IdleSleepMicro;
                continue;
            }
            // get the minimum next run time
            min_next_micro = _MIN_(nm,min_next_micro);
        }
        cur_micro = utils::TimeUtil::cur_micro();
        if (min_next_micro > cur_micro + MinSleepMicro) {
            // 2 micro is a safe upper bound for a context switch
            utils::TimeUtil::micro_sleep(min_next_micro-cur_micro-2);
        }
        // otherwise, just spin
    }

    bool FloorTrader::handlePositionReq_derived(const MsgType& msg, MsgType& msg_out) {
        /*
           gets the "set" request, check m_eiMap, the algo+symbol map, 
           if etinfo  exists:
               if not same type - remove that etinfo
               otherwise, 
                   run bool ET.onUpdate(pi, et_info)
           if not exists:
             check if this FT covers it
           if yes, then run ET.onEntry(pi)
        */

        // get PI from msg
        if(msg.type != FloorBase::SetPositionReq) {
            return false;
        }

        // expect a PositionInstruction
        // queue a instruction and ack the request
        m_msgout.ref = msg.ref;
        m_msgout.type = FloorBase::SetPositionAck;
        PositionInstruction* pip0 = (PositionInstruction*)msg.buf ; 
        logInfo("SetPositionReq: %s, %s, %d, %f", pip0->algo, pip0->symbol, (int)pip0->qty, (double)pip0->px);
        auto pip = std::make_shared<PositionInstruction>(pip0->algo, pip0->symbol, pip0->qty, pip0->px,pip0->target_utc, (PositionInstruction::TYPE) pip0->type);
        addPositionInstruction(pip);
        m_msgout.copyString("Ack");
        return true;
    }

    void FloorTrader::addPositionInstruction(const std::shared_ptr<PositionInstruction> & pi) {
        const auto&key (getKey(pi->algo, pi->symbol));
        auto iter = m_eiMap.find(key);
        if (iter != m_eiMap.end()) {
            auto& etinfo (iter->second);
            const auto& old_pi(etinfo->_pi);
            if (old_pi->type == pi->type) {
                m_et[pi->type]->onUpdate(pi, etinfo);
                return;
            }
            // remove that key
            logInfo("trader type switch detected for %s. New: (%s), Existing: (%s), removing existing", m_name.c_str(), pi->toString().c_str(), old_pi->toString().c_str());
            m_eiMap.erase(iter);
        }
        // should I handle it (type + symbol)
        auto trader (m_et[pi->type]);
        if (!trader) {
            // not my business
            return;
        }
        logInfo("trader %s add PI(%s)", m_name.c_str(), pi->toString().c_str());
        m_eiMap[key] = trader->onEntry(pi);
    }

    void FloorTrader::handleExecutionReport_derived(const pm::ExecutionReport& er) {
         /* gets the etinfo from er's algo+symbol, if found, 
            call ET.onER(er, etinfo), which should update 
            etinfo's state such as done
         */
        const auto&key(getKey(er.m_algo, er.m_symbol)); 
        auto iter = m_eiMap.find(key);
        if (iter!=m_eiMap.end()) {
            auto& etinfo(iter->second);
            m_et[etinfo->_pi->type]->onER(er, etinfo);
        }
    }

    bool FloorTrader::handleUserReq_derived(const MsgType& msg, std::string& respstr) {
        // dump FT state
        respstr = "";
        const char* cmd = msg.buf;
        switch (cmd[0]) {
        case 'F': 
        {
            respstr = toString();
            return true;
        }
        default:
            break;
        }
        return false;
    }

    std::string FloorTrader::getKey(const std::string& algo, const std::string symbol) const {
        return algo+"::"+symbol;
    }

    std::shared_ptr<ExecutionTrader> FloorTrader::createTrader(int type, const std::string& trader_cfg_fn) {
        // this needs to create based on type
        switch (type) {
        case PositionInstruction::TWAP:
            // call derived ET constructor and 
            // return base class ET
            std::string name = m_name + "::" + PositionInstruction::TypeString(type);
            return std::make_shared<TWAPTrader>(name, *this, trader_cfg_fn);
        }
        throw std::runtime_error(std::string("unknown type ") + std::to_string(type));
    }

    void FloorTrader::checkOnlyMe() {
        if (! utils::OnlyMe::get().check()) {
            logError("Duplicate instances found for %s, exiting!", m_name.c_str());
            throw std::runtime_error(std::string("OnlyMe failed for ") + m_name);
        }
    }
}

