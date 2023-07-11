#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstdio>

#include <sys/types.h>
#include <sys/stat.h>

#include "plcc/PLCC.hpp"
#include "csv_util.h"
#include "time_util.h"
#include "symbol_map.h"
#include "PositionManager.h"
#include "RiskMonitor.h"

#define EoDPositionCSVFile "eod_pos"
#define FillCSVFile "fills.csv"

namespace pm {
    PositionManager::PositionManager(const std::string& name,
            const std::string& recover_path) :
    m_name(name),
    m_recovery_path(recover_path.size()==0? plcc_getString("RecoveryPath"): recover_path),
    m_load_second(loadEoD()),
    m_last_micro(0)
    {}

    std::string PositionManager::getLoadUtc() const {
        // from YYYYMMDD-HH:MM:SS local
        // see persist()
        return m_load_second;
    }

    std::string PositionManager::daily_eod_csv() const {
        return m_recovery_path + "/" + EoDPositionCSVFile + "_" + utils::TimeUtil::curTradingDay1() + ".csv";
    }

    std::string PositionManager::eod_csv() const {
        return m_recovery_path + "/" + EoDPositionCSVFile + ".csv";
    }

    std::string PositionManager::daily_eod_csv_mtm() const {
        return m_recovery_path + "/" + EoDPositionCSVFile + "_" + utils::TimeUtil::curTradingDay1() + "_mtm.csv";
    }

    std::string PositionManager::eod_csv_mtm() const {
        return m_recovery_path + "/" + EoDPositionCSVFile + "_mtm.csv";
    }

    std::string PositionManager::fill_csv() const {
        // getting current (snap to previous) trading day 
        const auto utc_second  { (time_t) (utils::TimeUtil::cur_micro() / 1000000ULL) };
        const auto trading_day { utils::TimeUtil::tradingDay(utc_second, -6,0,17,0,0,1) };
        return m_recovery_path + "/" + trading_day+"_"+FillCSVFile;
    }

    const utils::CSVUtil::FileTokens PositionManager::loadEoD_CSVLines(const std::string& eod_file, std::string* latest_day_ptr) const {
        // return the latest reconciled time string (in local time)
        // i.e. the latest time in the eod position file
        auto line_vec = utils::CSVUtil::read_file(eod_file);
        size_t line_cnt = line_vec.size();
        std::string latest_day;
        utils::CSVUtil::FileTokens ret_lines;
        if (line_cnt==0) {
            logInfo("EoD file empty: %s", eod_csv().c_str());

            // get the st_mtime of the eod file instead
            struct stat fileInfo;
            if (stat(eod_csv().c_str(), &fileInfo) != 0) {
                perror("failed to get file info of eod file, creating new!");
                if (latest_day_ptr) {
                    *latest_day_ptr = utils::TimeUtil::frac_UTC_to_string(0,0);
                }
                return ret_lines;
            } else {
                latest_day = utils::TimeUtil::frac_UTC_to_string(fileInfo.st_mtime,0);
                logInfo("getting modify time as last eod: %s", latest_day.c_str());
            }
        } else {
            latest_day = line_vec[line_cnt-1][0];
            // this should be YYYYMMDD,HH:MM:SS (local time) the time when 
            // positions are reconciled and persisted.
            // Note - all position entries for each persist
            // use the same time stamp 
            for (auto& token_vec: line_vec) {
                if (token_vec[0] != latest_day) {
                    continue;
                }
                // remove the first column - the timestamp
                token_vec.erase(token_vec.begin());
                ret_lines.push_back(token_vec);
            }
        }
        if (latest_day_ptr) {
            *latest_day_ptr = latest_day;
        }
        return ret_lines;
    }

    std::string PositionManager::loadEoD() {
        std::string latest_day;
        const auto& line_vec(loadEoD_CSVLines(eod_csv(), &latest_day));
        for (auto& token_vec: line_vec) {
            try {
                auto idp = std::make_shared<IntraDayPosition>(token_vec);
                if ( (!idp->hasPosition()) && (!idp->listOO().size())) {
                    logInfo("loadEoD(): Empty idp not loaded: %s", idp->toString().c_str());
                    continue;
                }
                auto symbol = idp->get_symbol();
                auto algo = idp->get_algo();
                // add idp to the AlgoPosition
                if ( (m_algo_pos.find(algo)!=m_algo_pos.end()) &&
                     (m_algo_pos[algo].find(symbol)!=m_algo_pos[algo].end()) ) {
                    // we find a duplicated line, failed
                    throw std::runtime_error(std::string("PM load failed: duplicated position found: ") + idp->toString());
                }
                m_algo_pos[algo][symbol]=idp;
                m_symbol_pos[symbol][algo]=idp;
            } catch (const std::exception& e) {
                logError("Failed to read position: %s, please check position!", e.what());
            }
        };
        return latest_day;
    };

    bool PositionManager::update(const ExecutionReport& er, bool persist_fill, bool update_risk, bool do_notify_pause) {
        // check for duplicate fill
        // this could happen if the recovery overlaps with real-time
        // fills, or otherwise a previous fill is resent
        bool is_newfill = false;
        // make sure we update risk before update pm
        if (update_risk) {
            bool ret = risk::Monitor::get().updateER(er, *this, do_notify_pause);
            if (__builtin_expect(!ret, 0)) {
                logError("PM update er resulted in risk violation");
            }
        }

        if (er.isFill()) {
            is_newfill = true;
            if (updateThisFill(er)) {
                logInfo("update(): Warning! duplicated fill not updated: %s", er.toString().c_str());
                return false;
            }
        }
        if (__builtin_expect((er.m_symbol[0] == 0) || (er.m_algo[0] == 0), 0)) {
            // 35=9 does not have symbol (empty string)
            logInfo("PM received ER with empty error symbol/algo, not processed: %s", er.toString().c_str());
            return false;
        }

        // update the intra-day position
        auto idp = m_algo_pos[er.m_algo][er.m_symbol];
        if (!idp) {
            idp = std::make_shared<IntraDayPosition>(er.m_symbol, er.m_algo);
            m_algo_pos[er.m_algo][er.m_symbol] = idp;
            m_symbol_pos[er.m_symbol][er.m_algo] = idp;
        }
        idp->update(er);
        m_last_micro = er.m_recv_micro;

        // if it's a fill, need to persist
        if (persist_fill && er.isFill()) {
            utils::CSVUtil::write_line_to_file(er.ExecutionReport::toFillsCSVLine(), fill_csv(), true);
        }
        // update the clOrdId to oo map
        if (__builtin_expect(!isMLegUnderlyingUpdate(er),1)) {
            // since the mleg underlying updates share the same clOrdId with
            // the mleg itself, oo_map not updated with those (synthetic) updates
            const auto& oo(idp->findOO(er.m_clOrdId));
            if (oo) {
                m_oo_map[er.m_clOrdId] = oo;
            } else {
                m_oo_map.erase(er.m_clOrdId);
            }
        }
        return is_newfill;
    }

    bool PositionManager::isMLegUnderlyingUpdate(const ExecutionReport& er) const {
        const auto& oo_iter(m_oo_map.find(er.m_clOrdId));
        return (oo_iter != m_oo_map.end()) &&
                oo_iter->second->m_idp->is_mleg() &&
               (!utils::SymbolMapReader::get().isMLegSymbol(er.m_symbol));
    }

    bool PositionManager::reconcile(const std::string& recovery_file, std::string& diff_logs, bool adjust) {
        // this loads the latest positions from eod_position file
        // update with the execution reports from the given recovery_file
        // recovery_file, typically requested by replay for a time period,
        // and compare the result with this position
        // return true if match, otherwise, mismatches noted in diff_logs
        // If adjust is set to true, this position becomes the recovery position.
        PositionManager pm ("reconcile", m_recovery_path);
        // remove the fill_csv() file
        std::remove(fill_csv().c_str());

        pm.loadRecovery(recovery_file, true, false);
        auto diff_log1 = diff(pm);
        auto diff_log2 =  pm.diff(*this);

        if ((diff_log1.size()) == 0 && (diff_log2.size()) == 0) {
            return true;
        }
        diff_logs = diff_log1 + "\n" + diff_log2;
        if (adjust) {
            logInfo("Warning, %s is going to copy from %s, all existing open order trackings are lost. Download open order again if needed!", m_name.c_str(), pm.m_name.c_str());
            *this=pm;
        }
        return false;
    }

    bool PositionManager::loadRecovery(const std::string& recovery_file, bool persist_fill, bool update_risk) {
        const std::string fname = m_recovery_path + "/" + recovery_file;
        try {
            const bool do_notify_pause = false;
            for(auto& line : utils::CSVUtil::read_file(fname)) {
                update(pm::ExecutionReport::fromCSVLine(line), persist_fill, update_risk, do_notify_pause);
            }
        } catch (const std::exception& e) {
            logError("Failed to load recover.  fname: %s, error: %s\npm:%s", 
                    fname.c_str(), e.what(), toString().c_str());
            return false;
        }
        return true;
    }

    void PositionManager::resetPnl() {
        for (auto iter = m_algo_pos.begin();
             iter != m_algo_pos.end();
             ++iter) {
            for (auto iter2 = iter->second.begin();
                     iter2 != iter->second.end();
                     ++iter2) {
                iter2->second->resetPnl();
            }
        }
    }

    bool PositionManager::persist() const {
        // write each algo's intra-day position csv line
        // the time string in local time
        std::string eod_timestamp = utils::TimeUtil::frac_UTC_to_string(0,0);
        utils::CSVUtil::FileTokens line_vec;  // this includes only realized pnl
        utils::CSVUtil::FileTokens line_vec_mtm;  // this includes both realized pnl and mtm pnl since vap
        utils::CSVUtil::FileTokens line_vec_mtm_daily;  // this includes only mtm pnl since sod today

        // get previous day's mtm lines
        std::string latest_day;
        const auto& line_vec_mtm_prev(loadEoD_CSVLines(eod_csv_mtm(), &latest_day));
        logInfo("persist(): previous mtm pnl file loaded %d lines on %s", (int)line_vec_mtm_prev.size(), latest_day.c_str());

        for (auto iter = m_algo_pos.begin();
             iter != m_algo_pos.end();
             ++iter) {
            for (auto iter2 = iter->second.begin();
                     iter2 != iter->second.end();
                     ++iter2) {
                // realized pnl file (original)
                auto token_vec = iter2->second->toCSVLine();
                token_vec.insert(token_vec.begin(), eod_timestamp);
                line_vec.push_back(token_vec);
                // mtm file
                auto token_vec_mtm = iter2->second->toCSVLineMtm();
                token_vec_mtm.insert(token_vec_mtm.begin(), eod_timestamp);
                line_vec_mtm.push_back(token_vec_mtm);
                // mtm daily
                auto token_vec_mtm_daily = iter2->second->toCSVLineMtmDaily(line_vec_mtm_prev);
                token_vec_mtm_daily.insert(token_vec_mtm_daily.begin(), eod_timestamp);
                line_vec_mtm_daily.push_back(token_vec_mtm_daily);
            }
        };
        // try write a daily position file without append
        if (!utils::CSVUtil::write_file(line_vec, daily_eod_csv(), false)) {
            logError("Failed to write daily position file %s", daily_eod_csv().c_str());
        }
        /* daily_eod_csv_mtm file be written separately as part of end of day operation
         * that deals with settlement prices and exchange rates
        if (!utils::CSVUtil::write_file(line_vec_mtm_daily, daily_eod_csv_mtm(), false)) {
            logError("Failed to write mtm daily file %s", daily_eod_csv_mtm().c_str());
        }
        */
        if (!utils::CSVUtil::write_file(line_vec_mtm, eod_csv_mtm())) {
            logError("Failed to append mtm position file %s", eod_csv_mtm().c_str());
        }
        return utils::CSVUtil::write_file(line_vec, eod_csv());
    }

    bool PositionManager::operator==(const PositionManager& pm) const {
        return (diff(pm)+pm.diff(*this)).size()==0;
    }

    std::string PositionManager::diff(const PositionManager& pm) const {
        std::string diff_str;
        if (pm.m_load_second != m_load_second) {
            // print a warning instead of failing it
            std::string warn_str = "load utc: " + m_load_second + "(" + m_name+") != " + pm.m_load_second + "("+pm.m_name+")";
            logInfo("Warning: %s\n", warn_str.c_str());
        }
        for (auto iter = m_algo_pos.begin();
             iter != m_algo_pos.end();
             ++iter) {
            for (auto iter2 = iter->second.begin();
                 iter2 != iter->second.end();
                 ++iter2) {

                const auto idpv = pm.listPosition(&iter->first, &iter2->first);
                if (idpv.size()==1) {
                    diff_str+=iter2->second->diff(*idpv[0]);
                } else {
                    // is this a zero position?
                    if (iter2->second->hasPosition()) {
                        char buf[256];
                        snprintf(buf, sizeof(buf), "%s miss %s\n", pm.getName().c_str(), iter2->second->toString().c_str());
                        diff_str+=std::string(buf);
                    }
                }
            }
        }
        return diff_str;
    }

    int64_t PositionManager::getPosition(const std::string& algo, const std::string& symbol, double* ptr_vap, double* pnl, int64_t* oqty, int64_t* out_qty) const {
        if (algo.size()==0) {
            return getPosition(symbol, ptr_vap, pnl, oqty);
        }

        int64_t qty = 0;
        if (oqty) {
            *oqty = 0;
        }
        if (out_qty) {
            *out_qty = 0;
        }
        const auto& pos = listOutInPosition(algo, symbol);
        if (pos.size()>0) {
            const IntraDayPosition* idp = pos[0].get();
            IntraDayPosition pd;
            // include N0 if it's a N1 contract
            if (pos.size() > 1) {
                logDebug("out-position detected for %s: [%s]",
                         symbol.c_str(), 
                         pos[0]->toString().c_str());
                if (out_qty) {
                    *out_qty = pos[0]->getPosition();
                }
                pd = *idp;
                pd += (*pos[1]);
                idp = &pd;
            }

            qty = idp->getPosition(ptr_vap, pnl);
            if (oqty) {
                *oqty = idp->getOpenQty();
            }
        }
        return qty;
    }

    int64_t PositionManager::getPosition(const std::string& symbol, double* ptr_vap, double* pnl, int64_t* oqty) const {
        int64_t qty = 0;
        if (oqty) {
            *oqty = 0;
        }
        // get an aggregated position for the given symbol
        // intended usage is for risk
        // note this doesn't include out-contract position
        // see also the getPosition() above
        const auto pos = listPosition(nullptr, &symbol);
        if(pos.size()>0) {
            IntraDayPosition idp(*pos[0]);
            for (size_t i = 1; i<pos.size(); ++i) {
                idp += (*pos[i]);
            }
            qty = idp.getPosition(ptr_vap, pnl);
            if (oqty) {
                *oqty = idp.getOpenQty();
            }
        }
        return qty;
    }

    int64_t PositionManager::getPosition_Market(const std::string* algo_p, const std::string* mkt_p, double* mtm_pnl_p) const {
        std::vector<std::string> algos;
        if (algo_p) {
            if (__builtin_expect( m_algo_pos.find(*algo_p) == m_algo_pos.end(), 0)) {
                logInfo("PositionManager: no position found for algo %s", (*algo_p).c_str());
                if (mtm_pnl_p) *mtm_pnl_p=0;
                return 0;
            }
            algos.push_back(*algo_p);
        } else {
            for (const auto& al: m_algo_pos) {
                algos.push_back(al.first);
            }
        }
        int64_t qty = 0;
        double mtm_pnl = 0;
        for (const auto& algo: algos) {
            std::vector<std::string> symbols;
            for (const auto& sym : m_algo_pos.find(algo)->second) {
                const auto& tradable (sym.first);
                if ((!mkt_p) || (*mkt_p == utils::SymbolMapReader::get().getTradableMkt(tradable))) {
                    const auto& idp(sym.second);
                    qty += idp->getPosition();
                    qty += idp->getOpenQty();
                    if (mtm_pnl_p) {
                        mtm_pnl += idp->getMtmPnl();
                    }
                }
            }
        }
        if (mtm_pnl_p) {
            *mtm_pnl_p = mtm_pnl;
        }
        return qty;
    }

    std::vector<std::shared_ptr<const IntraDayPosition> > PositionManager::listPosition(const std::string* algo, const std::string* symbol) const {
        std::vector<std::shared_ptr<const IntraDayPosition> > vec;
        if (algo && (*algo).size()) {
            // use the algo map
            const auto iter = m_algo_pos.find(*algo);
            if (iter != m_algo_pos.end()) {
                if (symbol && (*symbol).size()) {
                    // we have a tradable symbol 
                    const std::string& tradable_symbol = utils::SymbolMapReader::get().getTradableSymbol(*symbol);
                    const auto& iter2 = iter->second.find(tradable_symbol);
                    if (iter2 != iter->second.end()) {
                        vec.push_back(iter2->second);
                    }
                } else {
                    addAllMapVal(vec, iter->second.begin(), iter->second.end());
                }
            }
        } else {
            // use the symbol map
            if (symbol && (*symbol).size()) {
                const std::string& tradable_symbol = utils::SymbolMapReader::get().getTradableSymbol(*symbol);
                const auto& iter = m_symbol_pos.find(tradable_symbol);
                if (iter != m_symbol_pos.end()) {
                    addAllMapVal(vec, iter->second.begin(), iter->second.end());
                }
            } else {
                // add all 
                for (auto iter = m_symbol_pos.begin(); iter!= m_symbol_pos.end(); ++iter) {
                    addAllMapVal(vec, iter->second.begin(), iter->second.end());
                }
            }
        }
        return vec;
    }

    std::vector<std::shared_ptr<const IntraDayPosition> > PositionManager::listOutInPosition(const std::string& algo, const std::string& symbol) const {
        auto vec = listPosition(&algo, &symbol);
        // if size more than one, then the vector has [out, in], otherwise, it has [in]
        // check N0 contract if this is a N1
        try {
            const std::string& tradable_symbol = utils::SymbolMapReader::get().getTradableSymbol(symbol);
            const auto* ti (utils::SymbolMapReader::get().getN0ByN1(tradable_symbol));
            if (ti) {
                // include N0 if exists to the beginning
                const auto& vec_out = listPosition(&algo, &ti->_tradable);
                if (vec_out.size()>0) {
                    const auto& idp (vec_out[0]);
                    if ( idp->hasPosition() || idp->listOO().size()) {
                        logInfo("listOutInPosition() got out-contract %s (%d out positions, %d in positions)", 
                            ti->toString().c_str(), (int)vec_out.size(), (int)vec.size());
                        vec.insert(vec.begin(), vec_out.begin(), vec_out.end());
                    }
                }
            }
        } catch (const std::exception& e) {
            logError("problem checking out contract for %s-%s: %s", algo.c_str(), symbol.c_str(), e.what());
        }
        return vec;
    }

    std::vector<std::shared_ptr<const OpenOrder> > PositionManager::listOO(const std::string* algo, const std::string* symbol, const bool* ptr_side) const {
        std::vector<std::shared_ptr<const OpenOrder> > vec_oo;
        const auto vec = listPosition(algo, symbol);
        for (const auto& idp: vec) {
            const auto oov = idp->listOO();
            for (const auto& oo:oov) {
                if (oo->m_open_qty==0) {
                    continue;
                }
                if (ptr_side) {
                    if ( ((*ptr_side) && (oo->m_open_qty < 0)) ||
                         (!(*ptr_side) && (oo->m_open_qty > 0))) 
                    {
                        continue;
                    }
                }
                vec_oo.push_back(oo);
            }
        }
        return vec_oo;
    }

    std::vector<std::shared_ptr<const OpenOrder> > PositionManager::listOO_FIFO(const std::string* algo, const std::string* symbol, const bool* ptr_side) const {
        const auto& vec_oo = listOO(algo, symbol, ptr_side);
        if (vec_oo.size() <= 1) {
            return vec_oo;
        }

        logDebug("got listOO_FIFO vec_oo.size (%d)", (int)vec_oo.size());
        std::map<int64_t, std::shared_ptr<const OpenOrder>> oo_map_;
        for (const auto& oo: vec_oo) {
            logDebug("Open Orders (%s)", oo->toString().c_str());
            oo_map_[-(int64_t)oo->m_open_micro]=oo;
        }

        std::vector<std::shared_ptr<const OpenOrder>> vec_oo_sorted;
        for (auto iter = oo_map_.begin(); iter != oo_map_.end(); ++iter) {
            logDebug("sorted oo: %s", iter->second->toString().c_str());
            vec_oo_sorted.push_back(iter->second);
        }
        return vec_oo_sorted;
    }

    void PositionManager::deleteOO(const char* clOrdId) {
        if (clOrdId && *clOrdId) {
            auto iter = m_oo_map.find(clOrdId);
            if (iter!=m_oo_map.end()) {
                const std::string& algo(iter->second->m_idp->get_algo());
                const std::string& symbol(iter->second->m_idp->get_symbol());
                m_algo_pos[algo][symbol]->deleteOO(clOrdId);
                m_oo_map.erase(iter);
            }
        } else {
            // delete all open orders
            for (auto iter = m_algo_pos.begin();
                 iter != m_algo_pos.end();
                 ++iter) {
                for (auto iter2 = iter->second.begin();
                     iter2 != iter->second.end();
                     ++iter2) {
                    iter2->second->deleteOO(nullptr);
                }
            }
            m_oo_map.clear();
        }
    }

    double PositionManager::getPnl(const std::string* algo, 
                  const std::string* symbol) const {
        const auto vec = listPosition(algo, symbol);
        double pnl=0;
        for (const auto& idp: vec) {
            double pnl0;
            idp->getPosition(nullptr, &pnl0);
            pnl+=pnl0;
        };
        return pnl;
    }

    void PositionManager::operator=(const PositionManager& pm) {
        logInfo("%s gets all positions from %s", m_name.c_str(), pm.m_name.c_str());
        m_algo_pos = pm.m_algo_pos;
        m_symbol_pos = pm.m_symbol_pos;
        m_last_micro = pm.m_last_micro;
        m_fill_execid = pm.m_fill_execid;
    }

    std::vector<FloorBase::PositionServerDetails> PositionManager::getPositionServerDetails() const
    {
        IntraDayPosition total_intra_day_positions{};

        std::vector<FloorBase::PositionServerDetails> positions_vec{};

        const auto& positions(listPosition(nullptr, nullptr));

        double total_pnl{};
        double total_pnl_mtm{};
        int64_t total_quantity{};

        for (const auto& position : positions) {

            double position_pnl{};
            double position_mtm{};
            int64_t position_quantity{};

            position_quantity = position->getPosition(nullptr, &position_pnl);
            total_quantity += position_quantity;
            total_pnl += position_pnl;

            position_mtm = position->getMtmPnl();
            total_pnl_mtm += position_mtm;
            total_intra_day_positions += *position;

            // Add to our copied vector (Only add aggregate at the end!!)
            positions_vec.emplace_back(FloorBase::PositionServerDetails{
                .market = "ALL",
                .symbol = position->get_symbol(),
                .algo = position->get_algo(),
                .value = position->getVAP(),
                .quantity = position_quantity,
                .order_id = 0ul,
                .version = 0u,
                .profit_and_loss = position_pnl,
                .mark_to_market = position_mtm
                });
        }

        // Needs fixing: have to add total aggregate to each position for update filtering!!
        for (auto& position : positions_vec) {
            position.total_profit_and_loss = total_pnl;
            position.total_mark_to_market = total_pnl_mtm;
            position.has_aggregate = true;
        }

        return positions_vec;
    }

    std::string PositionManager::toString(const std::string* ptr_algo, const std::string* ptr_symbol, bool summary) const {
        std::string ret;
        const auto idp_vec(listPosition(ptr_algo, ptr_symbol));
        if (idp_vec.size() == 0) {
            return "Nothing Found.";
        }
        IntraDayPosition idp0;
        double pnl = 0, pnl_mtm=0;
        int64_t qty = 0;

        char buf[256];
        snprintf(buf, sizeof(buf), 
                "%-13s  %-16s  %-5s   %-9s    %-20s  last_updated"
                "\n---------------------------------"
                  "---------------------------------"
                  "---------------------------------\n",
                "strategy", "symbol", "qty", "vap", "mtm_pnl  (realized)");
        ret += std::string(buf);
        for (const auto& idp:idp_vec) {
            ret += (idp->toString() + "\n");
            double pnl0;
            qty += idp->getPosition(nullptr, &pnl0);
            pnl += pnl0;
            pnl_mtm += idp->getMtmPnl();
            idp0 += *idp;
        }
        if (summary) {
            double vwap = 0;
            idp0.getPosition(&vwap, nullptr);
            //ret += ("***Total qty: " + std::to_string(qty) + " avg_px: " + PriceString(vwap) + " pnl: " +  PnlString(pnl_mtm) + " ( realized: " + PnlString(pnl)+" )");
            ret += ("***\nPnL: " +  PnlStringColor(pnl_mtm) + " ( realized: " + PnlStringColor(pnl)+" )");
        }

        std::string retoo;
        for (const auto& idp:idp_vec) {
            if (idp->listOO().size() > 0) {
                retoo += ("\n"+idp->dumpOpenOrder());
            }
        }
        if (retoo.size()) {
            ret += "\n***Open Orders:";
            ret += retoo;
        };

        return ret;
    }

    bool PositionManager::updateThisFill(const ExecutionReport& er) {
        // assuming this er is a fill, check to see if we have seen this er
        const std::string key = std::string(er.m_execId) + std::string(er.m_clOrdId);
        if (m_fill_execid.find(key) != m_fill_execid.end()) {
            return true;
        }
        m_fill_execid.insert(key);
        return false;
    }

    bool PositionManager::haveThisFill(const ExecutionReport& er) const {
        const std::string key = std::string(er.m_execId) + std::string(er.m_clOrdId);
        return (m_fill_execid.find(key) != m_fill_execid.end());
    }

    std::shared_ptr<const OpenOrder> PositionManager::getOO(const std::string& clOrdId) const {
        const auto iter = m_oo_map.find(clOrdId);
        if (iter != m_oo_map.end()) {
            return iter->second;
        }
        return std::shared_ptr<const OpenOrder>();
    }

    std::vector<std::pair<std::shared_ptr<const OpenOrder>, int64_t> > PositionManager::matchOO(const std::string& symbol, int64_t qty, const double* px, const std::string* algo) const {
        bool isBuy = (qty<0);
        const auto& oolist = listOO_FIFO(nullptr, &symbol, &isBuy);
        std::vector<std::pair<std::shared_ptr<const OpenOrder>, int64_t> > vec;
        for (const auto& oo : oolist) {
            if (qty == 0) {
                break;
            }
            double px_diff = 0;
            // allow matching for same algo
            if (algo && (*algo == oo->m_idp->get_algo())) {
                // match all same algo open orders
                // we should only have one sided open order for each algo
                px_diff = 0;
            } else {
                if (px) {
                    px_diff = oo->m_open_px - *px;
                }
            }
            px_diff = isBuy? px_diff:-px_diff;
            if (px_diff > -1e-10) {
                // match this oo
                int64_t qty0 = oo->m_open_qty;
                if (qty0 == 0) {
                    continue;
                }
                if ((qty + qty0)*qty < 0) {
                    // match upto qty
                    qty0 = -qty;
                }
                vec.push_back(std::make_pair(oo, qty0));
                qty += qty0;
            }
        }
        return vec;
    }
}

