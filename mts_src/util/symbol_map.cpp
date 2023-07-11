#include "symbol_map.h"
#include "plcc/PLCC.hpp"
#include "thread_utils.h"
#include <set>

#define SymbolMapConfig "SymbolMap"

namespace utils {

    TradableInfo::TradableInfo()
    : _tick_size(0), _point_value(0), _px_multiplier(0), _bbg_px_multiplier(0), _N(0), 
    _expiration_days(0), _lotspermin(10), _subscribed(false) {}

    TradableInfo::TradableInfo(
            const std::string& tradable,
            const std::string& symbol,
            const std::string& exch_symbol,
            const std::string& mts_contract,
            const std::string& mts_symbol,
            const std::string& venue,
            const std::string& type,
            const std::string& contract_month,
            const std::string& tt_security_id,
            const std::string& tt_venue,
            const std::string& bbg_id,
            const std::string& currency,
            const std::string& expiration_date,
            double tick_size,
            double point_value,
            double px_multiplier,
            double bbg_px_multiplier,
            int N,
            int expiration_days,
            int lotspermin,
            bool subscribed)
    : _tradable(tradable), _symbol(symbol), _exch_symbol(exch_symbol),
      _mts_contract(mts_contract), _mts_symbol(mts_symbol),
      _venue(venue), _type(type), _contract_month(contract_month),
      _tt_security_id(tt_security_id), _tt_venue(tt_venue), _bbg_id(bbg_id),
      _currency(currency), _expiration_date(expiration_date), _tick_size(tick_size),
      _point_value(point_value), _px_multiplier(px_multiplier), _bbg_px_multiplier(bbg_px_multiplier), _N(N),
      _expiration_days(expiration_days), _lotspermin(lotspermin), _subscribed(subscribed) {};

    std::string TradableInfo::toString() const {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %.7lf, %.7lf, %.7lf, %.7lf, %d, %d, %d, %d",
                _tradable.c_str(), _symbol.c_str(), _exch_symbol.c_str(), _mts_contract.c_str(), _mts_symbol.c_str(),
                _venue.c_str(), _type.c_str(), _contract_month.c_str(),
                _tt_security_id.c_str(), _tt_venue.c_str(), _bbg_id.c_str(), _currency.c_str(), _expiration_date.c_str(),
                _tick_size, _point_value, _px_multiplier, _bbg_px_multiplier,
                _N, _expiration_days, _lotspermin,
                (int)_subscribed);
        return std::string(buf);
    }

    SymbolMapReader::SymbolMapReader(const std::string& cfg_file)
    : m_cfg_file(((cfg_file == "") ? plcc_getString(SymbolMapConfig) : cfg_file) )
    {
        reload();
    }

    void SymbolMapReader::reload() {
        try {
            const utils::ConfigureReader cfg (m_cfg_file.c_str());
            const auto& tradables = cfg.getReader("tradable");
            clear();
            auto keys = tradables.listKeys();
            for (const auto& k:keys) {
                TradableInfo* ti = new TradableInfo(k, tradables.getReader(k));
                m_tradable[k] = ti;
                m_tradable_by_symbol[ti->_symbol].push_back(ti);
                m_tradable_by_mts[ti->_mts_symbol] = ti;
                m_tradable_by_ttsecid[ti->_tt_security_id] = ti;
                m_tradable_by_mts_contract[ti->_mts_contract] = ti;
                m_tradable_by_bbg_id[ti->_bbg_id] = ti;
            }

            // setup N0/N1 map
            for (const auto& kv : m_tradable) {
                const auto* ti(kv.second);
                if (ti->_N == 1) {
                    // if this is N1, see if N0 exists
                    const auto& tivec(getAllByTradable(kv.first));
                    for (const auto* ti0 : tivec) {
                        if (ti0->_N == 0) {
                            m_n0_by_n1[kv.first]=ti0;
                            break;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            logError("Error reload the symbol map: %s", e.what());
            throw e;
        }
    }

    const SymbolMapReader& SymbolMapReader::get(bool reload) {
        static SymbolMapReader reader;
        if (reload) {
            reader.reload();
        }
        return reader;
    }

    const SymbolMapReader SymbolMapReader::getFile(const std::string& file) {
        return SymbolMapReader(file);
    }

    void SymbolMapReader::clear() {
        for (const auto& t: m_tradable) {
            delete t.second;
        }
        m_tradable.clear();
        m_tradable_by_symbol.clear();
        m_tradable_by_mts.clear();
        m_tradable_by_ttsecid.clear();
    }

    SymbolMapReader::~SymbolMapReader() {
        clear();
    }

    const TradableInfo* SymbolMapReader::getByMtsSymbol(const std::string& mts_symbol, bool return_null) const {
        const auto iter = m_tradable_by_mts.find(mts_symbol);
        if (iter == m_tradable_by_mts.end()) {
            if (return_null) return nullptr;
            logError("No tradable found for MTS symbol: %s!", mts_symbol.c_str());
            throw std::runtime_error("tradable not found for mts symbol!");
        }
        return iter->second;
    }

    const std::vector<const TradableInfo*> SymbolMapReader::getAllBySymbol(const std::string& symbol) const {
        const auto iter = m_tradable_by_symbol.find(symbol);
        if (iter == m_tradable_by_symbol.end()) {
            logError("No tradable found for symbol: %s!", symbol.c_str());
            throw std::runtime_error("tradable not found for mts symbol!");
        }
        return iter->second;
    }

    const TradableInfo* SymbolMapReader::getByTradable(const std::string& tradable_symbol, bool return_null) const {
        const auto iter = m_tradable.find(tradable_symbol);
        if (__builtin_expect(iter == m_tradable.end(),0)) {
            if (return_null) return nullptr;
            logDebug("No tradable found for tradable symbol: %s!", tradable_symbol.c_str());
            throw std::invalid_argument("tradable not found for tradable symbol!");
        }
        return iter->second;
    }

    const TradableInfo* SymbolMapReader::getByTTSecId(const std::string& tt_security_id, bool return_null) const {
        const auto iter = m_tradable_by_ttsecid.find(tt_security_id);
        if (iter == m_tradable_by_ttsecid.end()) {
            if (return_null) return nullptr;
            logError("No tradable found for TT Security ID: %s!", tt_security_id.c_str());
            throw std::runtime_error(std::string("tradable not found for tt security id ") + tt_security_id);
        }
        return iter->second;
    }

    const TradableInfo* SymbolMapReader::getByMtsContract(const std::string& mts_contract, bool return_null) const {
        const auto iter = m_tradable_by_mts_contract.find(mts_contract);
        if (iter == m_tradable_by_mts_contract.end()) {
            if (return_null) return nullptr;
            logError("No tradable found for mts contract: %s!", mts_contract.c_str());
            throw std::runtime_error(std::string("tradable not found for mts contract ") + mts_contract);
        }
        return iter->second;
    }

    const TradableInfo* SymbolMapReader::getByBbgId(const std::string& bbg_id, bool return_null) const {
        const auto iter = m_tradable_by_bbg_id.find(bbg_id);
        if (iter == m_tradable_by_bbg_id.end()) {
            if (return_null) return nullptr;
            logError("No tradable found for bbg id: %s!", bbg_id.c_str());
            throw std::runtime_error(std::string("tradable not found for bbg id ") + bbg_id);
        }
        return iter->second;
    }

    const std::vector<const TradableInfo*> SymbolMapReader::getAllByMtsVenue(const std::vector<std::string>& mts_venue, int max_n) const {
        std::set<std::string> vs (mts_venue.begin(), mts_venue.end());
        std::vector<const TradableInfo*> vec;
        for (const auto& kv:m_tradable) {
            if (vs.find(kv.second->_venue) != vs.end()) {
                if (kv.second->_N <= max_n) {
                    vec.push_back(kv.second);
                }
            }
        }
        return vec;
    }

    // reads the subscription from the main.cfg and
    // 1. find out the primary and backup venue and symbol for the given provider
    // 2. return two lists of tradable symbols, first the primary sub, second the backup sub
    // 3. it throws if the returned primary and backup list has overlap (misconfiguration)
    // 4. it also throws if the primary lists for any providers (defined in main.cfg) overlaps
    // Note, if provider is empty, then it returns all subscribed symbols as primary and empty set as secondary
    const std::pair<std::vector<std::string>, std::vector<std::string>>
        SymbolMapReader::getSubscriptions(const std::string& provider, int max_n) const
    {
        std::set<std::string> primary_list;
        std::pair<std::set<std::string>, std::set<std::string>> sym_pair;
        const utils::ConfigureReader rdr (utils::PLCC::getConfigPath()) ;
        const auto& provider_list (rdr.getReader("MDProviders").listKeys());
        // check if provider in the provider_list, this is to prevent from the "typo"
        if (!provider.empty()) {
            const std::set<std::string> pset (provider_list.begin(), provider_list.end());
            if (pset.find(provider) == pset.end()) {
                logError("provider %s not found in the provider list of %d!", provider.c_str(), (int)provider_list.size());
                throw std::runtime_error("provider " + provider + " not found in the provider list!");
            }
        }
        if (max_n == -1) {
            max_n = rdr.get<int>("MaxN");
        }
        for (const auto& p : provider_list) {
            const auto& sub (SymbolMapReader::getSubscriptionsUnsafe(p, max_n));
            if (p == provider) {
                sym_pair = sub;
            } else if (provider.empty()) {
                // append everything to the primary
                sym_pair.first.insert(sub.first.begin(), sub.first.end());
                sym_pair.first.insert(sub.second.begin(), sub.second.end());
            }
            for (const auto& s:sub.first) {
                if (primary_list.find(s) != primary_list.end()) {
                    logError("%s is primary for more than one provider!", s.c_str());
                    throw std::runtime_error(s + " is primary for more than one provider!");
                }
                primary_list.insert(s);
            }
        }
        const auto& psym(sym_pair.first);
        const auto& bsym(sym_pair.second);
        for (const auto& ps: psym) {
            if (bsym.find(ps) != bsym.end()) {
                // this shouldn't happen because when it is a backup, some provider must be a primary
                // and this contradicts with previous check
                logInfo("provier %s has %s in both primary and backup lists, this shouldn't happen!", provider.c_str(), ps.c_str());
                throw std::runtime_error(provider + " has " + ps + " in both primary and backup list!");
            }
        }
        return std::make_pair<std::vector<std::string>, std::vector<std::string>> (
                std::vector<std::string>(psym.begin(), psym.end()),
                std::vector<std::string>(bsym.begin(), bsym.end())
                );
    }
 
    const std::vector<std::string> SymbolMapReader::getPrimarySubscriptions(int max_n) const {
        return getSubscriptions("", max_n).first;
    }

    const std::pair<std::set<std::string>, std::set<std::string>> SymbolMapReader::getSubscriptionsUnsafe(const std::string& provider, int max_n) const {
        // it gets the primary and secondary subscriptions from configuration 
        // this is unsafe because no checks on duplicate primaries across defined providers
        std::set<std::string> vp, vb, sp, sb;  // venue primary/back, symbol primary/back
        const utils::ConfigureReader rdr(utils::PLCC::getConfigPath());
        const auto& sr (rdr.getReader("MTSSymbol"));
        const auto& sl (sr.listKeys());
        for (const auto& s: sl) {
            const auto& pl (sr.getArr<std::string>(s.c_str()));
            if (pl.empty()) {
                continue;
            }
            if (pl[0] == provider) {
                sp.insert(s);
                continue;
            }
            if (pl.size() == 2) {
                if (pl[1] == provider) {
                    sb.insert(s);
                    continue;
                }
            }
            if (pl.size() > 2) {
                logError("Failed to parse the subscription from main.cfg for symbol %s", s.c_str());
                throw std::runtime_error("Failed to parse the subscription from main.cfg for symbol " + s);
            }
        }

        const auto& vr (rdr.getReader("MTSVenue"));
        const auto& vl (vr.listKeys());
        for (const auto& v: vl) {
            const auto& pl (vr.getArr<std::string>(v.c_str()));
            if (pl.empty()) {
                continue;
            }
            if (pl[0] == provider) {
                vp.insert(v);
                continue;
            }
            if (pl.size() == 2) {
                if (pl[1] == provider) {
                    vb.insert(v);
                    continue;
                }
            }
            if (pl.size() > 2) {
                logError("Failed to parse the subscription from main.cfg for venue %s", v.c_str());
                throw std::runtime_error("Failed to parse the subscription from main.cfg for venue " + v);
            }
        }

        return std::make_pair<std::set<std::string>, std::set<std::string>> (
                getSubSymbols(max_n, vp, sp),
                getSubSymbols(max_n, vb, sb));
    }

    std::set<std::string> SymbolMapReader::getSubSymbols(int max_n, const std::set<std::string>& vas, const std::set<std::string>& sas) const {
        std::set<std::string> ret;
        for (const auto& kv:m_tradable) {
            const auto* t (kv.second);
            if (t->_N>max_n)
                continue;
            if ((vas.find(t->_venue)==vas.end()) && (sas.find(t->_symbol)==sas.end())) {
                continue;
            }
            ret.insert(t->_mts_symbol);
        }
        return ret;
    }

    std::string SymbolMapReader::getTradableSymbol(const std::string& symbol) const {
        const auto* ti = getTradableInfo(symbol);
        if (ti) {
            return ti->_tradable;
        }
        return "";
    }

    std::string SymbolMapReader::getTradableMkt(const std::string& symbol) const {
        const auto* ti = getTradableInfo(symbol);
        if (ti) {
            return ti->_symbol;
        }
        return "";
    }

    bool SymbolMapReader::isMLegSymbol(const std::string& symbol) const {
        const auto* ti = getTradableInfo(symbol);
        if (ti) {
            return ti->_type == "MLEG";
        }
        return false;
    }

    const TradableInfo* SymbolMapReader::getTradableInfo(const std::string& symbol) const {
        static std::unordered_map<std::string, const TradableInfo*> s_ti_map;
        static SpinLock::LockType s_lock(false);
        {
            auto lock = SpinLock(s_lock);
            const auto iter = s_ti_map.find(symbol);
            if (__builtin_expect(iter != s_ti_map.end(), 1)) {
                return iter->second;
            }

            const auto* ti (getByTradable(symbol, true));
            if (__builtin_expect(!ti, 0)) {
                ti = getByMtsSymbol(symbol, true);
                if (!ti) {
                    ti = getByTTSecId(symbol, true);
                    if (!ti) {
                        ti = getByMtsContract(symbol, true);
                        if (!ti) {
                            ti = getByBbgId(symbol, true);
                            if (!ti) {
                                logError("%s not found in symbol map under tradable, MTS Symbol, TT Sec_Id, MTS Contract or BBG ID!", symbol.c_str());
                                return NULL;
                            }
                        }
                    }
                }
            }
            s_ti_map[symbol] = ti;
            return ti;
        }
    }

    SymbolMapWriter::SymbolMapWriter() {};
    SymbolMapWriter::SymbolMapWriter(const std::string& cfg_file) {
        addTradable(SymbolMapReader::getFile(cfg_file));
    }
    SymbolMapWriter::~SymbolMapWriter() {
        clear();
    }

    void SymbolMapWriter::clear() {
        for (const auto&kv : m_tradable) {
            delete kv.second;
        }
        m_tradable.clear();
    }

    void SymbolMapWriter::addTradable(const TradableInfo* tradable) {
        m_tradable[tradable->_tradable] = tradable;
    }

    void SymbolMapWriter::addTradable(
            const std::string& tradable,
            const std::string& symbol,
            const std::string& exch_symbol,
            const std::string& mts_contract,
            const std::string& mts_symbol,
            const std::string& venue,
            const std::string& type,
            const std::string& contract_month,
            const std::string& tt_security_id, 
            const std::string& tt_venue,
            const std::string& bbg_id,
            const std::string& currency,
            const std::string& expiration_date,
            double tick_size,
            double point_value,
            double px_multiplier,
            double bbg_px_multiplier,
            int N,
            int expiration_days,
            int lotspermin,
            bool subscribed) {
        addTradable(new TradableInfo(tradable, symbol, exch_symbol, mts_contract, mts_symbol, venue, type,
                                     contract_month, tt_security_id, tt_venue, bbg_id, currency, expiration_date,
                                     tick_size, point_value, px_multiplier, bbg_px_multiplier, N, expiration_days,
                                     lotspermin, subscribed));
    }

    void SymbolMapWriter::addTradable(const SymbolMapReader& reader) {
        const auto& trd (reader.listAllTradable());
        for (const auto* t : trd) {
            addTradable(new TradableInfo(*t));
        }
    }

    void SymbolMapWriter::addTradable(const SymbolMapWriter& writer) {
        if (&writer == this) {
            return;
        }
        for (const auto& t : writer.m_tradable) {
            addTradable(new TradableInfo(*t.second));
        }
    }

    void SymbolMapWriter::delTradable(const std::string& tradable) {
        auto iter(m_tradable.find(tradable)) ; 
        if (iter != m_tradable.end()) {
            m_tradable.erase(iter);
        }
    }

    void SymbolMapWriter::toConfigFile(const std::string& cfg_file) const {
        try {
            utils::ConfigureReader cfg;
            auto* cfg_value = cfg.set("tradable");
            for (const auto& kv: m_tradable) {
                kv.second->toKeyValue(*cfg_value);
            }
            cfg.toFile(((cfg_file.empty()) ? plcc_getString(SymbolMapConfig) : cfg_file).c_str());
        } catch (const std::exception& e) {
            logError("Error writing the symbol map: %s", e.what());
            throw e;
        }
    }
}
