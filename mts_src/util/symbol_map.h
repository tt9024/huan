#pragma once
#include <unordered_map>
#include <set>

#include <ConfigureReader.hpp>

namespace utils {
    struct TradableInfo {
        std::string _tradable;     // i.e. CLX2
        std::string _symbol;       // i.e. WTI
        std::string _exch_symbol;  // i.e. CL
        std::string _mts_contract; // i.e. WTI_202211
        std::string _mts_symbol;   // i.e. WTI_N1
        std::string _venue;        // i.e. NYM
        std::string _type;         // i.e. FUT
        std::string _contract_month;

        // additional information
        std::string _tt_security_id;
        std::string _tt_venue;
        std::string _bbg_id;
        std::string _currency;
        std::string _expiration_date;

        double _tick_size;
        double _point_value;
        double _px_multiplier;
        double _bbg_px_multiplier;
        int _N;
        int _expiration_days;
        int _lotspermin;
        bool _subscribed;

        TradableInfo();
        TradableInfo(const std::string& tradable,
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
                     bool subscribed);

        // create from a key-value config
        TradableInfo(const std::string& tradable, const ConfigureReader& kv);

        // write to a key-value config
        void toKeyValue(ConfigureReader::Value& value) const;

        std::string toString() const;
    };

    class SymbolMapReader {
    public:
        // the path to config file is defined in main.cfg
        static const SymbolMapReader&          get             (bool reload=false);
        static const SymbolMapReader           getFile         (const std::string& cfg_file);
        const TradableInfo*                    getByMtsSymbol  (const std::string& symbol, bool return_null=false) const;
        const TradableInfo*                    getByTradable   (const std::string& tradable_symbol, bool return_null=false) const;
        const TradableInfo*                    getByTTSecId    (const std::string& tt_security_id, bool return_null=false) const;
        const TradableInfo*                    getByMtsContract(const std::string& mts_contract, bool return_null=false) const;
        const TradableInfo*                    getByBbgId      (const std::string& bbg_id, bool return_null=false) const;
        const TradableInfo*                    getN0ByN1       (const std::string& tradable_symbol) const;

        // get N1 ti from any contracts of this symbol, i.e. WTI_N0/1/2
        const TradableInfo*                    getN1           (const std::string& tradable_symbol) const;

        // list all contracts of the symbol, given by symbol i.e. "WTI'
        const std::vector<const TradableInfo*> getAllBySymbol  (const std::string& symbol)          const;
        // list all contracts of the symbol given by tradable_symbol i.e. "CLH2"
        const std::vector<const TradableInfo*> getAllByTradable(const std::string& tradable_symbol) const;
        const std::vector<const TradableInfo*> listAllTradable() const;
        const std::vector<const TradableInfo*> getAllByMtsVenue(const std::vector<std::string>& mts_venue, int max_n) const;

        // used by subscription specification, 
        // this reads "MaxN", "MTSVenue" and "MTSSymbol" from main config
        // return 2 lists of mts symbols, i.e. WTI_N1, first being the primary, second being the backup
        // when provider is empty string, it returns all subscribed symbols as primary and empty set as secondary
        // max_n controls the number of front contracts for each symbol, if -1, then read
        // MaxN from the main.cfg.  
        // Set max_n to 1 to get only the front contracts, useful when checking market data
        const std::pair<std::vector<std::string>,std::vector<std::string>> getSubscriptions(const std::string& provider, int max_n = -1) const;
        const std::vector<std::string> getPrimarySubscriptions(int max_n = -1) const;

        ~SymbolMapReader();

        static double TickSize(const std::string& tradable);

        // get a tradable symbol using either of the following:
        // 1. a tradable
        // 2. mts symbol
        // 3. tt_security id
        // 4. mts_contract
        // 5. bbg_id
        const TradableInfo* getTradableInfo(const std::string& symbol) const;

        // get a tradable symbol/mkt from a symbol given as above
        // return "" if not found
        std::string getTradableSymbol(const std::string& symbol) const;
        std::string getTradableMkt(const std::string& symbol) const;

        bool isMLegSymbol(const std::string& symbol) const;

    private:
        explicit SymbolMapReader(const std::string& cfg_file = "");
        SymbolMapReader(const SymbolMapReader&) = delete;
        void operator=(const SymbolMapReader&) = delete;
        void reload();
        void clear();

        const std::string m_cfg_file;
        std::unordered_map<std::string, const TradableInfo*> m_tradable;
        std::unordered_map<std::string, std::vector<const TradableInfo*> > m_tradable_by_symbol;
        std::unordered_map<std::string, const TradableInfo*> m_tradable_by_mts;
        std::unordered_map<std::string, const TradableInfo*> m_tradable_by_ttsecid;
        std::unordered_map<std::string, const TradableInfo*> m_tradable_by_mts_contract;
        std::unordered_map<std::string, const TradableInfo*> m_tradable_by_bbg_id;
        std::unordered_map<std::string, const TradableInfo*> m_n0_by_n1;

        const std::pair<std::set<std::string>, std::set<std::string>> getSubscriptionsUnsafe(const std::string& provider, int max_n) const;
        std::set<std::string> getSubSymbols(int max_n, 
                                               const std::set<std::string>& venue_set, 
                                               const std::set<std::string>& symbol_set) const;
    };

    class SymbolMapWriter {
    public:
        // the path to config file is defined in main.cfg
        SymbolMapWriter();
        explicit SymbolMapWriter(const std::string& cfg_file);
        ~SymbolMapWriter();

        void toConfigFile(const std::string& cfg_file = "") const;
        void addTradable(
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
            bool subscribed);
        void addTradable(const SymbolMapReader& reader);
        void addTradable(const SymbolMapWriter& writer);
        void addTradable(const TradableInfo* tradable);

        void delTradable(const std::string& tradable);
        void clear();

    private:
        SymbolMapWriter(const SymbolMapWriter&) = delete;
        void operator=(const SymbolMapWriter&) = delete;

        std::unordered_map<std::string, const TradableInfo*> m_tradable;
    };

    //
    // inline implementations
    //
    
    inline
    TradableInfo::TradableInfo(const std::string &tradable, const ConfigureReader& kv)
    : _tradable(tradable), 
      _symbol(kv.get<std::string>("symbol")),
      _exch_symbol(kv.get<std::string>("exch_symbol")),
      _mts_contract(kv.get<std::string>("mts_contract")),
      _mts_symbol(kv.get<std::string>("mts_symbol")),
      _venue(kv.get<std::string>("venue")),
      _type(kv.get<std::string>("type")),
      _contract_month(kv.get<std::string>("contract_month")),
      _tt_security_id(kv.get<std::string>("tt_security_id")),
      _tt_venue(kv.get<std::string>("tt_venue")),
      _bbg_id(kv.get<std::string>("bbg_id")),
      _currency(kv.get<std::string>("currency")),
      _expiration_date(kv.get<std::string>("expiration_date")),
      _tick_size(kv.get<double>("tick_size")),
      _point_value(kv.get<double>("point_value")),
      _px_multiplier(kv.get<double>("px_multiplier")),
      _bbg_px_multiplier(kv.get<double>("bbg_px_multiplier")),
      _N(kv.get<int>("N")),
      _expiration_days(kv.get<int>("expiration_days")),
      _lotspermin(kv.get<int>("lotspermin",nullptr, 1)),
      // subscribe not supported yet
      //_subscribed(kv.get<int>("subscribed") != 0)
      _subscribed(0)
    {}

    inline
    void TradableInfo::toKeyValue(ConfigureReader::Value& value) const {
        auto* v = value.addKey(_tradable);
        v->addKey("symbol")->set(_symbol);
        v->addKey("exch_symbol")->set(_exch_symbol);
        v->addKey("mts_contract")->set(_mts_contract);
        v->addKey("mts_symbol")->set(_mts_symbol);
        v->addKey("venue")->set(_venue);
        v->addKey("type")->set(_type);
        v->addKey("contract_month")->set(_contract_month);
        v->addKey("tt_security_id")->set(_tt_security_id);
        v->addKey("tt_venue")->set(_tt_venue);
        v->addKey("bbg_id")->set(_bbg_id);
        v->addKey("currency")->set(_currency);
        v->addKey("expiration_date")->set(_expiration_date);
        v->addKey("tick_size")->set(_tick_size);
        v->addKey("point_value")->set(_point_value);
        v->addKey("px_multiplier")->set(_px_multiplier);
        v->addKey("bbg_px_multiplier")->set(_bbg_px_multiplier);
        v->addKey("N")->set(_N);
        v->addKey("expiration_days")->set(_expiration_days);
        v->addKey("lotspermin")->set(_lotspermin);
        v->addKey("subscribed")->set((_subscribed? 1:0));
    }

    inline
    double SymbolMapReader::TickSize(const std::string& tradable) {
        return get().getTradableInfo(tradable)->_tick_size; 
    }

    inline
    const TradableInfo* SymbolMapReader::getN0ByN1(const std::string& tradable_symbol) const {
        const auto iter = m_n0_by_n1.find(tradable_symbol);
        if (iter != m_n0_by_n1.end()) {
            return iter->second;
        }
        return NULL;
    }

    inline
    const TradableInfo* SymbolMapReader::getN1(const std::string& tradable_symbol) const {
        const auto& ti_vec (getAllByTradable(tradable_symbol));
        for (const auto& ti:ti_vec) {
            if (ti->_N == 1) {
                return ti;
            }
        }
        return NULL;
    }

    inline
    const std::vector<const TradableInfo*> SymbolMapReader::getAllByTradable(const std::string& tradable_symbol) const {
        const auto* ti = getByTradable(tradable_symbol);
        return getAllBySymbol(ti->_symbol);
    }

    inline
    const std::vector<const TradableInfo*> SymbolMapReader::listAllTradable() const {
        std::vector<const TradableInfo*> vec;
        for (const auto& kv:m_tradable) {
            vec.push_back(kv.second);
        }
        return vec;
    }
}


