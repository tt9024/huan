#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <set>

#include "plcc/PLCC.hpp"
#include "time_util.h"
#include "symbol_map.h"
#include "queue.h"  
#include "csv_util.h"

/*
 * Low level L2 Book, building and querying. 
 */

namespace md {

/*
 * BookConfig, Global BookQ, BBO utilities
 */

class VenueConfig {
public:
    static const VenueConfig& get() {
        static VenueConfig vc;
        return vc;
    }

    int start_hour(const std::string& venue) const { 
        return SMOD(findValue(venue, 0), 24);
    }

    int start_min (const std::string& venue) const {
        return findValue(venue, 1);
    }

    int end_hour(const std::string& venue) const {
        return findValue(venue, 2);
    }

    int end_min(const std::string& venue) const {
        return findValue(venue, 3);
    }

    bool isTradingTime(const std::string& venue, const time_t cur_utc) const {
        const auto iter = venue_map.find(venue);
        if (iter == venue_map.end()) {
            logError("Venue not found: %s", venue.c_str());
            throw std::runtime_error("Venue not found!");
        }
        const auto& hm = iter->second;
        return utils::TimeUtil::isTradingTime(cur_utc, hm[0], hm[1], hm[2], hm[3]);
    }

    std::pair<time_t, time_t> startEndUTC(const std::string& venue, const time_t cur_utc, int snap) const {
        // getting the venue's starting and ending time given the current time.  
        // If the current time is a trading time, getting the current trading day's starting time
        // If the current time is not a trading time, 
        //    snap = 0: return 0
        //    snap = 1: return previous trading day's starting time
        //    snap = 2: return next trading day's starting time
        // Note for over-night venues, the trading day of over-night session is the next day.
        // For example, CME starting at 18:00, ends at 17:00.  
        // If current time is 20:00, then trading day is next day, startUTC is today's 18:00.
        // If current time is 17:55pm on Sunday, snap = 2, then trading day is Monday,
        // and startUTC is today's 18:00.
        //
        // Return: a pair of utc time stamp in seconds for starting and ending time of the
        //         trading day.

        const auto iter = venue_map.find(venue);
        if (iter == venue_map.end()) {
            logError("Venue not found: %s", venue.c_str());
            throw std::runtime_error("Venue not found!");
        }
        const auto& hm = iter->second;
        const auto curDay = utils::TimeUtil::tradingDay(cur_utc, hm[0], hm[1], hm[2], hm[3], 0, snap);
        time_t sutc = utils::TimeUtil::string_to_frac_UTC(curDay.c_str(), 0, "%Y%m%d");
        return std::pair<time_t, time_t>( 
                sutc + hm[0]*3600 + hm[1]*60, 
                sutc + hm[2]*3600 + hm[3]*60
               ) ;
    }

    time_t sessionLength(const std::string& venue) const {
        // return number of seconds in a trading day
        const auto iter = venue_map.find(venue);
        if (iter == venue_map.end()) {
            logError("Venue not found: %s", venue.c_str());
            throw std::runtime_error("Venue not found!");
        }
        const auto& hm = iter->second;
        return hm[2]*3600 + hm[3]*60 - hm[0]*3600 - hm[1]*60;
    }

private:
    std::map<std::string, std::vector<int> > venue_map;
    VenueConfig() {
        const auto cfg = plcc_getString("SymbolMap");
        const auto& vc = utils::ConfigureReader(cfg.c_str()).getReader("venue");
        auto vl = vc.listKeys();
        for (const auto& v : vl) {
            auto hm = vc.getReader(v).getArr<std::string>("hours");
            // hm in [ start_hour, start_min, end_hour, end_min ]
            if (hm.size() != 4) {
                logError("Venue Reading Error for %s: wrong size.", v.c_str());
                throw std::runtime_error("Venue Reading Error!");
            }
            int sh = std::stoi(hm[0]);
            int sm = std::stoi(hm[1]);
            int eh = std::stoi(hm[2]);
            int em = std::stoi(hm[3]);
            if (eh < 0 || em < 0 || sm < 0 ||
                (eh-sh > 24) || (sh > eh)) {
                logError("Venue %s minutes negative!", v.c_str());
                throw std::runtime_error("Venue minutes negative!");
            }
            std::vector<int> hv {sh, sm, eh, em};
            venue_map.emplace(v, hv);
        }
    };

    int findValue(const std::string& venue, int offset) const {
        const auto iter = venue_map.find(venue);
        if (iter == venue_map.end()) {
            logError("Venue not found: %s", venue.c_str());
            throw std::runtime_error("Venue not found: " + venue);
        }
        return iter->second[offset];
    }
};

struct BookConfig {
    std::string venue;
    std::string symbol;
    std::string type; // "L1, L2, TbT"
    std::string provider; // BPIPE, TT, etc

    BookConfig():venue(""),symbol(""),type(""),provider(""){};

#ifndef NUMPY_COMPILE
    // provider is empty for prod feed, otherwise, the qname/bfname will have provider
    BookConfig(const std::string& v_, const std::string& s_, const std::string& bt_, const std::string& provider_) 
    : venue(v_), type(bt_), provider(provider_) {
        const auto* ti = utils::SymbolMapReader::get().getTradableInfo(s_);
        symbol = ti->_tradable;
        if (venue.size()==0) {
            venue = ti->_venue;
        }
    };

    BookConfig(const std::string& v_, const std::string& s_, const std::string& bt_) 
    : venue(v_), type(bt_) {
        const auto* ti = utils::SymbolMapReader::get().getTradableInfo(s_);
        symbol = ti->_tradable;
        if (venue.size()==0) {
            venue = utils::SymbolMapReader::get().getTradableInfo(symbol)->_venue;
        }
    };

    // construct by provider@venue/symbol and a book type: "L1 or L2"
    // or a tradable or MTS symbol in case "/" is not found
    // venue could have a '@' to specify a provider, i.e. bbg@CME, or tt@ICE
    // it is possible to omit venue, i.e.,  bbg@/SPX_N1
    BookConfig(const std::string& venu_symbol, const std::string& bt) :
        type(bt) {
        size_t n = strlen(venu_symbol.c_str());
        auto pos = venu_symbol.find("/");
        if (pos == std::string::npos) {
            // try tradable
            const auto* ti(utils::SymbolMapReader::get().getTradableInfo(venu_symbol));
            if (!ti) {
                logError("symbol cannot be parsed: %s", venu_symbol.c_str());
                throw std::runtime_error(std::string("symbol cannot be parsed!") + venu_symbol);
            }
            symbol = ti->_tradable;
            venue = ti->_venue;
        } else {
            // venue (and optional provider) and symbol specified
            const auto venue0 = venu_symbol.substr(0,pos);
            const auto symbol0 = venu_symbol.substr(pos+1,n);
            // parsing symbols
            const auto* ti(utils::SymbolMapReader::get().getTradableInfo(symbol0));
            if (!ti) {
                logError("symbol cannot be parsed: %s", symbol0.c_str());
                throw std::runtime_error(std::string("symbol cannot be parsed!") + symbol0);
            }
            symbol = ti->_tradable;

            // parsing venue
            pos = venue0.find("@");
            if (pos != std::string::npos) {
                provider = venue0.substr(0,pos);
                venue = venue0.substr(pos+1,venue0.size());
            } else {
                venue = venue0;
            }
            if (venue.size() == 0) {
                venue = ti->_venue;
            }
        }
        logDebug("BookConfig %s", toString().c_str());
    }

    explicit BookConfig(const BookConfig& bcfg)
    :venue(bcfg.venue), symbol(bcfg.symbol), type(bcfg.type), provider(bcfg.provider) {};

    virtual ~BookConfig() {};

    std::string qname() const {
        auto qn = venue+"_"+symbol+"_"+type;
        if (provider.size()>0) {
            qn = provider + "_" + qn;
        }
        return qn;
    }

    std::string toString() const {
        try {
            const auto* ti(utils::SymbolMapReader::get().getTradableInfo(symbol));

            return venue+"_"+ti->_mts_contract+"_"+type+"_"+provider + " qname:(" + qname() + ")";
        } catch (const std::exception& e) {
            return qname();
        }
    }

    std::string bfname(int barsec, const std::string& bar_path) const {
        // get the current bar file
        return bar_path+"/"+
               (provider.size()>0? (provider + "_"):"") + 
               venue+"_"+symbol+"_"+std::to_string(barsec)+"S.csv";
    }

    std::string bfname(int barsec) const {
        return bfname(barsec, plcc_getString("BarPath"));
    }

    std::string bhfname(int barsec) const {
        // get the previous bar file
        return bfname(barsec, plcc_getString("HistPath"));
    }

    std::vector<int> barsec_vec() const {
        std::vector<std::string> bsv = plcc_getStringArr("BarSec");
        if (bsv.size() == 0) {
            throw std::runtime_error(std::string("BarSec not found in config"));
        }
        std::vector<int> ret;
        for (const auto& bs: bsv) {
            ret.push_back(std::stoi(bs));
        };
        return ret;
    }

    const utils::TradableInfo* getTradableInfo() const {
        return utils::SymbolMapReader::get().getByTradable(symbol, true);
    }

    bool operator== (const BookConfig& cfg) const {
        return qname() == cfg.qname();
    }
#else    //NYMPY_COMPILE - 
    std::string toString() const {
        return venue+"_"+symbol;
    }
    BookConfig(const std::string& v_, const std::string& s_, const std::string& bt_) {};
    BookConfig(const std::string& venu_symbol, const std::string& bt) {};
    explicit BookConfig(const BookConfig& bcfg) {};
    virtual ~BookConfig() {};

    std::string qname() const { return "";};
    std::string bfname(int barsec, const std::string& bar_path) const { return "";};
    std::string bfname(int barsec) const { return "";};
    std::string bhfname(int barsec) const { return "";};
    std::vector<int> barsec_vec() const { return std::vector<int>();};
    const utils::TradableInfo* getTradableInfo() const { return nullptr; };
    bool operator== (const BookConfig& cfg) const { return false;};
#endif
};

/*********************
 * Level 2 Book Types
 *********************/

using Price = double;
using Quantity = int32_t;
using TSMicro = uint64_t;
using BBOTuple = std::tuple<Price, Quantity, Price, Quantity>; //bpx,bsz,apx,asz
using TradeTuple = std::tuple<Price, Quantity, uint32_t, uint64_t>; //tpx,tsz,attr,micro

struct PriceEntry {
#pragma pack(push,1)
    Price price;  
    Quantity size;
    TSMicro ts_micro;
    Quantity count;
#pragma pack(pop)
    PriceEntry() : price(0), size(0), ts_micro(0), count(0) {};
    PriceEntry(Price p, Quantity s, Quantity cnt, TSMicro ts) : price(p), size(s), ts_micro(ts), count(cnt) {
        if (!ts) {
            ts_micro = utils::TimeUtil::cur_micro();
        }
    };
    void set(Price px, Quantity sz, TSMicro ts) {
        price=px;
        size=sz;
        ts_micro=ts;
    }

    void reset() {
        price = 0; size = 0;ts_micro=0;count=0;
    }

    Price getPrice() const {
        return price;
    }

    std::string toString() const {
        char buf[64];
        snprintf(buf, sizeof(buf), "%lld(%s:%d)", (unsigned long long) ts_micro, PriceCString(getPrice()), size);
        return std::string(buf);
    }

    std::string toStringPxSz() const {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d:%s",  (int) size, PriceCString(getPrice()));
        return std::string(buf);
    }
};

template<int BookLevel>
struct BookDepotLevel {
#pragma pack(push,1)
    // this structure needs to be aligned
    uint64_t update_ts_micro;  // this can be obtained from pe's ts
    // bid first, ask second
    int update_level;
    int update_type;  // 0: bid, 1: ask, 2: trade, 3: special trade(see trade_attr)
    PriceEntry pe[2*BookLevel];
    int avail_level[2];
    Price trade_price;
    Quantity trade_size;
    uint32_t trade_attr;  // buy(0)/sell(1), 2/3(unsure), 4+(special trade types)
#pragma pack(pop)

    // ENUM types
    enum {
        BidUpdateType = 0,
        AskUpdateType = 1,
        TradeUpdateType = 2,
        TradeSpecialUpdateType = 3,
        UnknownUpdateType = 4
    };

    enum {
        TradeDirBuy = 0,
        TradeDirSell= 1,
        TradeDirUnknown = 2
        // special trades from bits 8
    };

    BookDepotLevel() {
        reset();
    };

    BookDepotLevel<BookLevel>& operator = (const BookDepotLevel<BookLevel>& book) {
        if (&book == this) {
            return *this;
        }
        memcpy ((char*)this, (char*)&book, sizeof(BookDepotLevel<BookLevel>));
        return *this;
    };

    explicit BookDepotLevel<BookLevel>(const std::string& csv_line) {
        reset();
        updateFrom(utils::CSVUtil::read_line(csv_line));
    }

    explicit BookDepotLevel<BookLevel>(const std::vector<std::string>& csv_line_tokens) {
        updateFrom(csv_line_tokens);
    }

    void reset() {
        memset((char*)this, 0, sizeof(BookDepotLevel<BookLevel>));
    }

    /*******************************
     * Prices from book
     *     Getters
     *******************************/

    // assuming *size not null
    Price getBid(Quantity* size) const {
        *size=0;
        if (__builtin_expect(avail_level[0] < 1,0))
            return 0;
        for (int i = 0; i<avail_level[0]; ++i) {
            if (__builtin_expect(pe[i].size > 0,1)) {
                *size=pe[i].size;
                return pe[i].getPrice();
            }
        }
        return 0;
    }

    Price getAsk(Quantity* size) const {
        *size=0;
        if (__builtin_expect(avail_level[1] < 1,0))
            return 0;
        for (int i = BookLevel; i<BookLevel+avail_level[1]; ++i) {
            if (__builtin_expect(pe[i].size > 0,1)) {
                *size=pe[i].size;
                return pe[i].getPrice();
            }
        }
        return 0;
    }

    BBOTuple getBBOTuple() const {
        Quantity bsz,asz;
        Price bpx=getBid(&bsz), apx=getAsk(&asz);
        return {bpx,bsz,apx,asz};
    }

    BBOTuple getBBOTupleUnSafe() const {
        const auto* pe0=pe+BookLevel;
        return {pe->price, pe->size, pe0->price, pe0->size};
    }

    Price getBid() const {
        if (__builtin_expect(avail_level[0] < 1,0))
            return 0;
        for (int i = 0; i<avail_level[0]; ++i) {
            if (__builtin_expect(pe[i].size > 0,1)) {
                return pe[i].getPrice();
            }
        }
        return 0;
    }

    Price getAsk() const {
        if (__builtin_expect(avail_level[1] < 1,0))
            return 0;
        for (int i = BookLevel; i<BookLevel+avail_level[1]; ++i) {
            if (__builtin_expect(pe[i].size > 0,1)) {
                return pe[i].getPrice();
            }
        }
        return 0;
    }

    Price getBestLevel(bool isBid, Quantity*size) const {
        return isBid? getBid(size) : getAsk(size);
    }

    Price getMid() const {
        if (__builtin_expect(avail_level[0] * avail_level[1] == 0,0))
            return 0;
        return (getBid() + getAsk())/2;
    }

    Price getVWAP(int level, bool isBid, Quantity* q=NULL) const {
        const int side=isBid?0:1;
        int lvl = _MIN_(level, avail_level[side]);
        Quantity qty = 0;
        Price px = 0;
        const PriceEntry* p = pe + side*BookLevel;
        const PriceEntry* p0 = p + lvl;
        while (p < p0) {
                px += p->price * p->size;
                qty += p->size;
                ++p;
        }
        if (q) *q = qty;
        return px/qty;
    }

    // const state validators
    bool isTradeUpdate() const {
        return ((update_type == TradeUpdateType) || 
                (update_type == TradeSpecialUpdateType));
    }

    bool isNormalTradeUpdate() const {
        return (update_type == TradeUpdateType) &&
               (! isSpecialTrade(trade_attr>>8));
    };

    bool isQuoteUpdate() const {
        return ((update_type == BidUpdateType) ||
                (update_type == AskUpdateType));
    }

    bool isValidQuote() const {
        if (__builtin_expect(avail_level[0]*avail_level[1] == 0, 0)) {
            return false;
        }
        return (pe->price + 1e-10) < (pe+BookLevel)->price;
    }

    bool isValidNormalTrade() const {
        return (update_type==TradeUpdateType) && 
               (trade_price!=0) && (trade_size!=0);
    }

    bool isValidSpecialTrade() const {
        return (update_type==TradeSpecialUpdateType) &&
               ((trade_price!=0) || (trade_size!=0));
    }

    bool isValid() const {
        if (__builtin_expect(isQuoteUpdate(),1)) {
            return isValidQuote();
        }
        if (__builtin_expect(isNormalTradeUpdate(), 1)) {
            return isValidNormalTrade();
        }
        if (__builtin_expect(isTradeUpdate(), 1)) {
            return isValidSpecialTrade();
        }
        return false;
    }

    /*******************************
     * Prices from book
     *     Setters (L2)
     *******************************/

    bool newPrice(Price price, Quantity size, unsigned int level, bool is_bid, uint64_t ts_micro) {
        int side = is_bid?0:1;
        unsigned int levels = (unsigned int) avail_level[side];
        // checking on the level
        if (__builtin_expect(((level>levels)  || (levels >= BookLevel)), 0)) {
            logError("new price wrong level %d", level);
            return false;
            //throw new std::runtime_error("error!");
        }

        update_level = level;
        update_type = (is_bid?BidUpdateType:AskUpdateType);

        // move subsequent levels down
        PriceEntry* pe = getEntry(level, side);
        if (levels > level) {
            memmove(pe+1, pe, (levels - level)*sizeof(PriceEntry));
        };

        pe->set(price, size, ts_micro);
        ++(avail_level[side]);
        update_ts_micro=ts_micro;
        return true;
    }

    bool delPrice(unsigned int level, bool is_bid, uint64_t ts_micro) {
        int side = is_bid?0:1;
        if (__builtin_expect((level>=(unsigned int)avail_level[side]), 0)) {
            logError("del price wrong level %d", level);
            return false;
            //throw new std::runtime_error("error!");
        }

        update_level = level;
        update_type = (is_bid?BidUpdateType:AskUpdateType);
        // move subsequent levels up
        unsigned int levels = avail_level[side];
        if (levels > level + 1) {
            PriceEntry* pe = getEntry(level, side);
            memmove(pe, pe+1, (levels-level-1)*sizeof(PriceEntry));
        };
        --(avail_level[side]);
        update_ts_micro=ts_micro;
        return true;
    }

    bool updPrice(Price price, Quantity size, unsigned int level, bool is_bid, uint64_t ts_micro) {
        int side = is_bid?0:1;
        if (__builtin_expect((level>=(unsigned int)avail_level[side]), 0)) {
            logError("update price wrong level %d", level);
            return false;
            //throw new std::runtime_error("error!");
        }

        PriceEntry* pe = getEntry(level, side);
        if (__builtin_expect((std::abs(pe->price-price)<1e-10) && (pe->size == size), 0)) {
            // nothing changed, but not a failure, in case of bbo update, only one
            // side could be updated
            return false;
        }
        update_level = level;
        update_type = (is_bid?BidUpdateType:AskUpdateType);
        pe->set(price, size, ts_micro);
        update_ts_micro=ts_micro;
        return true;
    }

    bool updBBO(Price price, Quantity size, bool is_bid, uint64_t ts_micro) {
        int side = is_bid?0:1;
        if (__builtin_expect((0 == avail_level[side]), 0)) {
            return newPrice(price, size, 0, is_bid, ts_micro);
        }
        return updPrice(price, size, 0, is_bid, ts_micro);
    }

    bool updBBO(const BBOTuple& bbo, uint64_t ts_micro) {
        bool ret=true;
        const auto [bp,bs,ap,as]=bbo;
        // there could be one sided updates, 
        // so return true if at least one side
        // is updatd
        ret  = updBBO(bp,bs,true,ts_micro);
        ret |= updBBO(ap,as,false,ts_micro); 
        return ret;
    }

    /**************************
     * Trade from the Book
     *    Getters
     *************************/

    // Buy or Unknown: trade_size, Sell: -trade_size
    Quantity getTradeVolumeSigned() const {
        return ((trade_attr&3) == TradeDirSell? -trade_size: trade_size);
    }

    std::string getSpecialTradeTypeString(int type) const {
        return "Special";
    }

    std::string getTradeAttrString(uint32_t attr) const {
        std::string ret;
        if (__builtin_expect( (attr>>8)>0 , 0)) {
            // special trade type
            ret = getSpecialTradeTypeString(attr>>8);
            ret += "-";
        }
        if (__builtin_expect((attr&0x2) == 0,1)) {
            return ret+((attr&1)==0?"Buy":"Sell");
        }
        return ret + "NoDir";
    }

    TradeTuple getTradeTuple() const {
        if (isTradeUpdate()) {
            return {trade_price, trade_size, trade_attr, update_ts_micro};
        }
        return {0, 0, 0, 0};
    }

    int getTradeAttr(Price tpx) const {
        // return trade_attr based on tpx w.r.t. current BBO -
        //     0: buy, 
        //     1: sell
        // in case tpx in the middle of BBO, return -
        //     2: NA/prev_buy, 
        //     3: NA/prev_sell
        //
        if (__builtin_expect(isValidQuote(),1)) {
            const Price bd=std::abs(getBid()-tpx);
            const Price ad=std::abs(getAsk()-tpx);
            if (ad>bd+1e-10) {
                return TradeDirSell;  // trade attr 1 - sell
            } else if (__builtin_expect(bd>ad+1e-10,1)) {
                return TradeDirBuy; // trade attr 0 - buy
            }
        }
        // cannot decide, save previous attr, adding unknown bit
        return (trade_attr&0x1)|TradeDirUnknown; // 2 or 3
    }

    /**************************
     * Trade from the Book
     *    Setters
     *************************/

    // normal trade
    void addTrade(Price px, Quantity sz, bool is_buy, uint64_t ts_micro) {
        trade_price=px;
        trade_size=sz;
        setTradeSide(is_buy);
        update_ts_micro = ts_micro;
    }

    void addTrade(const TradeTuple& tt) {
        const auto [px, sz, attr, ts_micro] = tt;
        trade_price=px;
        trade_size=sz;
        setTradeAttr(attr); // may include a type
        update_ts_micro = ts_micro;
    }

    // special trade with non-zero type, i.e. block, EFP, or other trades
    void addSpecialTrade(int type, Price px, Quantity sz, int normal_attr, uint64_t ts_micro) {
        trade_price=px;
        trade_size=sz;
        setSpecialTradeAttr(type, normal_attr);
        update_ts_micro = ts_micro;
    }

    /*********************
     * serialization
     *********************/
    std::string toCSV(int bbo_level=1) const {
        // update_micro, update_type(0,1,2,3), update_level, trd_px, trd_sz, trd_attr, bid_levels, bid_0_px, bid_0_sz,... ask_levels, ask_0_px, ask_0_sz (,...)

        char buf[1024];
        int n = 0;
        n += snprintf(buf+n, sizeof(buf)-n, "%llu, %d, %d, %.7f, %d, %d",
                        (unsigned long long)update_ts_micro,update_type,update_level,
                        trade_price, (int)trade_size, trade_attr);
        for (int s = 0; s < 2; ++s) {
            int levels = avail_level[s];
            levels = _MIN_(levels, bbo_level);
            n += snprintf(buf+n, sizeof(buf)-n, ", %d", levels);
            const PriceEntry* pe_ = &(pe[s*BookLevel]);
            for (int i = 0; i<levels; ++i) {
                n += snprintf(buf+n, sizeof(buf)-n, ", %.7f, %d", 
                        (double) pe_->price, (int) pe_->size);
                    ++pe_;
            }
        }
        return std::string(buf);
    }

    void updateFrom(const std::vector<std::string>& tk) {
        // read the line from toCSV(), entertaining multiple levels
        update_ts_micro = std::stoll(tk[0]);
        update_type = std::stoi(tk[1]);
        update_level = std::stoi(tk[2]);
        trade_price = std::stod(tk[3]);
        trade_size = std::stoi(tk[4]);
        trade_attr = std::stoi(tk[5]);
        int ix = 6;
        for (int s=0; s<2; ++s) {
            avail_level[s] = std::stoi(tk[ix++]);
            for (int i=0; i<avail_level[s]; ++i) {
                double px = std::stod(tk[ix++]);
                int sz = std::stoi(tk[ix++]);
                pe[s*BookLevel+i] = PriceEntry(px, sz, 1, update_ts_micro);
            }
        }
    }

    /**************************
     * to strings
     **************************/

    const char* getUpdateType() const {
        switch (update_type) {
        case BidUpdateType : return "Quote(Bid)";
        case AskUpdateType : return "Quote(Ask)";
        case TradeUpdateType : return "Trade";
        case TradeSpecialUpdateType : return "Special Trade";
        case UnknownUpdateType : return "Unknown";
        }
        logError("unknown update type %d", update_type);
        throw std::runtime_error("unknown update type!");
    }

    std::string toString() const {
        char buf[1024];
        int n = 0;
        const char* update_type = getUpdateType();
        n += snprintf(buf+n, sizeof(buf)-n, "%llu (%s-%d)",
             (unsigned long long)update_ts_micro,update_type,update_level);
        for (int s = 0; s < 2; ++s)
        {
            int levels = avail_level[s];
            n += snprintf(buf+n, sizeof(buf)-n, " %s(%d) [ ", s==0?"Bid":"Ask", levels);
            const PriceEntry* pe_ = &(pe[s*BookLevel]);
            for (int i = 0; i<levels; ++i) {
                if (pe_->size) {
                    n += snprintf(buf+n, sizeof(buf)-n, " (%d)%s ", i, pe_->toString().c_str());
                }
                ++pe_;
            }
            n += snprintf(buf+n, sizeof(buf)-n, " ] ");
        }
        n += snprintf(buf+n, sizeof(buf)-n, "%s %d@%f",
                      getTradeAttrString(trade_attr).c_str(),
                      trade_size, trade_price);
        return std::string(buf);
    }

    std::string prettyPrint() const {
        char buf[1024];
        size_t n = snprintf(buf,sizeof(buf), "%lld:%s,upd_lvl(%d-%d:%d)\n",
                           (long long)update_ts_micro,
                            getUpdateType(),
                            update_level,
                            avail_level[0],
                            avail_level[1]);
        if (isTradeUpdate()) {
            // trade
            n+=snprintf(buf+n,sizeof(buf)-n,"   %s %d@%.7lf\n",
                        getTradeAttrString(trade_attr).c_str(),
                        trade_size, trade_price);
        } else {
            // quote
            int lvl=avail_level[0];
            if (lvl > avail_level[1]) lvl=avail_level[1];
            for (int i=0;i<lvl;++i) {
                n+=snprintf(buf+n,sizeof(buf)-n,"\t%d\t%.7lf:%.7lf\t%d\n",
                            pe[i].size,
                            pe[i].price,
                            pe[i+BookLevel].price,
                            pe[i+BookLevel].size);
            }
        }
        return std::string(buf);
    }

    const PriceEntry* getEntryConst(unsigned int level, int side) const {
        return (const PriceEntry*) &(pe[side*BookLevel+level]);
    }

    // this controls wheter a trade is included into bar price's
    // normal trade, i.e. the trade volume and imbalance, etc
    // 0 is 'TSUM' from bpipe, below 20 is all implied trades, i.e.
    // ST, CT etc, above 20 is BLK, EFP, Open, VOL, BASIS, Crack, etc
    // they still will be parsed and written to book queue as a "special
    // trade", i.e. update_type = 3, with type in trade_attr.
    bool isSpecialTrade(int type) const {
        return type>=20;
    }
private:
    void setTradeAttr(int attr) {
        // type in the attr>>8
        trade_attr = attr;
        const int type = attr>>8;
        if (__builtin_expect(!isSpecialTrade(type),1)) {
            // normal
            update_type=TradeUpdateType;
        } else {
            update_type=TradeSpecialUpdateType;
        }
    }

    void setTradeSide(bool is_buy) {
        setTradeAttr(is_buy?TradeDirBuy:TradeDirSell);
    }

    void setSpecialTradeAttr(int type, int normal_attr) {
        // type!=0, a special trade,
        // noamrl_attr is either buy/sell/unknown
        setTradeAttr((normal_attr&0x3)+(type<<8));
    }

    PriceEntry* getEntry(unsigned int level, int side) const {
        return (PriceEntry*) &(pe[side*BookLevel+level]);
    }

};

// Short-cut for backward compatibility, to be removed
//using BookDepot = BookDepotLevel<1>;

/*****************
 * TradeDir
 ****************/
static inline
std::pair<Price, Quantity> bid_reducing(const double prev_px, const uint32_t prev_sz, const double px, const uint32_t sz) {
    // get the px and size for reducing case due to cancel or trade onto bid quotes
    // ask size can be obtained with negative price
    // return the size and the starting price of the reduction
    // For example, if quote goes from [88.2, 10] to [88.3, 2], then 
    // return pair{88.2, 10}
    double ret_px = 0;
    uint32_t ret_sz = 0;
    if (__builtin_expect(prev_px*px!=0,1)) {
        if (std::abs(prev_px-px)<1e-10) {
            // same lvel
            if (sz < prev_sz) {
                ret_px = px;
                ret_sz = prev_sz-sz;
            }
        } else {
            // bid level removed
            if (prev_px-px > 1e-10) {
                // prev_px, prev_sz removed, count as reduce
                ret_px = prev_px;
                ret_sz = prev_sz;
            }
        }
    }
    return std::make_pair(ret_px, ret_sz);
}

template<typename BookType>
static inline
const std::pair<Price, Quantity> getBidReducing (const BookType& book, Price bpx1, Quantity bsz1) {
    Quantity bsz0;
    Price bpx0 = book.getBid(&bsz0);
    return bid_reducing (bpx0, bsz0, bpx1, bsz1);
}

template<typename BookType>
static inline
const std::pair<Price, Quantity> getAskReducing (const BookType& book, Price apx1, Quantity asz1) {
    Quantity asz0;
    Price apx0 = book.getAsk(&asz0);
    const auto [apx_r, asz_r] = bid_reducing (-apx0, asz0, -apx1, asz1);
    return {-apx_r, asz_r};
};

static inline
const BBOTuple getBBOReducing (const BBOTuple& prev_bbo, const BBOTuple& new_bbo) {
    const auto [bpx0, bsz0, apx0, asz0] = prev_bbo;
    const auto [bpx1, bsz1, apx1, asz1] = new_bbo;
    const auto [bpx_r, bsz_r] = bid_reducing (bpx0,bsz0, bpx1,bsz1);
    const auto [apx_r, asz_r] = bid_reducing (-apx0, asz0, -apx1, asz1);
    return {bpx_r, bsz_r, -apx_r, asz_r};
};

template<typename BookType>
static inline
const BBOTuple getBBOReducing (const BookType& book, const BBOTuple& new_bbo) {
    return getBBOReducing(book.getBBOTuple(), new_bbo);
};

template<typename BookDepotType>
class TradeDirection {
public:
    explicit TradeDirection(const BookDepotType& book, const BookConfig& cfg=BookConfig()) :
        book_cfg(cfg),
        cur_book(book), 
        has_pending(false), 
        last_tpx  (std::get<0>(pending_trade)),
        cum_tsz   (std::get<1>(pending_trade)),
        trade_attr(std::get<2>(pending_trade)),
        last_micro(std::get<3>(pending_trade)),
        pending_type(0)
    {
        // check book pointter
        if (book.isValidQuote()) {
            last_micro = book.update_ts_micro;
        }
        reset();
    }

    // use this bbo to see if the previously undecided trade direction
    // can be decided.  In case yes, it returns the decided trade.
    const TradeTuple* updBBO(const BBOTuple& bbo, uint64_t ts_micro) {
        if (__builtin_expect(!has_pending,1)) {
            return nullptr;
        }

        // decide pending trade direction by 
        // checking this quote's reducing from previous quote
        // save at first_book
        TradeTuple* ret = nullptr;
        const auto& [bpx, bsz, apx, asz] = bbo;
        // only get direction from consecutive updates
        if (ts_micro < last_micro + QUOTE_TOO_LATE_MICRO) {
            // undecided direction pending
            // decide a reducing direction from first_book onto this
            const auto [bpx_r, bsz_r, apx_r, asz_r] = getBBOReducing(first_book, bbo);
            if ((bsz_r*asz_r==0)&&(bsz_r+asz_r!=0)) {
                // we have reducing at one and only one side,
                trade_attr=tradeAttr((bsz_r==0),pending_type);

                // debug
                logInfo("DECIDED trade %lf, %d to direction of %d, from quote: %f,%d,%f,%d\nfirst book: %s(%s)", 
                        last_tpx, cum_tsz, trade_attr, 
                        bpx, (int) bsz, apx, (int) asz,
                        first_book.toString().c_str(), 
                        book_cfg.toString().c_str());

                ret = &pending_trade;
            } else {
                // TODO - add a check on double side update, by checking the first_bbo
                if (!BBOBothUpdated(bbo)) {
                    // no update since first_book, check the next
                    return nullptr;
                }
                // debug
                logInfo("NO reducing quotes found, discard the trade %lf, %d, %d.  From quote: %f,%d,%f,%d\nfirst book: %s(%s)",
                        last_tpx, cum_tsz, trade_attr,
                        bpx, (int) bsz, apx, (int) asz,
                        first_book.toString().c_str(), book_cfg.toString().c_str());
            }
        } else {
            // too late
            logInfo("NO consecutive updates, discard trade %lf, %d, %d. From quote: %f,%d,%f,%d\nfirst book: %s(%s)",
                    last_tpx, cum_tsz, trade_attr,
                    bpx, (int) bsz, apx, (int) asz,
                    first_book.toString().c_str(),
                    book_cfg.toString().c_str());
        }
        reset();
        return ret;
    }

    const TradeTuple* updBBO(Price px, Quantity sz, bool is_bid, uint64_t ts_micro) {
        const BBOTuple bt = (is_bid? std::make_tuple(px,sz,(Price)0.0,(Quantity)0):
                                     std::make_tuple((Price)0.0,(Quantity)0,px,sz));
        return updBBO(bt,ts_micro);
    }

    // Return 2 trades: 
    //   first is the pending trade, if any,
    //   second is the current trade, if any.
    // caller is supposed to update pending and current
    // in order if not null
    std::pair<TradeTuple*, TradeTuple*> updTrade(Price px, Quantity sz, uint64_t ts_micro, int cur_type=0) {
        if (__builtin_expect(has_pending,0)) {
            if (ts_micro < last_micro + TRADE_TOO_LATE_MICRO) {
                // undecided direction pending
                if (std::abs(px-last_tpx)<1e-10) {
                    // same trade, adding to the sz
                    cum_tsz += sz;
                    last_micro = ts_micro;
                    // nothing to be updated, all undecided
                    return std::make_pair<TradeTuple*, TradeTuple*>(nullptr, nullptr);
                }

                // buy if px more than last_tpx
                trade_attr = tradeAttr((px>last_tpx), pending_type);

                // debug
                logInfo("DECIDED trade %lf, %d to direction of %d, from trade: %f,%d\nfirst book: %s(%s)",
                    last_tpx, cum_tsz, trade_attr, 
                    px, sz, 
                    first_book.toString().c_str(),
                    book_cfg.toString().c_str());

                // return both pending trade and cur trade
                cur_trade = {px, sz, tradeAttr(cur_book.getTradeAttr(px),cur_type), ts_micro};
                reset();
                return std::make_pair(&pending_trade, &cur_trade);
            } else {
                // debug
                logInfo("NO consecutive updates, discard trade %lf, %d, %d. From trade: %f,%d,micro(%lld)\nfirst book: %s(%s)",
                    last_tpx, cum_tsz, trade_attr,
                    px, sz, (long long)ts_micro,
                    first_book.toString().c_str(),
                    book_cfg.toString().c_str());
            }
            // discard any pending trades and fall through
            reset();
        }

        trade_attr = tradeAttr(cur_book.getTradeAttr(px), cur_type);

        if (__builtin_expect(trade_attr & 0x2, 0)) {
            // ignore the case where quotes are not valid
            if (__builtin_expect(cur_book.isValidQuote(),1)) {
                setup_pending(px, sz, ts_micro, cur_type);
            }
            // no previous pending, but the current trade starts a pending
            return std::make_pair<TradeTuple*, TradeTuple*>(nullptr, nullptr);
        }
        cur_trade = {px, sz, trade_attr, ts_micro}; 
        return std::make_pair<TradeTuple*, TradeTuple*>(nullptr, &cur_trade);
    }

    /*
     * update with a given direction. In case of undecided, enters the pending.
     * used for td_parser, for those matched/unmatched trades
     */
    std::pair<TradeTuple*, TradeTuple*> updTrade_Dir(Price px, Quantity sz, uint64_t ts_micro, bool is_buy, int cur_type=0) {
        // this is the case where a direction is decided before entering
        // the TradeDirection, i.e. by td_parser
        TradeTuple* tt0 = nullptr;
        if (__builtin_expect(has_pending, 0)) {
            // in case of pending, if this px is different
            // than pending px, tt1 is returned, otherwise,
            // use this direction as pending direction
            const auto prev_micro = last_micro;
            auto [tt1, tt2] = updTrade(px, sz, ts_micro);
            // tt1 can be null if 1) too late, 
            // 2) same px, in which case tt2 is null as well
            if ((tt1 == nullptr) && (tt2==nullptr)) {
                // pending price is same with px
                // take this direct as pending direction
                trade_attr = tradeAttr(is_buy, pending_type);
                cum_tsz -= sz;
                last_micro = prev_micro;
                tt1=&pending_trade;
            }
            tt0 = tt1;
            // debug
            if (tt0) {
                logInfo("updTrade_Dir decided pending as %lf, %d, %d, from trade %f, %d, %s",
                        last_tpx, cum_tsz, trade_attr,
                        px, sz, is_buy?"Buy":"Sell");
            }
            reset();
        }
        cur_trade = {px, sz, tradeAttr(is_buy, cur_type), ts_micro};
        return std::make_pair(tt0, &cur_trade);
    }

    std::pair<TradeTuple*, TradeTuple*> updTrade_NoDir(Price px, Quantity sz, uint64_t ts_micro, int cur_type=0) {
        // if pending, do enter as a price and then enter new
        // otherwise, just force enter as unknown. Literally
        // the first half of updTrade()
        TradeTuple* tt0 = (TradeTuple*) nullptr;
        if (__builtin_expect(has_pending,0)) {
            if (ts_micro < last_micro + TRADE_TOO_LATE_MICRO) {
                // undecided direction pending
                if (std::abs(px-last_tpx)<1e-10) {
                    // same trade, adding to the sz
                    cum_tsz += sz;
                    last_micro = ts_micro;
                    // nothing to be updated, all undecided
                    return std::make_pair<TradeTuple*, TradeTuple*>(nullptr, nullptr);
                }

                // buy if px more than last_tpx
                trade_attr = tradeAttr((px>last_tpx), cur_type);

                // debug
                logInfo("updTrade_NoDir DECIDED trade %lf, %d to direction of %d, from trade: %f,%d\nfirst book: %s(%s)",
                    last_tpx, cum_tsz, trade_attr, 
                    px, sz, 
                    first_book.toString().c_str(),
                    book_cfg.toString().c_str());

                // save the previous pending to the "cur_trade"
                // because we need pending_trade for this one.
                // Note, be sure not to modify cur_trade!
                cur_trade = pending_trade;
                tt0 = (TradeTuple*)(&cur_trade);
            } else {
                // debug
                logInfo("updTrade_NoDir too Late, discard trade %lf, %d, %d. From trade: %f,%d,micro(%lld)\nfirst book: %s(%s)",
                    last_tpx, cum_tsz, trade_attr,
                    px, sz, (long long)ts_micro,
                    first_book.toString().c_str(),
                    book_cfg.toString().c_str());
            }
        }

        // start a new pending with this one
        // previous pending is stored at tt0, using
        // cur_trade, pending_trade updated and NOT returned
        setup_pending(px, sz, ts_micro, cur_type);
        return std::make_pair(tt0, nullptr);
    }
    bool hasPending() const { return has_pending; };

    void reset() {
        has_pending = false;
        bbo_upd_bits = 0;
    }

private:
    enum { QUOTE_TOO_LATE_MICRO = 5000, TRADE_TOO_LATE_MICRO = 5000};
    const BookConfig book_cfg;
    const BookDepotType& cur_book; // the current book of BookWriter
    BookDepotType first_book; // a copy of book before first trade
    BBOTuple first_bbo;        // a copy of bbo of first_book, used to decide the two sided bbo
    bool has_pending;
    int bbo_upd_bits; // in case has_pending, bits on bid/ask quote updated: 0/1 for bid/ask
    TradeTuple pending_trade;  // holds the undecided trades
    TradeTuple cur_trade;      // holds the current trade
    Price& last_tpx;
    Quantity& cum_tsz;
    uint32_t& trade_attr; // 0,1: buy/sell
    uint64_t& last_micro;
    int pending_type;

    void setup_pending(Price px, Quantity sz, uint64_t ts_micro, int type) {
        first_book = cur_book;
        first_bbo = cur_book.getBBOTuple();
        last_tpx=px;
        cum_tsz=sz;
        last_micro=ts_micro;
        has_pending=true;
        pending_type=type;
    }

    bool BBOBothUpdated(const BBOTuple& bbo) {
        if (__builtin_expect(bbo_upd_bits<3,1)) {
            const auto [bpx, bsz, apx, asz]=bbo;
            const auto& [bpx0, bsz0, apx0, asz0]=first_bbo;
            const double bps = bpx*bsz, aps=apx*asz;
            if ((bps!=0) && (std::abs(bps-bpx0*bsz0)>1e-10)) {
                bbo_upd_bits |= 1;
            }
            if ((aps!=0) && (std::abs(aps-apx0*asz0)>1e-10)) {
                bbo_upd_bits |= 2;
            }
            return bbo_upd_bits == 3;
        };
        return true;
    }

    int tradeAttr(int attr, int type) {
        return ((type<<8) + (attr&0x3));
    }

    int tradeAttr(bool is_buy, int type) {
        return tradeAttr(is_buy?0:1, type);
    }

    // disallow copy - need to fix the references
    TradeDirection(const TradeDirection& trd_dir) = delete;
    void operator==(const TradeDirection& trd_dir) = delete;
};

/*****************
 * BookWriter 
 ****************/

// BookWriter is a passive object, driven by quote and trade
// updates from, say Live or historical feeds, and publish 
// the built book onto the OutputType. 
//
// It is a wrapper of BookDepotLevel<>, for update
// Quotes and Trades from say, live L1/L2 updates and publish the
// result state (the book) to OutType, which can be Shared Meory
// Queue, Bar Writer, or just a file, etc. a
//
// When updating Book, it has additional logic for getting trade direction.
// It also allows a Output for publish the result. 
//
// It holds a book of a given tradable contract, i.e. WTI_N1. 
template <typename OutType, int BookLevel=1>
class BookWriter {
public:
    const BookConfig _cfg;
    using BookType = BookDepotLevel<BookLevel>; // level 2 book
    explicit BookWriter(std::shared_ptr<OutType>& out, const BookConfig& cfg=BookConfig())
    : _cfg(cfg), _book(), _trade_dir(_book, _cfg), _out(out)
    {
        reset();
    }

    std::string toString() const {
        char buf[1024];
        int n = 0;
        n += snprintf(buf+n, sizeof(buf)-n, "Book %s { %s }", _cfg.toString().c_str(), _book.toString().c_str());
        return std::string(buf);
    }

    /******************
     * BBO quotes (L1)
     *****************/

    // return the status on output to _out
    // NOTE - _book.updBBO returns true if there is updates to be updated
    bool updBBO(Price price, Quantity size, bool is_bid, uint64_t ts_micro) {
        bool ret=true;
        const TradeTuple* pending_trade = _trade_dir.updBBO(price, size, is_bid, ts_micro);
        if (__builtin_expect(pending_trade!=nullptr, 0)) {
            ret=updateTradeTuple(*pending_trade);
        }
        if (__builtin_expect(_book.updBBO(price, size, is_bid, ts_micro), 1)) {
            ret &= updateOut(ts_micro);
        }
        return ret;
    }

    bool updBBO(const BBOTuple& bbo, uint64_t ts_micro) {
        bool ret = true;
        const TradeTuple* pending_trade = _trade_dir.updBBO(bbo, ts_micro);
        if (__builtin_expect(pending_trade!=nullptr, 0)) {
            ret = updateTradeTuple(*pending_trade);
        }
        const auto [bpx, bsz, apx, asz] = bbo;
        if (__builtin_expect(_book.updBBO(bpx, bsz, true, ts_micro), 1)) {
            ret &= updateOut(ts_micro);
        }
        if (__builtin_expect(_book.updBBO(apx, asz, false, ts_micro), 1)) {
            ret &= updateOut(ts_micro);
        }
        return ret;
    }

    bool updBBO(Price bp, Quantity bsz, Price ap, Quantity asz, uint64_t ts_micro) {
        const BBOTuple bbo ({bp,bsz,ap,asz});
        return updBBO(bbo, ts_micro);
    }

    /****************
     * Trade updates
     ****************/

    // Trade Normal, direction to be decided
    // this trade's direction to be inferred from _trade_dir
    bool updTrade(Price price, Quantity size, uint64_t ts_micro, int type=0) {

        // DEBUG
        /*
        if (type>0 && type<20) {
            printf("Here!");
        }*/

        if (__builtin_expect(!_book.isSpecialTrade(type), 1)) {
            const auto& trd_pair = _trade_dir.updTrade(price, size, ts_micro, type);
            return updateTradeTuplePair(trd_pair);
        }
        return updTrade_Special(price, size, type, ts_micro);
    }

    bool updTrade(Price price, Quantity size) {
        return updTrade(price, size, utils::TimeUtil::cur_micro());
    };

    // Normal Trade, direction given
    // the direction decided by i.e. td_parser
    // Note, still check _trade_dir for previous pending trades, if any
    bool updTrade_Dir(Price price, Quantity size, bool is_buy, uint64_t ts_micro, int type=0) {
        const auto& trd_pair = _trade_dir.updTrade_Dir(price, size, ts_micro, is_buy, type);
        return updateTradeTuplePair(trd_pair);
    }

    // Normal Trade, direction given as unknown (won't try to infer this one)
    // Note, check _trade_dir for previous pending trades if any, 
    // and start a new pending trade 
    bool updTrade_NoDir(Price price, Quantity size, uint64_t ts_micro, int type=0) {
        const auto& trd_pair = _trade_dir.updTrade_NoDir(price, size, ts_micro, type);
        return updateTradeTuplePair(trd_pair);
    }

    // Special trade
    // Recorded as is, won't go through _trade_dir.
    // direction can be set by 'normal_attr' optionally (0-buy, 1-sell, 2-unknown(default))
    bool updTrade_Special(Price price, Quantity size, int type, uint64_t ts_micro, int normal_attr=BookType::TradeDirUnknown) {
        _book.addSpecialTrade(type, price, size, normal_attr, ts_micro);
        return updateOut(ts_micro);
    }

    /*************************
     * Level 2 Quote updates
     *************************/

    // Update TradeDir if at level 0
    // Otherwise, use _book's L2 Price setter and updateOut
    bool newPrice(Price price, Quantity size, unsigned int level, bool is_bid, uint64_t ts_micro) {
        if (level==0) {
            return updBBO(price, size, is_bid, ts_micro);
        }
        // no trade dir
        if (__builtin_expect(_book.newPrice(price, size, level, is_bid, ts_micro), 1)) {
            return updateOut(ts_micro);
        }
        return false;
    }

    bool delPrice(unsigned int level, bool is_bid, uint64_t ts_micro) {
        bool ret=true;
        if (level==0) {
            const int side=is_bid?0:1;
            if ((_book.avail_level[side]>1)) {
                const PriceEntry* pe = _book.getEntryConst(1, side);
                const TradeTuple* pending_trade = _trade_dir.updBBO(
                        pe->price, pe->size, is_bid, ts_micro);
                if (__builtin_expect(pending_trade!=nullptr, 0)) {
                    ret=updateTradeTuple(*pending_trade);
                }
            }
        }
        if (__builtin_expect(_book.delPrice(level, is_bid, ts_micro), 1)) {
            ret &= updateOut(ts_micro);
        }
        return ret;
    }

    bool updPrice(Price price, Quantity size, unsigned int level, bool is_bid, uint64_t ts_micro) {
        if (level==0) {
            return updBBO(price, size, is_bid, ts_micro);
        }
        // no trade dir
        if (__builtin_expect(_book.updPrice(price, size, level, is_bid, ts_micro), 1)) {
            return updateOut(ts_micro);
        }
        return false;
    }

    void reset() {
        _book.reset();
        _trade_dir.reset();
        _out->reset();
    }

    std::shared_ptr<const OutType> getOut() const {
        return _out;
    }

private:
    BookType _book;
    TradeDirection<BookType> _trade_dir;
    std::shared_ptr<OutType> _out; // ShmQ, BarWriter, or TickStream, etc
                   
    bool updateOut(uint64_t ts_micro) {
        if (__builtin_expect(_book.isValid(), 1)) {
            _book.update_ts_micro = ts_micro;
            return _out->put(_book);
        }
        return false;
    }

    bool updateTradeTuple(const TradeTuple& trd_tuple) {
        _book.addTrade(trd_tuple);
        const auto ts_micro = std::get<3>(trd_tuple);
        return updateOut(ts_micro);
    }

    bool updateTradeTuplePair(const std::pair<TradeTuple*, TradeTuple*> & tuple_pair) {
        bool ret = true;
        const auto [pending_trd, cur_trd] = tuple_pair;
        if (__builtin_expect(pending_trd!=nullptr,0)) {
            ret &= updateTradeTuple(*pending_trd);
        }
        if (__builtin_expect(cur_trd!=nullptr,1)) {
            ret &= updateTradeTuple(*cur_trd);
        }
        return ret;
    }

    // disallow copy
    void operator==(const BookType& b1) = delete;
    BookWriter(const BookWriter& b2) = delete;
};
}  // namespace md
