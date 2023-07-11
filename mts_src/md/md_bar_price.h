#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <array>
#include <atomic>
#include <unordered_map>

#include "md_snap.h"

namespace md {

struct BarPrice {
public:
    time_t bar_time; // the close time of the bar
    double open;
    double high;
    double low;
    double close;
    uint32_t bvol;
    uint32_t svol;
    long long last_micro; // last trade update time
    double last_price;

    // extended fields
    int bqd; // bid quote diff
    int aqd; // ask quote diff

    double avg_bsz(long long cur_micro) const {
        return get_avg(cur_micro, cumsum_bsz, prev_bsz);
    }
    double avg_asz(long long cur_micro) const {
        return get_avg(cur_micro, cumsum_asz, prev_asz);
    }
    double avg_quote_sz(long long cur_micro) const {
        const auto bsz=avg_bsz(cur_micro);
        const auto asz=avg_asz(cur_micro);
        if (__builtin_expect(std::abs(bsz*asz) > 1,1)) {
            return (bsz+asz)/2;
        }
        const auto sz = _MAX_(asz,bsz);
        return _MAX_(sz,1);
    }

    double avg_spd(long long cur_micro) const {
        return get_avg(cur_micro, cumsum_spd, (prev_apx-prev_bpx));
    }

private:
    // weighted avg and state - calculated on the spot
    double cumsum_bsz; // cumsum bid size, for time weighted-avg
    double cumsum_asz; // cumsum ask size, for time weighted-avg
    double cumsum_spd; //  cumsum spread, for time weighted-avg

    // states 
    long long cumsum_micro;         // total micros in cumsum
    long long prev_micro_quote;  // last quote update time
    double prev_bpx, prev_apx;
    int prev_bsz, prev_asz;
    int prev_type;

    //optional volume
    bool write_optional; //for unmatched volumes
    int32_t opt_v1; // signed swipe trade size
    int32_t opt_v2; // signed unmatched trade size due to i.e. hiddne orders
    int32_t opt_v3; // cumsum trd_size from consecutive trade at same dir
    
    // optional tick_size for calculating swipe levels
    double tick_size;

    template<typename DType1, typename DType2>
    double get_avg(long long cur_micro, DType1 cum_qty, DType2 cur_qty) const {
        // assuming bar valid
        long long tau = cur_micro - prev_micro_quote;
        if (__builtin_expect(tau + cumsum_micro <= 0,0)) {
            tau = 1; // allow same micro with prev_micro_quote
        }
        return (double)(cum_qty + cur_qty * tau)/(double)(cumsum_micro+tau);
    }

    void upd_state(long long cur_micro, double bid_px, int bid_sz, double ask_px, int ask_sz) {
        // assumes time not going back
        if (__builtin_expect(!isValid(), 0)) {
            // note initial update should work since
            // prev_px/sz all zero
            prev_micro_quote = cur_micro;
        }
        long long tau (cur_micro-prev_micro_quote);
        if (__builtin_expect(tau>0 ,1)) {
            cumsum_bsz += (prev_bsz*tau);
            cumsum_asz += (prev_asz*tau);
            cumsum_spd += ((prev_apx-prev_bpx)*tau);
            cumsum_micro += tau;
            prev_micro_quote = cur_micro;
        }
        prev_bpx=bid_px; prev_bsz=bid_sz; prev_apx=ask_px; prev_asz=ask_sz;
    }

    void roll_state(long long cur_micro) {
        if (__builtin_expect(!isValid(),0)) {
            // don't roll if no update received
            return;
        }
        cumsum_bsz = 0; cumsum_asz = 0; cumsum_spd = 0;
        cumsum_micro = 0;
        prev_micro_quote = cur_micro;
    }

    void init( time_t utc_, double open_, double high_,
             double low_, double close_, uint32_t bvol_, uint32_t svol_, 
             long long last_micro_, double last_price_,
             double bid_sz_, double ask_sz_, double avg_spread_,
             int bqt_diff_, int aqt_diff_) {
        // simple assignments
        bar_time=utc_; // closing time of the bar in utc second
        open=open_; 
        high=high_; 
        low=low_; 
        close=close_;
        bvol=bvol_; 
        svol=svol_; 
        last_micro=last_micro_; // last trade time
        last_price =last_price_; 
        if (close == 0) { close = last_price; };

        // setup the ext and state to be ready for update/roll
        bqd=bqt_diff_; 
        aqd=aqt_diff_; 
        prev_type=2; // don't want to upd quote diff yet

        prev_bpx=(close==0?1000:close)-avg_spread_/2 ;
        prev_apx=(close==0?1000:close)+avg_spread_/2 ;
        prev_bsz=bid_sz_; prev_asz=ask_sz_;

        if (open*high*low*close!=0) {
            prev_micro_quote = bar_time*1000000LL;
        }
        roll_state((long long)bar_time*1000000LL);
    }

    int get_swipe_level(double prev_px, double px) const {
        if (tick_size ==0) return 1;
        int levels = (int)(std::abs(prev_px-px)/tick_size+0.5);
        if (__builtin_expect(levels<1,0)) levels=1;
        if (__builtin_expect(levels>10,0)) levels=10;
        return levels;
    }

public:
    BarPrice() {
        memset((char*)this, 0, sizeof(BarPrice));
        high = -1e+12;
        low = 1e+12;
        write_optional = false;
    }

    BarPrice(time_t utc_, double open_, double high_,
             double low_, double close_, uint32_t bvol_ = 0, uint32_t svol_=0, 
             long long last_micro_ = 0, double last_price_ = 0,
             double bid_sz_ = 0, double ask_sz_ = 0, double avg_spread_ = 0,
             int bqt_diff_ = 0, int aqt_diff_ = 0) {
        memset((char*)this, 0, sizeof(BarPrice));
        high = -1e+12;
        low = 1e+12;
        write_optional = false;
        init(utc_, open_, high_, low_, close_, bvol_, svol_, 
                last_micro_, last_price_, bid_sz_, ask_sz_, avg_spread_, 
                bqt_diff_, aqt_diff_);
    }

    BarPrice(const std::string& csvLine) {
        // format is utc, open, high, low, close, totvol, lastpx, last_micro, vbs
        // extended: avg_bsz, avg_asz, avg_spd, bqdiff, aqdiff
        
        memset((char*)this, 0, sizeof(BarPrice));
        high = -1e+12;
        low = 1e+12;
        write_optional = false;

        auto tk = utils::CSVUtil::read_line(csvLine);
        auto bar_time_ = (time_t)std::stoi(tk[0]);
        auto open_ = std::stod(tk[1]);
        auto high_ = std::stod(tk[2]);
        auto low_ = std::stod(tk[3]);
        auto close_ = std::stod(tk[4]);
        long long totval_ = std::stoll(tk[5]);
        auto last_price_ = std::stod(tk[6]);
        auto last_micro_ = std::stoll(tk[7]);
        long long vbs_ = std::stoll(tk[8]);

        auto bvol_ = (uint32_t) ((totval_ + vbs_)/2);
        auto svol_ = (uint32_t) (totval_ - bvol_);

        long long bsz_ = 0, asz_ = 0;
        double spd_ = 0;
        int bqd_ = 0, aqd_ = 0;
        if (tk.size() > 9) {
            bsz_ = (uint32_t) (std::stod(tk[9])+0.5);
            asz_ = (uint32_t) (std::stod(tk[10])+0.5);
            spd_ = std::stod(tk[11]);
            bqd_ = int(std::stod(tk[12])+0.5);
            aqd_ = int(std::stod(tk[13])+0.5);
        }
        init(bar_time_, open_, high_, low_, close_, bvol_, svol_,
                last_micro_, last_price_, bsz_, asz_, spd_,
                bqd_, aqd_);
        // optional unmatched 
        if (tk.size() > 14) {
            opt_v1 = std::stoi(tk[14]);
            opt_v2 = std::stoi(tk[15]);
            write_optional = true;
        }
    }

    std::string toCSVLine(time_t bar_close_time=0) const {
        // this writes a mts repo line from current bar
        if (__builtin_expect(bar_close_time == 0, 1)) {
            bar_close_time = bar_time; // the previous close, for read in bars
        };
        long long cur_micro = (long long)bar_close_time*1000000LL;
        char buf[256];
        size_t cnt = snprintf(buf, sizeof(buf), "%d, %s, %s, %s, %s, %lld, %s, %lld, %lld"
                                   ", %.1f, %.1f, %f, %d, %d",
                (int) bar_close_time, PriceCString(open), PriceCString(high),
                PriceCString(low), PriceCString(close), (long long)bvol + svol,
                PriceCString(last_price), last_micro, (long long)bvol - svol,
                avg_bsz(cur_micro), avg_asz(cur_micro), avg_spd(cur_micro), bqd, aqd
                );
        if (__builtin_expect(write_optional,0)) {
            snprintf(buf+cnt, sizeof(buf)-cnt, ", %d, %d", opt_v1, opt_v2);
        }
        return std::string(buf);
    };

    std::string toString(long long cur_micro = 0) const {
        return toCSVLine(cur_micro);
    }

    bool isValid() const {
        return prev_micro_quote>0;
    }

    // utilities for update this data structure
    std::string writeAndRoll(time_t bar_close_time) {
        // this is an atomic action to write the existing 
        // state and then roll forward
        bar_time = bar_close_time;
        std::string ret = toCSVLine();

        // roll forward
        open = close;
        high = close;
        low = close;
        bvol = 0; svol = 0;
        bqd = 0;  aqd = 0;
        // optional write - unmatched trade volumes
        opt_v1 = 0; opt_v2 = 0;
        roll_state((long long)bar_close_time*1000000LL);
        return ret;
    }

    void update(long long cur_micro, double last_trd_price, int32_t volume, int update_type,
                double bidpx, int bidsz, double askpx, int asksz) {
        // save the previous states, they may be modified during the update
        auto prev_bpx0=prev_bpx;
        auto prev_apx0=prev_apx;
        auto prev_bsz0=prev_bsz;
        auto prev_asz0=prev_asz;
        auto prev_type0=prev_type;

        // update at this time with type of update
        // type: 0 - bid update, 1 - ask update, 2 - trade update
        if (__builtin_expect(update_type>2,0)) {
            // TODO - special trade to be added here - maybe a vwap of price plus size
            return;
        }
        if (__builtin_expect(update_type==2,0)) {
    	    // trade update
            if (volume > 0) 
                bvol += volume;
            else 
                svol -= volume;
            last_price = last_trd_price;
            last_micro = cur_micro;
            close = last_trd_price;
            opt_v3 += volume;
        } else {
            // check for cross
            if (__builtin_expect(askpx-bidpx<1e-10,0)) {
                // remove crossed ticks
                return;
            }

            /* logic for updating a quote after a trade:
             * if the quote is immediately after a trade, then
             * it is taken as a response to the trade. In this
             *   case the following situations are considered
             *   - if staying same level with previous quote:
             *     - quote reducing at the trade side, comare with 
             *       the previous trades
             *       * reducing less than trade size, opt_v2 for iceberg
             *       * reducing more than trade size, (very unlikely) add to qbs
             *   - if staying at a different level:
             *     - swipe case:
             *       adjust qbs and opt_v1 using estimated L2 cancels compared with trade size
             *     - reverse (fall down to the qbs adjust, see below)
             *   if the quote diff are not applied, fall down for the qbs accounting
             */
            close = (bidpx+askpx)/2.0;
            if (__builtin_expect(write_optional&&(prev_type == 2), 0)) {
                // check unmatched volume
                int32_t r_sz = 0;
                if (prev_bsz*prev_asz*opt_v3!=0) {
                    if (opt_v3>0) {
                        if (std::abs(prev_apx-askpx)<1e-10) {
                            // buy without apx change
                            ////r_sz = bid_reducing(-prev_apx, prev_asz, -askpx, asksz).second;
                            r_sz = prev_asz-asksz;
                            if (__builtin_expect(r_sz < 0, 0)) {
                                // mismatched quote/trade, adjust qbs, opt_v1
                                aqd -= r_sz;
                                opt_v2+= opt_v3;
                            } else {
                                if (__builtin_expect(opt_v3 >= r_sz,1)) {
                                    opt_v2 += (opt_v3-r_sz);
                                } else {
                                    aqd -= (r_sz - opt_v3);
                                }
                            }
                            // fall through to qbs but don't process ask side anymore
                            prev_asz0 = asksz;
                            prev_apx0 = askpx;
                        } else {
                            if (__builtin_expect(askpx-prev_apx>1e-10, 1)) {
                                // buy swipe a level
                                int levels = get_swipe_level(prev_apx,askpx);
                                r_sz = (prev_asz+(int)((levels-1)*avg_quote_sz(cur_micro)+0.5));

                                // account for the opt_v1 - swipe trade size
                                int sz_l2 = opt_v3-prev_asz; // swipe size as trade size
                                opt_v1 += (sz_l2>0?sz_l2:0); // minus prev l1 size

                                // adjust aqd in case ask reduced size more than trade size
                                // Note if the other way, nothing can be inferred for aqd,
                                // because we don't know the initial size of new level
                                if (r_sz>opt_v3) {
                                    aqd -= (r_sz-opt_v3);
                                }

                                // fall through to qbs but don't process ask side anymore
                                prev_asz0 = asksz;
                                prev_apx0 = askpx;
                            } else {
                                // this ask price reduces on buy trade?
                                //// aqd += opt_v3;
                                // fall through
                            }
                        }
                        prev_type0=0;
                    } else if (__builtin_expect(opt_v3 < 0,1)) {
                        opt_v3=-opt_v3;
                        if (std::abs(prev_bpx-bidpx)<1e-10) {
                            r_sz = prev_bsz-bidsz;
                            if (__builtin_expect(r_sz < 0, 0)) {
                                bqd -= r_sz;
                                opt_v2-=opt_v3;
                            } else {
                                if (__builtin_expect(opt_v3 >= r_sz,1)) {
                                    opt_v2 -= (opt_v3-r_sz);
                                } else {
                                    bqd -= (r_sz - opt_v3);
                                }
                            }
                            prev_bsz0 = bidsz;
                            prev_bpx0 = bidpx;
                        } else {
                            if (__builtin_expect(bidpx-prev_bpx<-1e-10, 1)) {
                                int levels = get_swipe_level(prev_bpx,bidpx);
                                r_sz = (prev_bsz+(int)((levels-1)*avg_quote_sz(cur_micro)+0.5));
                                int sz_l2 = opt_v3-prev_bsz; // size into l2
                                opt_v1 -= (sz_l2>0?sz_l2:0);
                                if (r_sz>opt_v3) {
                                    bqd -= (r_sz-opt_v3);
                                }
                                prev_bsz0 = bidsz;
                                prev_bpx0 = bidpx;
                            } else {
                                // bqd -= opt_v3;
                            }
                        }
                        prev_type0=1;
                    }
                }
            }
            opt_v3 = 0;
        }
        if (__builtin_expect(!isValid(),0)) {
            open=close; high=close; low=close;
        }
        if (close > high) high = close;
        if (close < low)  low = close;

        // extended fields
        //
        // update the bqd/aqd -
        // bpipe sends a quote tick after any trade, reflecting
        // quote size reduction due to the trade. We don't want
        // the bid/ask quote diff to also include trade,
        // so kip if prev_update is a trade. This assumes 
        // strict sequential update from book queue, using getNextUpdate().
        // It is the case see the writer thread.
        // Note It fails if by latest snapshot, i.e.getLatestUpdate()
        if (__builtin_expect((update_type != 2)&&(prev_type0 != 2)&&(prev_bsz0*prev_asz0 != 0),1)) {
            // get bid quote diff
            if (__builtin_expect(std::abs(bidpx - prev_bpx0) < 1e-10,1)) {
                // still the same level
                bqd += (bidsz - prev_bsz0);
            } else {
                // levels centered at 0, to be flipped and rounded properly
                int levels = get_swipe_level(prev_bpx0,bidpx);
                if (bidpx < prev_bpx0) {
                    // prev_bpx0 no more
                    bqd -= (prev_bsz0+(int)((levels-1)*avg_quote_sz(cur_micro)+0.5));//cancel prev_bsz0+(level-1)*avg_bsz
                } else {
                    // a new level
                    bqd += (bidsz+(int)((levels-1)*avg_quote_sz(cur_micro)+0.5));  //adding bidsz+(level-1)*avg_bsz
                }
            }
            // get ask quote diff
            if (__builtin_expect(std::abs(askpx - prev_apx0) < 1e-10,1)) {
                aqd += (asksz - prev_asz0);
            } else {
                int levels = get_swipe_level(prev_apx0,askpx);
                if (askpx > prev_apx0) {
                    // prev_apx0 no more
                    aqd -= (prev_asz0+(int)((levels-1)*avg_quote_sz(cur_micro)+0.5));
                } else {
                    // a new level
                    aqd += (asksz+(int)((levels-1)*avg_quote_sz(cur_micro)+0.5));
                }
            }
        }

        // update the average bid/ask size and spread -
        if (__builtin_expect(!isValid() || (update_type != 2), 1)) {
            upd_state(cur_micro, bidpx, bidsz, askpx, asksz);
        }
        prev_type = update_type;
    }

    template<int BookLevel>
    void update(const BookDepotLevel<BookLevel>& book) {
        if (__builtin_expect(!book.isValid(), 0)) {
            // skip invalid book
            return;
        }
        //this is called by updateState() from barWriter
        uint64_t upd_micro = book.update_ts_micro;
        auto [bpx, bsz, apx, asz] = book.getBBOTupleUnSafe();
        int update_type = book.update_type;
        Quantity volume = book.getTradeVolumeSigned();
        double px = book.trade_price;
        update(upd_micro, px, volume, update_type, bpx, bsz, apx, asz);
    }

    // used to replay from the tick-by-tick file saved by 
    // booktap -o, refer to the br_test.cpp
    template<int BookLevel>
    void update(const std::vector<std::string>& bookdepot_csv_line) {
        // doesn't hurt to have a L2 Book to hold L1
        update(BookDepotLevel<BookLevel>(bookdepot_csv_line));
    }

    // Shortcut to call bar update from l1-bbo-quote and trade, from i.e. tickdata quote/trade
    // Also possible to get a BookDepot to updated from such quote/trade, and then update
    // bar with update(const md::BookDepot).  Although it is slower.
    void updateQuote(long long cur_micro, double bidpx, int bidsz, double askpx, int asksz) {
        int update_type = 0;
        if (std::abs(prev_apx*prev_asz-askpx*asksz)>1e-10) {
            update_type=1;
        }
        update(cur_micro, 0, 0, update_type, bidpx, bidsz, askpx, asksz);
    }

    // Normal trades
    void updateTrade(long long cur_micro, double price, uint32_t size, bool is_buy) {
        int update_type = 2;
        update(cur_micro, price, (int32_t)is_buy?size:-(int32_t)size, update_type, prev_bpx, prev_bsz, prev_apx, prev_asz);
    }

    void set_write_optional(bool if_write) { write_optional = if_write; };
    void set_tick_size(double tick_size_) { tick_size = tick_size_; };
};

/*
 * Used to get Live BarPrice
 */
class BarReader{
public:
    BarReader(const BookConfig& bcfg_, int barsec_)
    : bcfg(bcfg_), barsec(barsec_), fn(bcfg.bfname(barsec)), fp(nullptr)
    {
        fp = fopen(fn.c_str(), "rt");
        if (!fp) {
            logError("failed to open bar file %s", fn.c_str());
            throw std::runtime_error(std::string("faile to open bar file ") + fn);
        }
        bp = getLatestBar();
    }

    bool read(BarPrice& bar) {
        bar = getLatestBar();
        if (bar.isValid()) {
            bp = bar;
            return true;
        }

        // no new bar
        bar = bp;
        return false;
    }

    bool readLatest(std::vector<std::shared_ptr<BarPrice> >& bars, int barcnt) {
        // read latest barcnt bars upto barcnt
        // Note the vector bars is appended with the new bars
        // The new bars are forward/backward filled to the barsec
        // See the forwardBackwardFill()
        if (barcnt <= 0) {
            return false;
        }
        BarPrice bar;
        read(bar);
        if (!bar.isValid()) {
            return false;
        }
        time_t end_bartime = bar.bar_time;
        time_t start_bartime = bartimeByOffset(end_bartime, -(barcnt-1));
        return readPeriod(bars, start_bartime, end_bartime);
    }

    bool readPeriod(std::vector<std::shared_ptr<BarPrice> >& bars, time_t start_bartime, time_t end_bartime) const {
        // getting bars from start_bartime to end_bartime, inclusive
        // Note the bars are appended to the given vector of bars

        if (start_bartime > end_bartime) {
            logError("bad period given to readPeriod: %lu - %lu", 
                    (unsigned long) start_bartime, (unsigned long) end_bartime);
            return false;
        }

        // times have to be a bar time
        if ((start_bartime/barsec*barsec != start_bartime) ||
            (end_bartime/barsec*barsec != end_bartime)) {
            logError("start_time %d or end_time %d not a bar time of %d", 
                    (int) start_bartime, (int) end_bartime, barsec);
            return false;
        }

        FILE* fp_ = fopen(fn.c_str(), "rt");
        if (!fp_) {
            logError("Failed to read bar file %s", fn.c_str());
            return false;
        }
        // read bars 
        char buf[256];
        buf[0] = 0;
        std::vector<std::shared_ptr<BarPrice> > allbars;
        try {
            while (fgets(buf, sizeof(buf)-1, fp_)){
                allbars.emplace_back(std::make_shared<BarPrice>(buf));
            }
        } catch (const std::exception & e) {
            logError("failed to get last bar price from %s: %s", fn.c_str(), e.what());
        }
        fclose(fp_);

        // forward and backward fill
        return forwardBackwardFill(allbars, start_bartime, end_bartime, bars);
    }

    // utilities
    bool forwardBackwardFill(const std::vector<std::shared_ptr<BarPrice> >& allbars,
                             time_t start_bartime,
                             time_t end_bartime,
                             std::vector<std::shared_ptr<BarPrice> >& bars) const
    {
        // forward and backward fill the bars for given start_bartime and end_bartime
        // if missing starting bars, the first bar available is backward filled with
        // its open price
        // if missing bars afterwards, they are forward filled by the previous bar
        // Bars outside of trading time of the venue are not included in the return.
        // Returns true if no bars are backward filled
        // Otherwise false
        if (end_bartime < start_bartime) {
            logError("bad period given in forwardBackwardFill: %lu - %lu",
                    (unsigned long) start_bartime, (unsigned long) end_bartime);
            return false;
        }

        // times have to be a bar time
        if ((start_bartime/barsec*barsec != start_bartime) ||
            (end_bartime/barsec*barsec != end_bartime)) {
            logError("start_time %d or end_time %d not a bar time of %d", 
                    (int) start_bartime, (int) end_bartime, barsec);
            return false;
        }

        // remove all bars outside of venue's trading time
        std::vector<std::shared_ptr<BarPrice> > allb;
        for (auto& bp: allbars) {
            // this is a hack to avoid excluding the last bar at end time
            // and to avoid including first bar at start time
            time_t bar_time_ = bp->bar_time-1;
            if (VenueConfig::get().isTradingTime(bcfg.venue, bar_time_)) {
                allb.emplace_back(bp);
            }
        }

        if (allb.size() < allbars.size()) {
            logInfo("forwardBackwardFill removed %d bars outside trading hour of %s", 
                    (int) (allbars.size() - allb.size()), bcfg.toString().c_str());
        }
        if (allb.size() == 0) {
            logError("No bars found (allb vector is empty)!");
            return false;
        }

        size_t bcnt = 0;
        auto bp0 = allb[0];
        time_t bt = start_bartime;
        BarPrice fill(*bp0);

        if (bp0->bar_time > bt) {
            // backward fill using the first bar's close price
            logInfo("start bar time %lu (%s)  earlier than first bar "
                     "in bar file %lu (%s)",
                     (unsigned long) bt, 
                     utils::TimeUtil::frac_UTC_to_string(bt, 0).c_str(),
                     (unsigned long) bp0->bar_time,
                     utils::TimeUtil::frac_UTC_to_string(bp0->bar_time, 0).c_str()
                    );
            // create a bar filled with bp0's open price
            
            fill.close = fill.open;
            while (bt < bp0->bar_time) {
                fill.writeAndRoll(bt);
                bars.emplace_back( new BarPrice(fill) );
                bt = nextBar(bt);
            }
        }

        // forward fill any missing
        while((bt <= end_bartime) && (bcnt < allb.size())) {
            bp0 = allb[bcnt];
            while ( (bcnt < allb.size()-1) && (bp0->bar_time < bt) ) {
                bp0 = allb[++bcnt];
            }
            if (bp0->bar_time > bt) {
                auto bp1 = allb[bcnt-1];
                // fill with bp1 from bt to (not including) bp0->bar_time
                fill = *bp1;
                fill.writeAndRoll(bt);
                while ((bt < bp0->bar_time) && (bt <= end_bartime)) {
                    bars.emplace_back(new BarPrice(fill));
                    bt += barsec;
                }
            }
            if (bt > end_bartime) {
                break;
            }

            // demand matching of bt and bp0 at this time
            if (bt == bp0->bar_time) {
                bars.emplace_back(bp0);
                bt+= barsec;
                ++bcnt;
            } else {
                if (bcnt == allb.size()-1) {
                    logInfo("Forward fill using a bar earlier than the starting time given!");
                    // we are at the last one already
                    // forward fill using this one
                    break;
                }
                logError("%s BarReader forwardBackwardFill failed: bartime mismatch!"
                         "barfile has a different barsec? BarFile: %s, barsec: %d\n"
                         "bt = %lu (%s), bp0 = %s",
                        bcfg.toString().c_str(), fn.c_str(), barsec, 
                        (unsigned long) bt, utils::TimeUtil::frac_UTC_to_string(bt, 0).c_str(),
                        bp0->toCSVLine().c_str());
                throw std::runtime_error("BarReader barsec mismatch " + bcfg.toString());
            }
        }

        // forward fill into bar times not covered by allb
        if (bt <= end_bartime) {
            fill = *allb[allb.size()-1];
            while (bt <= end_bartime) {
                fill.writeAndRoll(bt);
                bars.emplace_back(new BarPrice(fill));
                bt += barsec;
            }
        }

        return true;
    }

    time_t bartimeByOffset(time_t bartime, int offset) const {
        // gets the trading bar time w.r.t. offset in barsec from bartime
        time_t bt = bartime;
        int bs_ = barsec;
        if (offset<0) {
            bs_=-bs_;
            offset = -offset;
        };
        while (offset > 0) {
            bt += bs_;
            if (VenueConfig::get().isTradingTime(bcfg.venue, bt)) {
                --offset;
            }
        }
        return bt;
    }

    time_t prevBar(time_t bartime) const {
        return bartimeByOffset(bartime, -1);
    }

    time_t nextBar(time_t bartime) const {
        return bartimeByOffset(bartime, 1);
    }

    ~BarReader() {
        if (fp) {
            fclose(fp);
            fp = nullptr;
        }
    }

    std::string toString() const {
        char buf[256];
        snprintf(buf, sizeof(buf), "Bar Reader {config: %s, period: %d, bar_file: %s}",
                bcfg.toString().c_str(), barsec, fn.c_str());
        return std::string(buf);
    }

    BarReader(const BarReader&br)
    : bcfg(br.bcfg), barsec(br.barsec), fn(br.fn), fp(nullptr)
    {
        fp = fopen(fn.c_str(), "rt");
        if (!fp) {
            logError("failed to open bar file %s", fn.c_str());
            throw std::runtime_error(std::string("faile to open bar file ") + fn);
        }
        bp = getLatestBar();
    }

public:
    const BookConfig bcfg;
    const int barsec;
    const std::string fn;

private:
    FILE* fp;
    BarPrice bp;

    BarPrice getLatestBar() {
        char buf[256];
        buf[0] = 0;
        try {
            // For non-traditional linux device, 
            // such as AWS files, fseek/ftell
            // is necessary to refresh read stream
            auto pos = ftell(fp);
            fseek(fp, 0, SEEK_END);
            auto pos_end = ftell(fp);
            if (pos_end > pos) {
                // new updates
                fseek(fp, pos, SEEK_SET);
                while (fgets(buf  , sizeof(buf)-1, fp)) {};
                return BarPrice(std::string(buf));
            } else if (pos_end < pos) {
                logError("Bar file %s truncated!", fn.c_str());
            }
        } catch (const std::exception & e) {
            logError("failed to get last bar price from %s: %s", fn.c_str(), e.what());
        }
        return BarPrice();
    }

    void operator = (const BarReader&) = delete;
};

/* 
 * Utility for Input/Output types
 */
class NullOutput {
public:
    template<typename BookType>
    bool put(const BookType& data) { return true; };
    bool put(const std::string& str) { return true; };
    bool put(const char* data, size_t len) { return true; };
    void reset() const {};
};

class FileOutput {
public:
    explicit FileOutput(const std::string& fname, bool append=true, bool binary=false)
    : m_fname(fname), m_append(append), m_binary(binary), m_fp(nullptr) {
        setup();
    }

    FileOutput(const FileOutput& fo)
    : m_fname(fo.m_fname), m_append(fo.m_append), m_binary(fo.m_binary), m_fp(nullptr) {
        setup();
    }

    ~FileOutput() { 
        if (m_fp) fclose(m_fp);
        m_fp=nullptr;
    };

    template<typename BookType>
    bool put(const BookType& data) {
        if (m_binary) {
            if (__builtin_expect(fwrite((const void*)&data, sizeof(BookType), 1, m_fp) != 1,0)) {
                logError("Failed in fwrite object of %d bytes", (int) sizeof(BookType));
                return false;
            }
        } else {
            fprintf(m_fp, "%s\n", data.toCSV().c_str());
        }
        fflush(m_fp);
        return true;
    };

    bool put(const std::string& str) {
        return put(str.c_str(), str.size());
    }

    bool put(const char* data, size_t len) {
        size_t bytes = fwrite(data, 1, len, m_fp);
        std::string ret="\n";
        bytes+=fwrite(ret.c_str(), 1, 1, m_fp);
        if (__builtin_expect(bytes != len+1, 0)) {
            logError("Failed in fwrite string of %d bytes", (int) len);
            return false;
        }
        fflush(m_fp);
        return true;
    };

    void reset() const {};

private:
    const std::string m_fname;
    bool m_append, m_binary;
    FILE* m_fp;
    void setup() {
        std::string fm=(m_append?"a":"w");
        fm += (m_binary?"b":"t");
        m_fp=fopen(m_fname.c_str(),fm.c_str());
        if (!m_fp) {
            throw std::runtime_error(std::string("failed to create file to write: ") + m_fname);
        }
    }
    void operator=(const FileOutput& fo) = delete;
};

class TickDump: FileOutput {
public:
    TickDump(const std::string& fname) :
    FileOutput(fname, true, false) {};
};

// just a collection of BookType objects in a vector
class StringVectorOutput {
public:
    StringVectorOutput()
    : m_vec(std::make_shared<std::vector<std::string>>()) 
    {};

    explicit StringVectorOutput(std::shared_ptr<std::vector<std::string>>& vec)
    : m_vec(vec) 
    {};

    template<typename BookType>
    bool put(const BookType& book) { 
        // just use the tick csv
        return put(book.toCSV());
    };
    bool put(const std::string& str) {
        m_vec->push_back(str);
        return true;
    };
    bool put(const char* data, size_t len) {
        m_vec->push_back( std::string(data,len) );
        return true;
    }
    void reset() const { };
    std::shared_ptr<std::vector<std::string>> m_vec;
};

// a simple BarWriter with live time management logics removed, 
// fit for offline(historical) tick ingestions.
// Compared with BarWriter in md_bar.h, it
// doesn't care about the local_time/exchange_time, don't call onOneSecond()
// when idle, and don't roll over night.
// OutputType usually a file writer for writing bar line.
// It also entertain an optional tickdump for trace purpose
//
// Could be used as Output for BookWriter
//  * Example of quote/trade update to bar file
//    - BookWriter<BarPriceOutput<FileOutput>, BookLevel>
//  * Example of quote/trade update to both bar and tick file
//    - BookWriter<BarPriceOutput<FileOutput, TickDump>, BookLevel>
//  * Example of quote/trade update to tick file
//    - BookWriter<FileOutput, BookLevel>
//
//  refer to td_parser.h
//
template<typename BarOutputType, typename TickOutputType=NullOutput>
class BarPriceOutput {
public:
    BarPriceOutput(time_t start_utc, time_t end_utc, int barsec,
                  std::shared_ptr<BarOutputType>& bar_out, 
                  std::shared_ptr<TickOutputType>& tick_out,
                  Price tick_size=0, bool write_optional=false)
    : m_bar_out(bar_out), m_tick_out(tick_out),
      m_sutc(start_utc), m_eutc(end_utc), m_bardue(m_sutc+barsec), m_barsec(barsec)
    {
        m_bp.set_write_optional(write_optional);
        m_bp.set_tick_size(tick_size);

        if (m_bar_out==nullptr || m_tick_out==nullptr) {
            logError("null output in BarPriceOutput!");
            throw std::runtime_error("null output in barPriceOutput!");
        }

    }
    
    template<typename BookType>
    bool put(const BookType& book) {
        check_roll(book.update_ts_micro);
        m_bp.update(book);
        // tick dump
        m_tick_out->put(book);
        return true;
    }

    ~BarPriceOutput() {
        // forward fill for the bar in case no tick updates
        if (m_bardue <= m_eutc) {
            check_roll(m_eutc*1000000LL);
        }
    }

    void reset() {
        m_bar_out->reset();
        m_tick_out->reset();
        m_bardue = m_sutc+m_barsec;
    };

private:
    BarPrice m_bp;
    std::shared_ptr<BarOutputType> m_bar_out;
    std::shared_ptr<TickOutputType> m_tick_out;
    time_t m_sutc, m_eutc, m_bardue;
    int m_barsec;
    
    void check_roll(long long cur_micro) {
        time_t cur_sec = (time_t) (cur_micro/1000000LL);
        for (; m_bardue<=cur_sec; m_bardue+=m_barsec) {
            m_bar_out->put(m_bp.writeAndRoll(m_bardue));
        }
    };
};
}
