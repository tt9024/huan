#pragma once

#include "md_bar_price.h"
#include "csv_util.h"
#include "thread_utils.h"

/*******************************************************
 * LIVE Market Data Ingestions for Both Ticks and Bars
 *******************************************************/

//maximum wait beyond bar close time in case of idle
//A tradeoff between timely of bar and accuracy of tick-inclusions
#define LiveBarIdleWait_Milli 250 

namespace md {

/*
 * BarWriter deals with tricky timing of 'quote','trade' time (exchange time)
 * and the local time, observe bar periods, and roll over days.
 * BarOutType implements put(string). Itself can be used as an Output.
 */
template<typename BarOutType>
class BarWriter {
public:
    explicit BarWriter(const BookConfig& bcfg, const std::map<int, BarOutType>& out_map, time_t cur_second=0)
    : m_bcfg(bcfg), m_out_map(out_map), m_start_utc(0), m_end_utc(0), 
      m_active(false), last_micro(0), last_local_micro(0)
    {
        if (cur_second == 0) cur_second = utils::TimeUtil::cur_utc();
        setup(); // setup the output objects,they persist over days
    }

    template<typename BookType>
    bool onUpdate(long long cur_micro, const BookType& book) {
        static const long long TradeTimeReduce = 30*1000LL; // see hack
        if (__builtin_expect(!book.isValid(),0)) {
            return false;
        }
        int32_t cur_utc = (int32_t) (cur_micro/1000000LL);
        int64_t upd_micro = book.update_ts_micro;

        // check on local time and time from update (exchange time)
        time_t upd_utc = (time_t)(upd_micro/1000000LL);
        if (__builtin_expect(upd_utc>cur_utc+1, 0)) {
            logError("BarWriter(%s) received update for %s(%lld), that is more than 1 second ahead local time! %s(%lld)\nCheck clock sync or exchange feed format in parsing sending time stamps.\nbook: %s",
                    m_bcfg.toString().c_str(), 
                    utils::TimeUtil::frac_UTC_to_string(upd_micro,6).c_str(),
                    (long long) upd_micro,
                    utils::TimeUtil::frac_UTC_to_string(cur_micro,6).c_str(),
                    (long long) cur_micro,
                    book.toString().c_str());
            //return false;
        };

        BookType book0(book); // book with some modifications
        // enforce non-decreasing update time
        if (__builtin_expect(upd_micro < last_micro,0)) {
            upd_micro = last_micro;
            book0.update_ts_micro=upd_micro;
        }
        else {
            // This is a Hack:
            // Since bpipe stream's trade timestamp may be ahead of
            // quotes, the time maybe pushed forward with 
            // non-decreasing constraints, possibly across bar close.
            // Here we deduce trade's timestamp 100 millis if possible.
            // This is not important, as typically a quote will immediately 
            // follow after a trade, with quote sending timestamp.
            //
            // TODO - review the trade/quote sending time diverge issue.
            // Compared to TickData, BPIPE is not that bad.  TickData's
            // timestamp is their receive time, and that could add latency
            // between 10-milli to 500-milli compared with BPIPE's sending time,
            if (__builtin_expect(book0.isTradeUpdate(),0)) {
                upd_micro-=TradeTimeReduce;
                upd_micro=_MAX_(last_micro,upd_micro);
                book0.update_ts_micro=upd_micro;
            }
        }

        // apply update
        if (__builtin_expect(checkActive(upd_utc, book0),1)) {
                // debug
                /*
                if (m_bcfg.symbol=="CLN3") {
                    fprintf(fp_cl, "U%d,%lld,%lld,%lld\n", (int)upd_utc,(long long)upd_micro,(long long)cur_micro, (long long)last_local_micro);
                    fflush(fp_cl);
                }*/

            // only output bar lines during active hours
            checkRoll((time_t)upd_utc);
        }

        // but always update bar states
        updateState(book);
        last_micro = upd_micro;
        last_local_micro=cur_micro;
        return true;
    }

    // This is provided but not necessary as long as live bar reader
    // do forward fill based on local time. 
    // However the python code sort of relies on the update of barfile.
    // So BarWriterLive still need to calls this to output bar upon
    // due time in case idle.
    template<typename BookType>
    void onOneSecond(long long cur_micro, const BookType& book) {
        static const long long shift_micro=LiveBarIdleWait_Milli*1000LL; 
        if (__builtin_expect(!book.isValid(),0)) {
            //logError("invalid book updated at BarWrite onOneSecond() bookq empty?: %s", m_bcfg.toString().c_str());
            return;
        }

        time_t cur_utc = cur_micro/1000000LL;
        if (__builtin_expect(cur_utc <= last_local_micro/1000000LL,1)) {
            return;
        }

        // No new updates, check bar due against local cur_utc.
        // A tricky case is that although no new updates, the future
        // updates will still be less that the current bar close, but
        // our local time has already exceeded the current bar close.
        if (__builtin_expect(checkActive(cur_utc, book),1)) {
            // allow for a slight delay milli-second
            // This is an important parameter, a trade off between
            // responsiveness and accuracy.  During heavy load, the current
            // time maybe behind sending time by quite a lot. This suggests
            // to increase this threshold.  On the other hand, during idle
            // time, the bar would thus be closed later than it should be. 
            //
            // The consequence now is that bar is almostly always driven by
            // sending time, but during idle periods, the bar could be delayed
            // by about "shift_micro" during idle time. The tradeoff cause
            // it to be increased from 0.1 second to 0.25 second.
            //
            if (__builtin_expect(
                        (cur_micro>last_local_micro+2*shift_micro) &&
                        (cur_micro%1000000LL>shift_micro),1)) {
                checkRoll(cur_utc);
            }
        }
    }

    ~BarWriter() {
        //fclose(fp_cl);
    }
    const BookConfig m_bcfg;

    // as an OutType
    template<typename Booktype>
    bool put(const Booktype& book) {
        return onUpdate(book.update_ts_micro, book);
    }
    bool put(const std::string& str) {
        throw std::runtime_error(std::string("BarWriter (") + m_bcfg.toString() + ") cannot write str (" + str + ")! Use maybe a FileWriter: ");
    }
    bool put(const char* bytes, size_t len) {
        throw std::runtime_error(std::string("BarWriter (") + m_bcfg.toString() + ") cannot write bytes. Use maybe a FileWriter: ");
    }
    std::map<int, BarOutType> m_out_map;

private:
    struct BarInfo {
        BarOutType bar_out;
        time_t due;
        time_t start;
        time_t end;
        BarPrice bar;
        BarInfo(BarOutType& bout)
        : bar_out(bout), due((time_t)-1), start(0), end(0) 
        {}
    };

    std::map<int, std::shared_ptr<BarInfo> > m_bar;
    time_t m_start_utc, m_end_utc;
    bool m_active;
    // debug
    //FILE* fp_cl;
    int64_t last_micro;
    int64_t last_local_micro;

    void setup() {
        // get the tick size from bcfg
        double tick_size = 0;
        try {
            const auto* ti(utils::SymbolMapReader::get().getByTradable(m_bcfg.symbol));
            tick_size = ti->_tick_size;
        } catch (const std::exception& e) {
            logError("BarWriter(%s) failed to get tick size!", m_bcfg.toString().c_str());
            throw std::runtime_error("BarWriter failed to get tick size for " +  m_bcfg.toString());
        }
        for (auto& out_item: m_out_map) {
            std::shared_ptr<BarInfo> binfo(new BarInfo(out_item.second));
            /*
            binfo->fn = out_item.second;
            binfo->fp = fopen(binfo->fn.c_str(), "at");
            if (!binfo->fp) {
                logError("BarWriter(%s) failed to create bar file %s!", m_bcfg.toString().c_str(), binfo->fn.c_str());
                throw std::runtime_error("BarWriter failed to create bar file " + binfo->fn);
            }
            */

            binfo->bar.set_tick_size(tick_size);
            binfo->bar.set_write_optional(true);
            m_bar.emplace(out_item.first, binfo);
        }
    };

    void resetTradingDay(time_t cur_second) {
        logDebug("Getting trading hours for %s", m_bcfg.toString().c_str());
        // get the start stop utc for current trading day, snap to future
        if (!VenueConfig::get().isTradingTime(m_bcfg.venue, cur_second)) {
            logDebug("%s not currently trading, wait to the next open", m_bcfg.venue.c_str());
        }
        const auto start_end_pair = VenueConfig::get().startEndUTC(m_bcfg.venue, cur_second, 2);
        time_t sutc = start_end_pair.first;
        time_t eutc = start_end_pair.second;

        logDebug("%s BarWriter got trading hours [%lu (%s), %lu (%s)]", m_bcfg.venue.c_str(),
                (unsigned long) sutc,
                utils::TimeUtil::frac_UTC_to_string(sutc, 0).c_str(),
                (unsigned long) eutc, 
                utils::TimeUtil::frac_UTC_to_string(eutc, 0).c_str());

        for (const auto& out_item : m_out_map) {
            auto bs = out_item.first;
            auto& binfo = m_bar[bs];
            time_t due = (time_t)((int)(cur_second/bs+1)*bs);

            if (due < sutc+bs) {
                // first bar time
                due = sutc+bs;
            } else if (due > eutc) {
                // this should never happen!
                logError("%s %d second BarWriter next due %lu (%s) outside trading session,"
                        "skip to next open %lu (%s)",
                        m_bcfg.venue.c_str(), bs,
                        (unsigned long) due,
                        utils::TimeUtil::frac_UTC_to_string(due, 0).c_str(),
                        sutc + 3600*24 + bs,
                        utils::TimeUtil::frac_UTC_to_string(sutc+3600*24 + bs, 0).c_str());
                due = sutc + 3600*24 + bs;
                sutc += (3600*24);
                eutc += (3600*24);
            }
            binfo->due = due;
            binfo->start = sutc;
            binfo->end = eutc;
            logDebug("%s %d Second BarWriter next due %lu (%s)", 
                    m_bcfg.venue.c_str(), (int)bs,
                    (unsigned long) binfo->due,
                    utils::TimeUtil::frac_UTC_to_string(binfo->due, 0).c_str());
        }
        m_start_utc=sutc;
        m_end_utc=eutc;
    }

    template<typename BookType>
    void init(const BookType& book, time_t cur_utc) {
        // prime the pump, setting the update time to be
        // 1 second earlier. 
        // This is called at very first update of BarWriter
        // after TradingDay has been setup.
        // Refer to checkActive()
        last_micro=book.update_ts_micro-1000000LL;
        last_local_micro=(long long)(cur_utc-1)*1000000LL;

        BookType book0(book);
        for (int i=0; i<3; ++i) {
            // initialize with all 3 update types
            book0.update_type = i;
            for (auto& bitem: m_bar) {
                time_t prev_due = bitem.second->due - bitem.first;
                book0.update_ts_micro =  (_MAX_(prev_due, m_start_utc)) * 1000000LL - 1;
                bitem.second->bar.update(book0);
            }
        }
    }

    template<typename BookType>
    bool checkActive(time_t cur_utc, const BookType& book) {
        if (__builtin_expect(cur_utc > m_end_utc, 0)) {
            resetTradingDay(cur_utc);
            if (m_active) {
                logInfo("BarWriter(%s) de-activated, next start/stop at: %s to %s",
                        m_bcfg.toString().c_str(), 
                        utils::TimeUtil::frac_UTC_to_string(m_start_utc, 0).c_str(),
                        utils::TimeUtil::frac_UTC_to_string(m_end_utc, 0).c_str());
            } else {
                if (last_micro==0) {
                    init(book, cur_utc);
                };
            }
            m_active=false;
        }
        if (__builtin_expect(cur_utc < m_start_utc, 0)) {
            m_active=false;
            return false;
        }

        // CASES of cur_utc>=m_start_utc
        if (__builtin_expect(!m_active ,0)) {
            // The bar was initialized with the snap update at init() upon creation.
            // If the creation time is before this start, a roll is needed
            // to make the "time weighted average" to calculate correctly
            // for the first bar. 
            // Note the initialization with snap update at init() is necessary, 
            // for the case when no update for first bars

            // not needed!
            //if (cur_utc == m_start_utc) {
            
            for (auto& bitem: m_bar) {
                time_t prev_due = bitem.second->due - bitem.first;
                bitem.second->bar.writeAndRoll(_MAX_(m_start_utc, prev_due));
            }
            logInfo("BarWriter(%s) activated", m_bcfg.toString().c_str());
            m_active=true;
        }
        return m_active;
    }

    void checkRoll(time_t cur_sec) {
        // checkRoll should be reentrant/stateless
        for (auto& bitem: m_bar) {
            auto& binfo = bitem.second;
            if (__builtin_expect(!binfo->bar.isValid(),0)) {
                continue;
            }
            const auto bsec = bitem.first;
            // due for current bar close, possibly catch up
            if (__builtin_expect(cur_sec >= binfo->due,0)) {
                while(true) {
                    const auto line = binfo->bar.writeAndRoll(binfo->due);

                    // debug
                    /*
                    if ((m_bcfg.symbol=="CLN3") && (bsec==1)) {
                        fprintf(fp_cl, "R%d,%lld,%lld,%lld,%s\n", (int)cur_sec,
                                (long long) m_book.update_ts_micro,
                                (long long) utils::TimeUtil::cur_micro(),
                                (long long) last_local_micro,
                                line.c_str());
                        fflush(fp_cl);
                    }*/

                    binfo->bar_out.put(line);
                    binfo->due += bsec;
                    if (__builtin_expect(cur_sec < binfo->due,1)) {
                        break;
                    }
                }
            }
        }
    }

    template<typename BookType>
    void updateState(const BookType& book) {
        // update all bars with different bar period
        for (auto& bitem: m_bar) {
            auto& bar = bitem.second->bar;
            bar.update(book);
        }
    }
};

template<int BookLevel>
class BarWriterLive {
public:
    using BarOutType = FileOutput; // output to bar file according to run/config.cfg
    using BookQType  = BookQ<BookLevel>;
    using TimerType  = utils::TimeUtil;

    explicit BarWriterLive(const std::string& bar_path="")
    : m_bar_path(bar_path==""? plcc_getString("BarPath"):bar_path),
      m_writer_cnt(0), m_should_run(false), m_running(false)
    {}

    // thread safe, this could be called any time, 
    // even as the thread is running. 
    void add(std::shared_ptr<BookQ<BookLevel>> bq, uint64_t cur_micro = 0) {
        if (cur_micro == 0) {
            cur_micro = TimerType::cur_micro();
        }

        // need to protect with a lock
        {
            utils::SpinLock spin(m_lock);
            if (__builtin_expect(m_writer_cnt >= MAX_WRITERS, 0)) {
                logError("Too many bar writers (%d), not adding more!", (int)m_writer_cnt);
                throw std::runtime_error("Too many bar writers!");
            }
            //const auto cnt = m_writer_cnt.fetch_add(1, std::memory_order_relaxed);
            m_writers[m_writer_cnt] = std::make_shared<WriterInfo>(bq, m_bar_path);
            m_writer_cnt+=1;
        }
    }

    void run( [[maybe_unused]] void* param = NULL) {
        m_should_run = true;
        m_running = true;
        const int64_t min_sleep_micro = 400;
        const int64_t max_sleep_micro = 2000;
        int64_t sleep_micro = min_sleep_micro;
        int64_t last_update_micro=TimerType::cur_micro();
        int64_t FiveMin = 300LL*1000LL*1000LL;

        while (m_should_run) {
            bool has_update = false;
            const size_t cnt = m_writer_cnt;
            for (size_t i=0; i<cnt; ++i) {
                auto& bw (m_writers[i]);
                has_update |= bw->check(TimerType::cur_micro());
            }
            int64_t cur_micro=TimerType::cur_micro();
            if (!has_update) {
                if (__builtin_expect(!utils::TimeUtil::isTradingTime((time_t)(cur_micro/1000000LL)-5, -6, 0, 17, 0),0)) {
                    m_should_run=false;
                    break;
                }

                if (__builtin_expect((cur_micro-last_update_micro>FiveMin) && (cnt>0),0)) {
                    logError("BarWriterLive no any update from all %d books for %d minutes! If this happens persistently, check and bounce the market data feed!",
                            (int) cnt, (int) ((cur_micro-last_update_micro)/60000000LL));
                    last_update_micro=cur_micro;
                }
                sleep_micro+=(sleep_micro/10*2); // increase by 1.2
                sleep_micro+=(cur_micro%min_sleep_micro); // small random
                sleep_micro = _MIN_(max_sleep_micro,sleep_micro);
                TimerType::micro_sleep(sleep_micro);
            } else {
                sleep_micro=min_sleep_micro;
                last_update_micro=cur_micro;
            }
        }
        m_running = false;
        logInfo("Bar Writer Stopped");
    }

    void stop() {
        m_should_run = false;
    }

    bool running() const {
        return m_running;
    }

    bool should_run() const {
        return m_should_run;
    }

private:
    enum { MAX_WRITERS = 1024*8};
    const std::string m_bar_path;

    struct WriterInfo {
        std::shared_ptr<BookQType> _bq;
        std::shared_ptr<typename BookQType::Reader> _br;
        std::shared_ptr<BarWriter<BarOutType>> _out;
        BookDepotLevel<BookLevel> _book;
        long long _last_local_micro;

        WriterInfo(std::shared_ptr<BookQ<BookLevel>> bq, const std::string& bar_path)
        : _bq(bq), _br(bq->newReader()),
          _out(std::make_shared<BarWriter<BarOutType>>(_bq->_cfg, genOutMap(bq->_cfg, bar_path))),
          _last_local_micro(0) 
        {
            // try to init the book in case needed for init
            if (!_br->getLatestUpdate(_book)) {
                logError("BarWriterLive cannot read book:%s", bq->_cfg.toString().c_str());
                /*
                throw std::runtime_error(
                        std::string("BarWriterLive cannot read book:") +
                        bq->_cfg.toString());
                */
            }
        }

        bool check(long long cur_micro) {
            if (_br->getNextUpdate(_book)) {
                _out->onUpdate(cur_micro, _book);
                _last_local_micro = cur_micro;
                return true;
            }
            // has the idle across the utc yet?
            if (__builtin_expect( (cur_micro/1000000LL>_last_local_micro/1000000LL) &&
                        ((cur_micro%1000000LL)>(LiveBarIdleWait_Milli*1000LL)),0)) {
                _out->onOneSecond(cur_micro, _book);
            }
            return false;
        }
    private:
        const std::map<int, BarOutType> genOutMap(const BookConfig& bcfg, const std::string& bar_path) const {
            std::map<int, BarOutType> out;
            const auto& bs_vec(bcfg.barsec_vec());
            bool append = true;
            bool binary = false;
            for (const auto bs : bs_vec) {
                const auto fn = bcfg.bfname(bs, bar_path);
                if (!out.try_emplace(bs, fn, append, binary).second) {
                    logError("Failed to add bar writer for barsec %d, fn %s", 
                            (int) bs, fn.c_str());
                }
            }
            return out;
        };
    };

    std::array<std::shared_ptr<WriterInfo>, MAX_WRITERS> m_writers;
    std::atomic<size_t> m_writer_cnt;
    volatile bool m_should_run, m_running;
    utils::SpinLock::LockType m_lock;
};

/*
 * Driven by bpipe/FIX parser thread. It runs on parser's thread, allow the parser
 * to obtain via getWriter() to publish Quotes/Trade to shm. It also creates a BarWriterLive
 * thread to read from the shm to bar files.
 */
template<int BookLevel>
class MD_Publisher {
public:
    using BookOutputType = typename BookQ<BookLevel>::Writer;
    using BookWriterType = BookWriter<BookOutputType, BookLevel>;

    explicit MD_Publisher(const std::string& provider="")
    : m_provider(provider), m_bar_writer(), m_bar_writer_thread(m_bar_writer) {
        logInfo("MD_Publisher(%s) started!", (provider.size()>0? provider.c_str():"default"));
        m_bar_writer_thread.run(NULL);
    }

    ~MD_Publisher() {};

    std::shared_ptr<BookWriterType> getWriter(const std::string& symbol, const std::string& venue="") {
        const std::string key = getKey(symbol, venue);
        const auto iter = m_book_writer.find(key);
        if (__builtin_expect(iter == m_book_writer.end(),0)) {
            logError("symbol %s(%s) not found in publisher's symbol list, adding with provider (%s)", 
                    symbol.c_str(), venue.c_str(), m_provider.c_str());
            return addWriter(symbol, venue, m_provider);
        }
        return iter->second;
    }

    void stop() {
        logInfo("Stopping md publisher: is_running (%s), should_run(%s)", 
                m_bar_writer.running()?"Yes":"No",
                m_bar_writer.should_run()?"Yes":"No");
        m_bar_writer.stop();
    }

    // note the book config with provider "" is served as a primary feed, refer to BookConfig for detail
    std::shared_ptr<BookWriterType> addWriter(const std::string& symbol,
                                        const std::string& venue, 
                                        const std::string& provider) {
        const md::BookConfig bcfg (venue, symbol, "L1", provider);
        auto bq (std::make_shared<BookQ<BookLevel>>(bcfg, false));

        m_bar_writer.add(bq);
        const auto key = getKey(symbol, venue);
        auto bwriter ( std::make_shared<BookWriterType>(bq->theWriter(),bcfg) );
        m_book_writer[key]=bwriter;
        m_bookq[key]=bq; // save an instance top the bookq
        return bwriter;
    }

    const std::string m_provider;

protected:
    // read/write bookQ, saved and used for both BookWriter (write to snap queue)
    // and for BarWriter (read from 
    std::unordered_map<std::string, std::shared_ptr<BookQ<BookLevel>>> m_bookq;
    std::unordered_map<std::string, std::shared_ptr<BookWriterType>> m_book_writer;
    BarWriterLive<BookLevel> m_bar_writer;
    utils::ThreadWrapper<BarWriterLive<BookLevel>> m_bar_writer_thread;

    const std::string getKey(const std::string& symbol, const std::string& venue) const {
        return symbol + venue;
    }
};
}
