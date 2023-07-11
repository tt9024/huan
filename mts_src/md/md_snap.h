#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "md_book.h"

namespace md {

/*********************
 * BookQ shm read/write 
 *********************/

template <int BookLevel>
class BookQ {
public:
    static const int BookLen = sizeof(BookDepotLevel<BookLevel>);
    static const int QLen = (1024*64*BookLen);
    // This is to enforce that for SwQueue, at most one writer should
    // be created for each BookQueueu
    using QType = utils::SwQueue<QLen, BookLen, utils::ShmCircularBuffer>;
    const BookConfig _cfg;
    const std::string _q_name;
    class Writer;
    class Reader;

    BookQ(const BookConfig& config, bool readonly, bool init_to_zero=false) :
        _cfg(config), _q_name(_cfg.qname()), _q(_q_name.c_str(), readonly, init_to_zero),
        _writer(readonly?nullptr:std::make_shared<Writer>(*this))
    {
        if (!readonly && (_writer==nullptr))
            throw std::runtime_error("BookQ writer instance NULL");
        /*
        logInfo("BookQueueu %s started %s configs (%s).",
                _q_name.c_str(),
                readonly?"ReadOnly":"ReadWrite",
                _cfg.toString().c_str());
        */
    };

    std::shared_ptr<Writer>& theWriter() {
        return _writer;
    }

    std::shared_ptr<Reader> newReader() {
        return std::make_shared<Reader>(*this);
    }

    ~BookQ() {};

private:
    QType _q;
    std::shared_ptr<Writer> _writer;
    friend class Writer;
    friend class Reader;

public:

    // Writer uses BookType interface of new|del|upd|Price()
    // and getL2(), it always writes L2 entries
    class Writer {
    public:
        explicit Writer(BookQ<BookLevel>& bq) : _bq(bq), _wq(_bq._q.theWriter()) {
        }

        bool put (const BookDepotLevel<BookLevel>& book) {
            _wq.put((char*)&book);
            return true;
        }

        // Output Types
        bool put (const std::string& str) { throw std::runtime_error("BookWriter can't write str");};
        bool put (const char* byte, size_t len) {throw std::runtime_error("BookWriter can't write bytes");};
        void reset() {};
    private:
        BookQ<BookLevel>& _bq;
        typename BookQ<BookLevel>::QType::Writer& _wq;  // the writer's queue
        friend class BookQ<BookLevel>;

        // disallow copy
        explicit Writer (const Writer& wtr) = delete;
        void operator==(const Writer& wtr) = delete;
    };

    // reader always assumes a normalized BookDepot
    class Reader {
    public:
        bool getNextUpdate(BookDepotLevel<BookLevel>& book) {
            utils::QStatus stat = _rq->copyNextIn((char*)&book);
            switch (stat) {
            case utils::QStat_OK :
                _rq->advance();
                return true;
            case utils::QStat_EAGAIN :
                return false;
            case utils::QStat_OVERFLOW :
            {
                int lost_updates = _rq->catchUp();
                logError("venue read queue %s overflow, lost %d updates. Trying to catch up."
                        ,_bq._q_name.c_str(), lost_updates);
                return getNextUpdate(book);
            }
            case utils::QStat_ERROR :
            {
                _rq->advanceToTop();
                logError("venue read queue %s error. Trying to sync."
                        ,_bq._q_name.c_str());
                return getNextUpdate(book);
            }
            }
            logError("getNextUpdate read queue %s unknown qstat %d, exiting..."
                    ,_bq._q_name.c_str(), (int) stat);
            throw std::runtime_error("BookQ Reader got unknown qstat.");
        }

        bool getLatestUpdate(BookDepotLevel<BookLevel>& book) {
            _rq->seekToTop();
            utils::QStatus stat = _rq->copyNextIn((char*)&book);
            switch (stat) {
            case utils::QStat_OK :
                return true;
            case utils::QStat_EAGAIN :
                return false;
            default:
                logError("getLatestUpdate read queue %s unknown qstat %d, exiting..."
                    ,_bq._q_name.c_str(), (int) stat);
                throw std::runtime_error("BookQ Reader got unknown qstat.");
            }
        }

        bool getLatestUpdateAndAdvance(BookDepotLevel<BookLevel>& book) {
            if (__builtin_expect(!_rq->advanceToTop(),0)) {
                return false;
            }
            utils::QStatus stat = _rq->copyNextIn((char*)&book);
            if (__builtin_expect(stat == utils::QStat_OK, 1)) {
                _rq->advance();
                return true;
            }
            switch (stat) {
            case utils::QStat_EAGAIN :
                return false;
            case utils::QStat_OVERFLOW :
            {
                // try again
                _rq->advanceToTop();
                stat = _rq->copyNextIn((char*)&book);
                if (stat != utils::QStat_OK) {
                    return false;
                }
                return true;
            }
            case utils::QStat_ERROR :
            {
                _rq->advanceToTop();
                logError("venue read queue %s error. Trying to sync."
                    ,_bq._q_name.c_str());
                return false;
            }
            default :
                    logError("read queue %s unknown qstat %d, existing...", _bq._q_name.c_str(), (int) stat);
                    throw std::runtime_error("BookQ Reader got unknown qstat!");
            }
            return false;
        }

        Reader(BookQ<BookLevel>& bq) : _bq(bq), _rq(_bq._q.newReader()){};

        ~Reader() {
        }

    private:
        BookQ<BookLevel>& _bq;
        std::shared_ptr< typename BookQ<BookLevel>::QType::Reader> _rq;  // the reader's queue
        friend class BookQ<BookLevel>;
    };
};

/*
 * Utilities that uses the global BookQ
 */
// query book by symbol and a book type: "L1 or L2"
// symbol could be in form of venue/tradable, or just tradable, or just a MTS symbol
template<int BookLevel>
static inline
bool LatestBook(const std::string& symbol, const std::string& levelStr, BookDepotLevel<BookLevel>& myBook) {
    BookConfig bcfg(symbol, levelStr);
    BookQ<BookLevel> bq(bcfg, true);
    auto book_reader = bq.newReader();
    if (!book_reader) {
        logError("Couldn't get book reader!");
        return false;
    }
    bool ret = book_reader->getLatestUpdate(myBook);
    return ret;
}

// L1 convenience functions
static inline
bool getBBO(const std::string& symbol, double& bidpx, int& bidsz, double& askpx, int& asksz) {
    BookDepotLevel<1> myBook;
    if (!LatestBook(symbol, "L1", myBook)) {
        return false;
    }
    bidsz=0;
    asksz=0;
    bidpx = myBook.getBid(&bidsz);
    askpx = myBook.getAsk(&asksz);
    return true;
}

static inline
bool getBBOPriceEntry(const std::string& symbol, PriceEntry& bid_pe, PriceEntry& ask_pe) {
    BookDepotLevel<1> myBook;
    if (!LatestBook(symbol, "L1", myBook)) {
        return false;
    }
    if (__builtin_expect(myBook.isValid(),1)) {
        bid_pe = *myBook.getEntryConst(0,0);
        ask_pe = *myBook.getEntryConst(0,1);
        return true;
    }
    return false;
}

static inline
std::string getBBOString(const std::string& symbol, const double bidpx, const int bidsz, const double askpx, const int asksz) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s BBO (%d %s : %s %d)", symbol.c_str(), bidsz, PriceCString(bidpx), PriceCString(askpx), asksz);
    return std::string(buf);
}

static inline
double getBidPrice(const std::string& symbol) {
    int bsz = 0,asz = 0;
    double bpx = 0, apx = 0;
    getBBO(symbol, bpx,bsz,apx,asz);
    return bpx;

}

static inline
double getAskPrice(const std::string& symbol) {
    int bsz = 0,asz = 0;
    double bpx = 0, apx = 0;
    getBBO(symbol, bpx,bsz,apx,asz);
    return apx;
}

static inline
int getBidSize(const std::string& symbol) {
    int bsz = 0,asz = 0;
    double bpx = 0, apx = 0;
    getBBO(symbol, bpx,bsz,apx,asz);
    return bsz;
}

static inline
int getAskSize(const std::string& symbol) {
    int bsz = 0,asz = 0;
    double bpx = 0, apx = 0;
    getBBO(symbol, bpx,bsz,apx,asz);
    return asz;
}

// price string could be in the form of 
// [a|b][+|-][t|s]integer
// For example, 
// "a+t1", 1 tick above ask, or
// "b-s2", 2 spreads down the bid, 
//           where the spreads is the current spread size
//           measure by askpx-bidpx
// symbol is in form of venue/tradable, or tradable or MTS symnbol
// px_str is a null-terminating string
static inline
bool getPriceByStr(const std::string& symbol, const char* px_str, double& px) {
    BookConfig* bcfg;
    try {
        bcfg = new BookConfig(symbol, "L1");
    } catch (const std::exception& e) {
        logError("Failed to get price from string %s.  %s", px_str, e.what());
        return false;
    }
    const auto* ti(utils::SymbolMapReader::get().getByTradable(bcfg->symbol));

    int cnt=-1;
    bool side, sign, ticktype;
    const char* ptr = px_str;
    while (*ptr) {
        if ((*ptr != 'a') && (*ptr != 'b')) {
            ++ptr;
            continue;
        }
        side = (*ptr++ == 'b');
        break;
    }
    if (! *ptr) {
        try {
            px = std::stod(px_str);
            const int sn { std::signbit(px)?-1:1 };
            px = (int) (std::abs(px) / ti->_tick_size + 0.5) * sn * ti->_tick_size;
            logDebug("Parsed to a price string: %s - %lf", px_str, px);
            return true;
        } catch (const std::exception& e) {
            logError("price string parsing error: cannot find ask/bid AND not a price: %s", px_str);
            return false;
        }
    }

    while (*ptr) {
        if ((*ptr != '+') && (*ptr != '-')) {
            ++ptr;
            continue;
        }
        sign = (*ptr++ == '+');
        break;
    }
    if (! *ptr) {
        logError("price string parsing error: cannot find sign: %s", px_str);
        return false;
    }

    while (*ptr) {
        if ((*ptr != 's') && (*ptr != 't')) {
            ++ptr;
            continue;
        }
        ticktype = (*ptr++ == 's');
        break;
    }
    if (! *ptr) {
        logError("price string parsing error: cannot find unit type (tick or spread): %s", px_str);
        return false;
    }

    try {
        cnt = std::stoll(ptr);
    } catch (const std::exception& e) {
        logError("price string parsing error: cannot parse the count: %s", px_str);
        return false;
    }

    double bidpx, askpx;
    int bidsz, asksz;
    if (! getBBO(symbol, bidpx, bidsz, askpx, asksz)) {
        return false;
    }
    // check if the market has any quotes on the 'side'
    if ((side?bidsz:asksz) == 0) {
        logError("there is no quotes on the %s side, cannot make price string from %s", 
                (side?"bid":"ask"), px_str);
        return false;
    }
    px = (side? bidpx:askpx);
    double diff = askpx-bidpx;
    if (!ticktype) {
        diff = ti->_tick_size;
    }
    if (!sign) diff = -diff;
    px += diff*cnt;

    logDebug("parsed price string %s: side: %d, sign: %d, ticktype(spread or tick): %d, cnt: %d, "
            "[bpx,bsz,apx,asz]: [%.4lf,%lld,%.4lf,%lld], diff: %.4lf, px: %.4lf", 
            px_str, (int)side, (int)sign, (int)ticktype, cnt, bidpx, bidsz, askpx, asksz, 
            diff, px);
    return true;
};

}  // namespace md
