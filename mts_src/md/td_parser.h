#pragma once

#include <stdio.h>
#include <string>
#include <utility>
#include <functional>
#include <tuple>
#include "md_bar_price.h"

#define TickDataBookLevel 1

namespace md {
template <typename DType>
struct Array2D {
    // thin layer for c-style 2D array
    // no limit is checked
    Array2D(DType* d, size_t rows, size_t cols):
        _d(d), _N(rows), _M(cols) {};
    const DType* operator[](int i) const {
        return &_d[i*_M];
    }
    DType* operator[](int i) {
        return &_d[i*_M];
    }
    double* _d;
    size_t _N, _M;
};

class TickData2Bar {
public:
    TickData2Bar(const std::string& quote_csv,
                 const std::string& trade_csv,
                 time_t start_utc,
                 time_t end_utc,
                 int barsec,
                 double tick_size=0):
    _quote_csv(quote_csv), _trade_csv(trade_csv),
    _start_utc(start_utc), _end_utc(end_utc),
    _barsec(barsec), _tick_size(tick_size),
    _nullwtr(std::make_shared<NullOutput>())
    { };

    bool parse(std::shared_ptr<std::vector<std::string>>& bars) {
        using BarOutType = BarPriceOutput<StringVectorOutput>;

        auto str_out(std::make_shared<StringVectorOutput>(bars));
        auto bar_out(std::make_shared<BarOutType> (
                    _start_utc, _end_utc, _barsec, 
                    str_out, _nullwtr, _tick_size, true)
                );
        return get_bar(_quote_csv, _trade_csv, _start_utc, _end_utc, _barsec, bar_out);
    };

    bool parse(const std::string& out_csv_file) {
        using BarOutType = BarPriceOutput<FileOutput>;

        auto file_out(std::make_shared<FileOutput>(out_csv_file, true, false));
        auto bar_out(std::make_shared<BarOutType>(_start_utc, _end_utc, _barsec, 
                     file_out, _nullwtr, _tick_size, true));
        return get_bar(_quote_csv, _trade_csv, _start_utc, _end_utc, _barsec, bar_out);
    }

    template<typename QuoteType, typename TradeType>
    bool parseArray(const QuoteType& q, const TradeType& t,
            size_t qlen, size_t tlen,
            const std::string& out_csv_file) {
        using BarOutType = BarPriceOutput<FileOutput>;

        auto file_out(std::make_shared<FileOutput>(out_csv_file, true, false));
        auto bar_out(std::make_shared<BarOutType>(_start_utc, _end_utc, _barsec, 
                     file_out, _nullwtr, _tick_size, true));
        BookWriter<BarOutType, TickDataBookLevel> dump(bar_out);

        return get_bar_array(q, t, qlen, tlen, _start_utc, _end_utc, _barsec, dump);
    }

    bool parseDoubleArray(const double* q, const double* t,
            size_t qlen, size_t tlen,
            const std::string& out_csv_file) {
        FileOutput bars(out_csv_file, true, false);
        Array2D<double> qa((double*)q, qlen, 5);
        Array2D<double> ta((double*)t, tlen, 3);

        using BarOutType = BarPriceOutput<FileOutput>;

        auto file_out(std::make_shared<FileOutput>(out_csv_file, true, false));
        auto bar_out(std::make_shared<BarOutType>(_start_utc, _end_utc, _barsec,
                    file_out, _nullwtr, _tick_size, true));

        BookWriter<BarOutType, TickDataBookLevel> dump(bar_out);

        return get_bar_array(qa, ta, qlen, tlen, _start_utc, _end_utc, _barsec, dump);
    }

    // getting the tick-by-tick dump
    bool tickDump(std::shared_ptr<std::vector<std::string>>& tickdump) {
        using BarOutType = BarPriceOutput<NullOutput, StringVectorOutput>;

        auto str_out(std::make_shared<StringVectorOutput>(tickdump));
        auto bar_out(std::make_shared<BarOutType>(_start_utc, _end_utc, _barsec,
                     _nullwtr, str_out, _tick_size, true));
        return get_bar(_quote_csv, _trade_csv, _start_utc, _end_utc, _barsec, bar_out);
    };

    bool tickDump(const std::string& tickdump_file) {
        using BarOutType = BarPriceOutput<NullOutput, FileOutput>;

        auto file_out(std::make_shared<FileOutput>(tickdump_file, true, false));
        auto bar_out(std::make_shared<BarOutType>(_start_utc, _end_utc, _barsec,
                    _nullwtr, file_out, _tick_size, true));
        return get_bar(_quote_csv, _trade_csv, _start_utc, _end_utc, _barsec, bar_out);
    };

private:
    static const int cts = 0, cbp = 1, cbs = 2, cap = 3, cas = 4, // quote columns
              cpx = 1, csz = 2;  // trade columns
    static const int quote_cols = 5, trade_cols = 3;

    const std::string _quote_csv, _trade_csv;
    const time_t _start_utc, _end_utc;  // start_utc and end_utc, first bar close at start_utc+barsec
    const int _barsec;
    const double _tick_size;
    std::shared_ptr<NullOutput> _nullwtr;

    template<typename QuoteType, typename DumpType>
    void update_quote(const QuoteType& q, DumpType& dump) {
        long long cur_micro = ((long long)(q[cts]+0.5))*1000LL; //q[cts] in milli
        double bpx = q[cbp], apx = q[cap];
        int bsz = (int)(q[cbs]+0.5), asz = (int)(q[cas]+0.5);

#ifdef TD_PARSER_TRACE
        if (cur_micro/1000000LL == 1685714489LL) {
        FILE* fp=fopen("/tmp/td_parser.csv","at");
        fprintf(fp, "%lld,0,%lf,%d,%lf,%d,%s\n",cur_micro,bpx,bsz,apx,asz,bp.toString().c_str());
        fclose(fp);
        }
#endif
        dump.updBBO(bpx, bsz, apx, asz, cur_micro);
    }

    // update the normal trade
    template<typename TradeType, typename DumpType>
    void update_trade(const TradeType& t, bool is_buy, DumpType& dump) {
        long long cur_micro = ((long long)(t[cts]+0.5))*1000LL; //q[cts] in milli
        double tpx = t[cpx];
        uint32_t tsz = (uint32_t)(t[csz]+0.5);

#ifdef TD_PARSER_TRACE
        if (cur_micro/1000000LL == 1685714489LL) {
        FILE* fp=fopen("/tmp/td_parser.csv","at");
        fprintf(fp, "%lld,1,%lf,%d,0,0,%s\n",cur_micro,tpx,is_buy?tsz:-tsz,bp.toString().c_str());
        fclose(fp);
        }
#endif
        dump.updTrade_Dir(tpx, tsz, is_buy, cur_micro);
    };

    // update normal trade with undecided direction
    template<typename TradeType, typename DumpType>
    void update_trade(const TradeType& t, DumpType& dump) {
        long long cur_micro = ((long long)(t[cts]+0.5))*1000LL; //q[cts] in milli
        double tpx = t[cpx];
        uint32_t tsz = (uint32_t)(t[csz]+0.5);

#ifdef TD_PARSER_TRACE
        if (cur_micro/1000000LL == 1685714489LL) {
        FILE* fp=fopen("/tmp/td_parser.csv","at");
        fprintf(fp, "%lld,2,%lf,%d,0,0,%s\n",cur_micro,tpx,tsz,bp.toString().c_str());
        fclose(fp);
        }
#endif
        dump.updTrade_NoDir(tpx, tsz, cur_micro);
    };

    // update special trade with no direction
    template<typename TradeType, typename DumpType>
    void update_trade_special(const TradeType& t, int type, DumpType& dump) {
        long long cur_micro = ((long long)(t[cts]+0.5))*1000LL; //q[cts] in milli
        double tpx = t[cpx];
        uint32_t tsz = (uint32_t)(t[csz]+0.5);

#ifdef TD_PARSER_TRACE
        if (cur_micro/1000000LL == 1685714489LL) {
        FILE* fp=fopen("/tmp/td_parser.csv","at");
        fprintf(fp, "%lld,2,%lf,%d,0,0,%s\n",cur_micro,tpx,tsz,bp.toString().c_str());
        fclose(fp);
        }
#endif
        dump.updTrade_Special(tpx, tsz, type, cur_micro);
    };

    template<typename QuoteType>
    BBOTuple get_reducing(const QuoteType& prev_quote, const QuoteType& this_quote) {
        const BBOTuple& prev_bbo ={ prev_quote[cbp], (uint32_t)(prev_quote[cbs]+0.5),
                                    prev_quote[cap], (uint32_t)(prev_quote[cas]+0.5) };
        const BBOTuple& this_bbo ={ this_quote[cbp], (uint32_t)(this_quote[cbs]+0.5),
                                    this_quote[cap], (uint32_t)(this_quote[cas]+0.5) };
        return getBBOReducing(prev_bbo, this_bbo);
    }

    // QuoteType could be double** or double(*)[5], or vector<vector<double>>
    // TradeTu[e could be double** or double(*)[3], or vector<vector<double>>
    // BarDumpType could be a vector<std::string>, BarWriter or NullOutput
    template<typename QuoteType, typename TradeType, typename DumpType>
    bool get_bar_array(const QuoteType& q,
                 const TradeType& t,
                 size_t qlen,
                 size_t tlen,
                 const time_t start_utc,
                 const time_t end_utc,
                 const int barsec,
                 DumpType& dump) {

        size_t tix = 0, qix = 0;
        // initialze the first qix and tix
        // remove any tix earlier than qix

        // adjust qlen, tlen, qix, tix according to end_milli and start_milli
        long long start_milli = start_utc*1000LL;
        long long end_milli = end_utc*1000LL;
        size_t ix=qlen-1; for (; (q[ix][cts]>end_milli) && (ix>qix) ; --ix);   qlen=ix+1;
        ix=tlen-1;        for (; (t[ix][cts]>end_milli) && (ix>tix) ; --ix);   tlen=ix+1;
        ix=0;             for (; (q[ix][cts]<start_milli) && (ix<qlen); ++ix); qix=ix>0?ix-1:0;
        ix=0;             for (; (t[ix][cts]<start_milli) && (ix<tlen); ++ix); tix=ix;

        // check there are ticks 
        if (__builtin_expect(qix>=qlen-1 ,0)) {
            printf("no quote ticks found between  [%d,%d]\n", (int)start_utc, (int) end_utc);
            return false;
        }
        // update the first quote to initialize the bar
        update_quote(q[qix], dump);

        // skip any trades before qix
        while ((tix<tlen)&& (t[tix][cts] < q[qix][cts])) {
            ++tix;
        }
        int qix_upd = ++qix; // last updated qix
        bool is_buy = false; // save the previous trade direction, for swipe cases
        while ((tix<tlen) && (qix<qlen)) {
            /* verbose
            if (__builtin_expect((tix%1000==0)||(qix%1000==0),0)) {
                printf("qix(%d/%d), tix(%d/%d)\n", (int)qix, (int)qlen, (int)tix, (int)tlen);
            } */

            // update all qix before tix, if any
            if (q[qix][cts]<t[tix][cts]) {
                update_quote(q[qix], dump);
                qix_upd = ++qix;
                continue;
            }

            // match trade with quote at the same milli-second
            int qix0 = -1; // an initial insertion point for the trade
            bool is_buy0 = is_buy; // save the dir with qix0
            uint32_t r_sz = 0;
            const double tpx = t[tix][cpx], tsz = t[tix][csz];
            while ((qix<qlen) && (q[qix][cts] == t[tix][cts])) {
                const auto [br_px,br_sz,ar_px,ar_sz] = get_reducing(q[qix-1], q[qix]);
                // set r_sz from quote reducing size match with tpx
                // the tpx is same or better, for case of swipe
                // 
                if ( (br_sz!=0) && (tpx-br_px<1e-10)) {
                    r_sz = br_sz;
                    is_buy = false; // sell
                } else {
                    if ( (ar_sz!=0) && (ar_px-tpx<1e-10)) {
                        r_sz = ar_sz;
                        is_buy = true; // buy
                    }
                }
                // if reducing at the tpx
                //     set qix0 if qix0 is -1
                //     if match sz, 
                //        set qix0 and break
                if (r_sz != 0) {
                    // save the initial reducing ix, in case no match
                    // found, this will be used as the best match
                    if (qix0==-1) {
                        qix0 = qix;
                        is_buy0 = is_buy;
                    }
                    if (r_sz == tsz) {
                        // exact match!
                        qix0 = qix;
                        is_buy0 = is_buy;
                        break;
                    }
                }
                ++qix;
            }
            // if qix0 is -1, case of no matching time/px found before or at t_utc,
            // update the trade and then update all quotes before or at it
            // otherwise, tix is inserted before qix0
            
            if (__builtin_expect(r_sz == 0,0)) {
                qix0 = _MAX_(qix_upd-1,0);
                auto ts0 = t[tix][cts];

                /* 
                 * Experimented _NoDir, not better match to bpipe
                 */
                // decide the trade direction
                bool no_dir=false;
                double bdiff = std::abs(tpx-q[qix0][cbp]), adiff = std::abs(tpx-q[qix0][cap]);
                if (bdiff>adiff+1e-10) {
                    is_buy = true;
                } else {
                    if (bdiff<adiff-1e-10) {
                        is_buy = false;
                    } else {
                        no_dir=true;
                    }
                } // else - use the previous direction
                // update trade first, it's likely that it is a swipe
                if (__builtin_expect(!no_dir,1)) {
                    update_trade(t[tix], is_buy, dump);
                } else {
                    update_trade(t[tix], dump);
                }
                ++tix;

                // apply all trades with the same tpx at the same milli
                while ((tix<tlen)&&(t[tix][cts]==ts0)&&(std::abs(t[tix][cpx]-tpx)<1e-10)) {
                    if (!no_dir) {
                        update_trade(t[tix], is_buy, dump);
                    } else {
                        update_trade(t[tix], dump);
                    }
                    ++tix;
                }

                // getting back to where we started 
                qix=qix_upd;
            } else {
                // if there were match
                // update from qix_upd+1 to qix0-1
                // and then update the trade at tix
                // take care of swipe
                // then update the quote at qix0
                
                // is_buy0 is the saved dir for qix0
                // this is needed in case match is not exact, is_buy could
                // flip later in the same milli that is not intended for qix0
                is_buy=is_buy0;
                for (; qix_upd<qix0; ++qix_upd) {
                    update_quote(q[qix_upd], dump);
                }
                update_trade(t[tix], is_buy, dump);
                ++tix;

                // check in case of swipe - 
                // update all tix that is 
                // 1. same milli
                // 2. increasing tpx
                // 3. strictly less than new quote px
                //
                const double pmul = is_buy?1.0:-1.0;
                auto new_px = q[qix0][is_buy?cap:cbp];
                if (__builtin_expect((new_px-tpx)*pmul>1e-10,0)) {
                    // swipe case
                    auto ts0 = t[tix-1][cts];
                    new_px*=pmul;
                    auto tpx0 = tpx*pmul;
                    while ((tix<tlen)&&(t[tix][cts]==ts0)) {
                        auto tpx1 = t[tix][cpx]*pmul;
                        if ((tpx1>tpx0-1e-10)&&(tpx1<new_px-1e-10)) {
                            // apply this if
                            // this trade at px that are same or more aggressive with tpx0
                            // but is not at the level of the new quote
                            update_trade(t[tix], is_buy, dump);
                            ++tix;
                            tpx0 = tpx1;
                        } else {
                            break;
                        }
                    }
                }


                qix = qix0;
                if (tsz == r_sz) {
                    // update quote at qix0 if there were exact matching
                    update_quote(q[qix_upd++], dump);
                    qix = qix0+1;
                }
            }
        }
        // update remaining any tix/qix
        for(; qix<qlen; ++qix) {
            update_quote(q[qix], dump);
        }
        for (; tix<tlen; ++tix) {
            qix = qlen-1;
            double tpx = t[tix][cpx];
            // decide on a trade direction
            double bdiff = std::abs(tpx-q[qix][cbp]), adiff = std::abs(q[qix][cap]-tpx);
            if (bdiff > adiff+1e-10) {
                is_buy = true;
            } else {
                if (bdiff < adiff-1e-10) {
                    is_buy=false;
                }
            }
            update_trade(t[tix], is_buy, dump);
        }

        // last bar taken care of by BarPriceOutput
        return true;
    };

    template<typename BarOutType>
    bool get_bar(const std::string& quote_csv,
                 const std::string& trade_csv,
                 time_t start_utc,
                 time_t end_utc,
                 int barsec, 
                 std::shared_ptr<BarOutType>& bar_out) {

        const auto& qs (utils::CSVUtil::read_file(quote_csv));
        const auto& ts (utils::CSVUtil::read_file(trade_csv));

        // note: q/t type is not double**, but double(*)[N]
        auto q = new double[qs.size()][quote_cols];
        auto t = new double[ts.size()][trade_cols];

        // populate the q/t
        for (size_t i=0; i<qs.size(); ++i) {
            // take care of time stamp
            if (qs[i][0].find('-') != std::string::npos) {
                q[i][0] = utils::TimeUtil::string_to_frac_UTC(qs[i][0].c_str(), 3);
            } else {
                q[i][0] = std::stod(qs[i][0]);
            }
            for (int k=1; k<quote_cols; ++k) {
                q[i][k] = std::stod(qs[i][k]);
            }
        }
        // populate the q/t
        for (size_t i=0; i<ts.size(); ++i) {
            if (ts[i][0].find('-') != std::string::npos) {
                t[i][0] = utils::TimeUtil::string_to_frac_UTC(ts[i][0].c_str(), 3);
            } else {
                t[i][0] = std::stod(ts[i][0]);
            }
            for (int k=1; k<trade_cols; ++k) {
                t[i][k] = std::stod(ts[i][k]);
            }
        }

        BookWriter<BarOutType, TickDataBookLevel> dump(bar_out);
        bool ret = get_bar_array(q, t, qs.size(), ts.size(), start_utc, end_utc, barsec, dump);
        delete []q;
        delete []t;
        return ret;
    };
};
}
