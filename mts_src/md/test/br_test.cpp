#include "td_parser.h"
#include "gtest/gtest.h"
#include <fstream>

const char* CFGFile = "/tmp/main.cfg";
const char* VENUEFile = "/tmp/venue.cfg";

void setupCfg() {
    {
        // write the main config to /tmp/main.cfg
        std::ofstream ofs;
        ofs.open (CFGFile, std::ofstream::out | std::ofstream::trunc);
        ofs << "Logger = /tmp/log" << std::endl;
        ofs << "BarPath = /tmp" << std::endl;
        ofs << "HistPath = /tmp" << std::endl;
        ofs << "BarSec = [ 5 ]" << std::endl;
        ofs << "SymbolMap = " << VENUEFile << std::endl;
    }
    {
        // write the venue config to /tmp/venue.cfg
        std::ofstream ofs;
        ofs.open (VENUEFile, std::ofstream::out | std::ofstream::trunc);

        ofs << "tradable = {\n";
        ofs << "    CLQ1 = {\n";
        ofs << "        symbol = WTI\n";
        ofs << "        exch_symbol = CL\n";
        ofs << "        venue = NYM\n";
        ofs << "        tick_size = 0.010000000000\n";
        ofs << "        point_value = 1000.0\n";
        ofs << "        px_multiplier = 0.010000000000\n";
        ofs << "        type = FUT\n";
        ofs << "        mts_contract = WTI_202108\n";
        ofs << "        contract_month = 202108\n";
        ofs << "        mts_symbol = WTI_N1\n";
        ofs << "        N = 1\n";
        ofs << "        expiration_days = 11\n";
        ofs << "        tt_security_id = 9596025206223795780\n";
        ofs << "        tt_venue = CME\n";
        ofs << "        currency = USD\n";
        ofs << "        expiration_date = 2021-07-20\n";
        ofs << "        bbg_id = CLQ1 COMDTY\n";
        ofs << "        bbg_px_multiplier = 1.0\n";
        ofs << "        tickdata_id = CLQ21\n";
        ofs << "        tickdata_px_multiplier = 1.000000000000\n";
        ofs << "        tickdata_timezone = America/New_York\n";
        ofs << "        lotspermin = 40\n";
        ofs << "    }\n";
        ofs << "}\n";
        ofs << "venue = {" << std::endl;
        ofs << "NYM   = { hours = [ -6, 0, 17, 0 ]   } " << std::endl;
        ofs << "ICE   = { hours = [ -4, 30, 17, 15 ] } " << std::endl;
        ofs << "}" << std::endl;
    }
    // set the config path NOW!
    utils::PLCC::setConfigPath(CFGFile);
}

class BRFixture : public testing::Test {
public:
    BRFixture ():
    _bar("0, 40.0, 40.1, 39.9, 40.05, 0, 0, 0, 0"),
    _bcfg("", "WTI_N1", "L1"),
    _barsec(5)
    {
        {
            // clear the bar file
            std::ofstream ofs;
            ofs.open(_bcfg.bfname(_barsec), std::ofstream::out | std::ofstream::trunc);
        }
    }

    void writeBar(time_t t, double px ) {
        _bar.update((long long)t*1000000LL, px, 1, 2, 0,0,0,0);
        std::string line = _bar.writeAndRoll(t);
        {
            std::ofstream ofs;
            ofs.open(_bcfg.bfname(_barsec), std::ofstream::out | std::ofstream::app);
            ofs << line << std::endl;
        }
    }

    bool double_eq(double v1, double v2) const {
        return (long long) (v1*1e+10+0.5) == (long long) (v2*1e+10+0.5);
    }

    std::string write_file(const std::vector<std::string>& csv_lines) const {
        char fn [256];
        snprintf(fn,sizeof(fn),"/tmp/br_test%lld.csv", (long long) utils::TimeUtil::cur_micro());
        FILE* fp = fopen(fn, "wt");
        for(const auto& line:csv_lines) {
            fprintf(fp,"%s\n",line.c_str());
        }
        fclose(fp);
        return std::string(fn);
    }

    bool line_compare(const std::vector<std::string>& line, const std::string& ref) const {
        if (line.size() == 0) return ref=="";
        char buf[256];
        size_t cnt = sprintf(buf, "%s", line[0].c_str());
        for (size_t i=1; i<line.size(); ++i) {
            cnt += snprintf(buf+cnt, sizeof(buf)-cnt, ", %s",line[i].c_str());
        }
        return ref == std::string(buf);
    }

protected: 
    md::BarPrice _bar;
    md::BookConfig _bcfg;
    int _barsec;
};

TEST_F (BRFixture, BookDepot) {
    std::string line = "1675198786000000, 0, 0, 101.2, 3, 0, 1, 101.1, 11, 1, 101.2, 2";
    md::BookDepotLevel<1> book(line);
    EXPECT_TRUE(book.isValid());
    EXPECT_DOUBLE_EQ(book.getMid(), 101.15);

    //EXPECT_EQ(book, md::BookDepot(book.toCSV()));

    book.addTrade(101.1, 1, false, 1675198786000000ULL);
    EXPECT_EQ(book.trade_attr, 1);

    EXPECT_DOUBLE_EQ(book.getMid(), 101.15);
    const auto& line1(book.toCSV(1));
    const auto& line2(md::BookDepotLevel<1>(book.toCSV(1)).toCSV(1));
    for (size_t i=0; i<line1.size(); ++i) {
        EXPECT_EQ(line1[i], line2[i]);
    }
}

TEST_F (BRFixture, RunSecond) {
    // run a tick-by-tick csv file, format of booktap dump:
    // upd_micro, upd_type, upd_level, last_px, last_sz, last_trd_attr, bid_levels, bpx, bsz, ask_levels, apx, asz
    // 1675285201000659, 1, 0, 4133.5000000, 42, 1, 1, 4133.7500000, 36, 1, 4134.0000000, 10
    const std::string csv_file = "tests/bar_test.csv";
    const auto& lines(utils::CSVUtil::read_file(csv_file));
    md::BarPrice bar;
    for (const auto& line: lines) {
        bar.update<1>(line);
    }
    auto last_micro = std::stoll(lines[lines.size()-1][0]); 
    auto bar_line = bar.writeAndRoll(last_micro/1000000LL+1);
    // repo_line shown as
    std::string repo_line="1675285202, 4133.875, 4134.0, 4133.5, 4133.625, 342, 4133.5, 1675285201072262, -180, 67.4, 163.3, 0.250193, 98, 281";
    EXPECT_EQ(bar_line, repo_line);
}

TEST_F (BRFixture, Extended) {
    // utc, open, high, low, close, totvol, lastpx, last_micro, vbs
    // extended: avg_bsz, avg_asz, avg_spd, bqdiff, aqdiff
    std::string line0 = "1, 50.1, 50.1, 50.1, 50.1, 0, 0, 0, 0";
    std::string line1 = "1, 50.1, 50.1, 50.1, 50.1, 0, 0, 0, 0, 4, 3, 0.2, 0, 0";
    std::vector<md::BarPrice> bars = 
        {md::BarPrice(), md::BarPrice(line0), md::BarPrice(line1) };

    /*
    void update(long long cur_micro, double price, int32_t volume, int update_type,
                double bidpx, int bidsz, double askpx, int asksz)
    */

    for (int k = 0; k<3; ++k) {
        md::BarPrice& bar(bars[k]);
        long long t = 1000000; // start at a whole second 
        long long t0 = t;

        // all from t0
        bar.update(t, 50.1, 0, 0, 50.0, 4, 50.2, 3);
        for (int i=0; i<3; ++i) {
            bar.update(++t, 50.1, 0, 1, 50.0, 4, 50.2, 2); // ask sz -1
            bar.update(++t, 50.1, 0, 0, 50.0, 5, 50.2, 2); // bid sz +1
            bar.update(++t, 50.1, 2, 2, 50.0, 5, 50.2, 2); // buy 2
            bar.update(++t, 50.15, 2, 1, 50.0, 5, 50.3, 5); // quote for buy 2

            EXPECT_EQ(bar.bqd, 1);
            EXPECT_EQ(bar.aqd, -1);

            //EXPECT_TRUE(double_eq(bar.spd, 0.8));
            //EXPECT_EQ(bar.bsz, 18);
            //EXPECT_EQ(bar.asz, 9);
            //EXPECT_EQ(bar.cumsum_micro, 4);

            EXPECT_TRUE(double_eq(bar.avg_bsz(t), 4.5));
            EXPECT_TRUE(double_eq(bar.avg_asz(t), 2.25));
            EXPECT_TRUE(double_eq(bar.avg_spd(t), 0.2));

            bar.update(++t, 50.2, 2, 0, 50.1, 1, 50.3, 5); // bid + 1
            bar.update(++t, 50.2, 2, 0, 50.1, 1, 50.3, 10); // ask + 5
            bar.update(++t, 50.2, -7, 2, 50.1, 1, 50.3, 10); // sell 7
            bar.update(++t, 50.1, 7, 0, 49.9, 3, 50.3,5); // quote for sell 7

            EXPECT_EQ(bar.bqd, 2);
            EXPECT_EQ(bar.aqd, 4);

            //EXPECT_TRUE(double_eq(bar.spd, 1.7));
            //EXPECT_EQ(bar.bsz, 26);
            //EXPECT_EQ(bar.asz, 39);
            //EXPECT_EQ(bar.cumsum_micro, 8);

            EXPECT_TRUE(double_eq(bar.avg_bsz(t+1), (26.0+3.0)/9.0));
            EXPECT_TRUE(double_eq(bar.avg_asz(t+1), (39.0+5.0)/9.0));
            EXPECT_TRUE(double_eq(bar.avg_spd(t+1), (1.7+0.4)/9.0));

            bar.update(++t, 50.05, 7, 1, 49.9, 3, 50.2, 2); // ask + 2
            bar.update(++t, 50.05, 7, 0, 49.9, 1, 50.2, 2); // bid -2
            bar.update(++t, 50.0, 7, 1, 49.9, 1, 50.1, 1); // ask + 1
            bar.update(++t, 50.05, 7, 1, 49.9, 1, 50.2, 2); // ask - 1

            EXPECT_EQ(bar.bqd, 0);
            EXPECT_EQ(bar.aqd, 6);
            //EXPECT_TRUE(double_eq(bar.spd, 2.9));
            //EXPECT_EQ(bar.cumsum_micro, 12);
            EXPECT_TRUE(double_eq(bar.avg_spd(t+1), (2.9+0.3)/13.0));

            t = t0 + (i+1)*1000000;
            // get back to the first quote before roll
            bar.update(t, 50.1, 0, 0, 50.0, 4, 50.2, 3);  // bid+4,ask+1
            auto line = bar.writeAndRoll( t/1000000LL );
            md::BarPrice bar0(line);

            EXPECT_EQ(bar0.bvol, 2);
            EXPECT_EQ(bar0.svol, 7);
            EXPECT_TRUE(double_eq(bar0.open, 50.1));
            EXPECT_TRUE(double_eq(bar0.high, 50.2));
            EXPECT_TRUE(double_eq(bar0.low, 50.0));
            EXPECT_TRUE(double_eq(bar0.close, 50.1));
            EXPECT_TRUE(double_eq(bar0.last_price, 50.2));
            EXPECT_EQ(bar0.last_micro, 1000007LL + i*1000000LL);

            EXPECT_TRUE(double_eq(bar0.last_price, 50.2));
            EXPECT_EQ(bar0.bqd, 4);
            EXPECT_EQ(bar0.aqd, 7);

            // bar already rolled, check at the rolling time
            // should get the latest update at t
            EXPECT_TRUE(double_eq(bar.avg_bsz(t), 4.0));
            EXPECT_TRUE(double_eq(bar.avg_asz(t), 3.0));
            EXPECT_TRUE(double_eq(bar.avg_spd(t), 0.2));
            EXPECT_EQ(bar.bvol, 0);
            EXPECT_EQ(bar.svol, 0);

            //printf("%s\n", line.c_str());
        }
    }
}


TEST_F (BRFixture, ReadBackwardForward) {

    // having the bar time as  { -20, -10, -5, 0, 3605, 3615}
    std::vector<int> bt { -20, -10, -5, 0, 3605};
    time_t utc_now = (time_t)utils::TimeUtil::string_to_frac_UTC("20201207-17:00:00",0);
    double px = 40.01;
    for (auto& b : bt) {
        px+=0.1;
        writeBar(utc_now + b, px);
    }
    md::BarReader br (_bcfg, _barsec);

    // read latest bar
    md::BarPrice bp;
    EXPECT_FALSE(br.read(bp));
    px+=0.1;
    writeBar(utc_now + 3615, px);
    EXPECT_TRUE(br.read(bp));
    EXPECT_DOUBLE_EQ(bp.close, px);
    EXPECT_EQ(bp.bar_time, utc_now + 3615);

    // read latest 2 bars, 3610 is forward filled by 3605
    std::vector<std::shared_ptr<md::BarPrice> > barHist;
    EXPECT_TRUE(br.readLatest(barHist, 2));
    EXPECT_EQ(bp.bar_time, barHist[1]->bar_time);
    EXPECT_DOUBLE_EQ(bp.close, barHist[1]->close);
    EXPECT_EQ(utc_now + 3610, barHist[0]->bar_time);
    EXPECT_DOUBLE_EQ(px-0.1, barHist[0]->close);

    // read -15 TO 0 as history, -15 is forward filled by -20
    EXPECT_TRUE(br.readPeriod(barHist, utc_now-15, utc_now));
    EXPECT_EQ(barHist.size(), 6);
    EXPECT_EQ(barHist[5]->bar_time, utc_now);
    EXPECT_EQ(barHist[4]->bar_time, utc_now-5);
    EXPECT_EQ(barHist[3]->bar_time, utc_now-10);
    EXPECT_EQ(barHist[2]->bar_time, utc_now-15);
    EXPECT_DOUBLE_EQ(barHist[5]->close, px-0.2);
    EXPECT_DOUBLE_EQ(barHist[2]->close, px-0.5);
    
    // read -25 TO -15, -25 backward filled
    EXPECT_TRUE(br.readPeriod(barHist, utc_now-25, utc_now-15));
    EXPECT_EQ(barHist.size(), 9);
    EXPECT_EQ(barHist[8]->bar_time, utc_now-15);
    EXPECT_EQ(barHist[7]->bar_time, utc_now-20);
    EXPECT_EQ(barHist[6]->bar_time, utc_now-25);
    EXPECT_DOUBLE_EQ(barHist[8]->close, px-0.5);
    EXPECT_DOUBLE_EQ(barHist[7]->close, px-0.5);
    EXPECT_DOUBLE_EQ(barHist[6]->close, px-0.5);

    // read 10 to 20, 10 forward filled from 5, 20 by 15
    EXPECT_TRUE(br.readPeriod(barHist, utc_now+3610, utc_now+3620));
    EXPECT_EQ(barHist.size(), 12);
    EXPECT_EQ(barHist[11]->bar_time, utc_now+3620);
    EXPECT_EQ(barHist[10]->bar_time, utc_now+3615);
    EXPECT_EQ(barHist[9]->bar_time, utc_now+3610);
    EXPECT_DOUBLE_EQ(barHist[11]->close, px);
    EXPECT_DOUBLE_EQ(barHist[10]->close, px);
    EXPECT_DOUBLE_EQ(barHist[9]->close, px-0.1);
}

TEST_F (BRFixture, Tickdata) {
    // case 1 the normal case

    const std::vector<std::string> quote0 = {
            "1675281149882, 4126.50000, 10, 4127.000000, 11", // removed
            "1675281150882, 4126.750000, 1, 4127.000000, 11", // first before
            "1675281151884, 4126.500000, 19, 4127.000000, 11", 
            "1675281151884, 4126.500000, 19, 4127.000000, 12", // qad +1
            "1675281151884, 4126.500000, 20, 4127.000000, 12", // qbd +1
            "1675281151893, 4126.500000, 18, 4127.000000, 12",
            "1675281151893, 4126.500000, 17, 4127.000000, 12",
            "1675281151893, 4126.500000, 18, 4127.000000, 12"  // qbd +1
    };
    const std::vector<std::string> trade0 = {
            "1675281150999, 4126.750000, 1", // removed
            "1675281151884, 4126.750000, 1", // sell
            "1675281151893, 4126.500000, 2", // sell
            "1675281151893, 4126.500000, 10", // sell
            "1675281153893, 4126.500000, 1", // sell, after last quote
            "1675281155893, 4126.500000, 2" // removed after last utc
    };
    time_t utc_start = 1675281151;
    time_t utc_end =   1675281154;
    std::string quote_file = write_file(quote0);
    std::string trade_file = write_file(trade0);

    try {
        md::TickData2Bar tp(quote_file, trade_file, utc_start, utc_end, 1);
        auto bar_sp = std::make_shared<std::vector<std::string>>();
        std::vector<std::string>& bars(*bar_sp);
        EXPECT_TRUE(tp.parse(bar_sp));

        // expect 3 bars
        EXPECT_EQ(bars[0], "1675281152, 4126.875, 4126.875, 4126.5, 4126.75, 13, 4126.5, 1675281151893000, -13, 2.8, 11.1, 0.275939, 2, 1, 0, -9");
        EXPECT_EQ(bars[1], "1675281153, 4126.75, 4126.75, 4126.75, 4126.75, 0, 4126.5, 1675281151893000, 0, 18.0, 12.0, 0.500000, 0, 0, 0, 0");
        EXPECT_EQ(bars[2], "1675281154, 4126.75, 4126.75, 4126.5, 4126.5, 1, 4126.5, 1675281153893000, -1, 18.0, 12.0, 0.500000, 0, 0, 0, 0");

        // work on a double**
        const double quotea[8][5] = {
            {1675281149882, 4126.50000, 10, 4127.000000, 11}, // removed
            {1675281150882, 4126.750000, 1, 4127.000000, 11}, // first before
            {1675281151884, 4126.500000, 19, 4127.000000, 11}, 
            {1675281151884, 4126.500000, 19, 4127.000000, 12}, // qad +1
            {1675281151884, 4126.500000, 20, 4127.000000, 12}, // qbd +1
            {1675281151893, 4126.500000, 18, 4127.000000, 12},
            {1675281151893, 4126.500000, 17, 4127.000000, 12},
            {1675281151893, 4126.500000, 18, 4127.000000, 12}  // qbd +1
        };
        const double tradea[6][3] = {
            {1675281150999, 4126.750000, 1}, // removed
            {1675281151884, 4126.750000, 1}, // sell
            {1675281151893, 4126.500000, 2}, // sell
            {1675281151893, 4126.500000, 10}, // sell
            {1675281153893, 4126.500000, 1}, // sell, after last quote
            {1675281155893, 4126.500000, 2} // removed after last utc
        };
        std::string out_csv = quote_file+".csv";
        EXPECT_TRUE(tp.parseDoubleArray((double*)quotea, (double*)tradea, 8, 6, out_csv));
        const auto& lines (utils::CSVUtil::read_file(out_csv));
        for (size_t i=0; i<lines.size(); ++i) {
            EXPECT_TRUE(line_compare(lines[i], bars[i]));
        }
        std::remove(out_csv.c_str());

        // write to one bars, with a 7-second bar, all included in one bar
        utc_start-=2;
        utc_end+=2;
        md::TickData2Bar tp1(quote_file, trade_file, utc_start, utc_end, utc_end-utc_start);
        bars.clear();
        EXPECT_TRUE(tp1.parse(bar_sp));
        EXPECT_EQ(bars.size(), 1);
        // dump ticks
        auto book_ticks = std::make_shared<std::vector<std::string>>();
        tp1.tickDump(book_ticks);

        md::BarPrice bp;
        bp.set_write_optional(true);
        for (const auto& tick: *book_ticks) {
            const auto& line (utils::CSVUtil::read_line(tick));
            bp.update<1>(line);
        }

        // bars[0] with opt_v1 being -2
        EXPECT_EQ(bp.writeAndRoll(utc_end), "1675281156, 4126.75, 4126.875, 4126.5, 4126.5, 17, 4126.5, 1675281155893000, -17, 13.9, 11.7, 0.459055, 3, 1, -1, -9");
    } catch (const std::exception& e) {
    }
    std::remove(quote_file.c_str()); 
    std::remove(trade_file.c_str());

    // try a longer quote/trade with partial matches
    // Note1: the trade of "578" is not matched, but
    //        should be inferred to the first reducing quote
    // Note2: multiple trades swipe levels
    const std::vector<std::string> trade1 = {
            "1675281151884, 4126.750000, 1",
            "1675281151893, 4126.500000, 2",
            "1675281151893, 4126.500000, 1",
            "1675281151897, 4127.000000, 14",
            "1675281151897, 4127.250000, 13", // these two swipe 2 levels
            "1675281151898, 4127.250000, 1",
            "1675281151898, 4127.250000, 1",
            "1675281151898, 4127.250000, 11",
            "1675281151898, 4127.250000, 1",
            "1675281151898, 4127.250000, 12",
            "1675281151898, 4127.250000, 2",
            "1675281151898, 4127.250000, 11",
            "1675281151898, 4127.250000, 6",
            "1675281151898, 4127.250000, 32",
            "1675281151898, 4127.250000, 2",
            "1675281151898, 4127.250000, 2",
            "1675281151898, 4127.250000, 1",
            "1675281151898, 4127.250000, 578", //a native iceberg order
            "1675281151898, 4127.250000, 1",
            "1675281151898, 4127.250000, 7",
            "1675281151899, 4127.250000, 2",
            "1675281151899, 4127.250000, 2",
            "1675281151899, 4127.250000, 12",
            "1675281151899, 4127.250000, 1",
            "1675281151899, 4127.250000, 10",
            "1675281151899, 4127.250000, 1",
            "1675281151899, 4127.250000, 12",
            "1675281151899, 4127.250000, 12",
            "1675281151899, 4127.250000, 1",
            "1675281151899, 4127.250000, 1",
            "1675281151899, 4127.250000, 10",
            "1675281151899, 4127.250000, 1",
            "1675281151899, 4127.250000, 1",
            "1675281151899, 4127.250000, 4",
            "1675281151899, 4127.250000, 1",
            "1675281151899, 4127.500000, 2",
            "1675281151899, 4127.500000, 1",
            "1675281151899, 4127.500000, 1",
            "1675281151899, 4127.500000, 1",
            "1675281151899, 4127.500000, 18",
            "1675281151899, 4127.500000, 2",
            "1675281151901, 4127.500000, 9",
            "1675281151902, 4127.500000, 1",
            "1675281151902, 4127.750000, 1",
            "1675281151902, 4127.500000, 3",
    };
    const std::vector<std::string> quote1 = {
            "1675281151882, 4126.750000, 1, 4127.000000, 11",
            "1675281151884, 4126.500000, 19, 4127.000000, 11",
            "1675281151884, 4126.500000, 19, 4127.000000, 12",
            "1675281151884, 4126.500000, 20, 4127.000000, 12",
            "1675281151893, 4126.500000, 18, 4127.000000, 12",
            "1675281151893, 4126.500000, 17, 4127.000000, 12",
            "1675281151893, 4126.500000, 17, 4127.000000, 14",
            "1675281151893, 4126.500000, 18, 4127.000000, 14",
            "1675281151897, 4127.250000, 12, 4127.500000, 38",
            "1675281151898, 4127.250000, 11, 4127.500000, 38",
            "1675281151898, 4127.250000, 11, 4127.500000, 39",
            "1675281151898, 4127.250000, 11, 4127.500000, 38",
            "1675281151898, 4127.250000, 10, 4127.500000, 38",
            "1675281151898, 4127.250000, 12, 4127.500000, 38",
            "1675281151898, 4127.250000, 12, 4127.500000, 37",
            "1675281151898, 4127.250000, 11, 4127.500000, 37",
            "1675281151898, 4127.250000, 14, 4127.500000, 37",
            "1675281151898, 4127.250000, 12, 4127.500000, 37",
            "1675281151898, 4127.250000, 14, 4127.500000, 37",
            "1675281151898, 4127.250000, 8, 4127.500000, 37",
            "1675281151898, 4127.250000, 2, 4127.500000, 37",
            "1675281151898, 4127.250000, 13, 4127.500000, 37",
            "1675281151898, 4127.250000, 11, 4127.500000, 37",
            "1675281151898, 4127.250000, 10, 4127.500000, 37",
            "1675281151898, 4127.250000, 4, 4127.500000, 37",
            "1675281151898, 4127.250000, 3, 4127.500000, 37",
            "1675281151898, 4127.250000, 9, 4127.500000, 37",
            "1675281151898, 4127.250000, 9, 4127.500000, 36",
            "1675281151898, 4127.250000, 10, 4127.500000, 36",
            "1675281151899, 4127.250000, 8, 4127.500000, 36",
            "1675281151899, 4127.250000, 6, 4127.500000, 36",
            "1675281151899, 4127.250000, 7, 4127.500000, 36",
            "1675281151899, 4127.250000, 6, 4127.500000, 36",
            "1675281151899, 4127.250000, 9, 4127.500000, 36",
            "1675281151899, 4127.250000, 9, 4127.500000, 37",
            "1675281151899, 4127.250000, 8, 4127.500000, 37",
            "1675281151899, 4127.250000, 10, 4127.500000, 37",
            "1675281151899, 4127.250000, 10, 4127.500000, 36",
            "1675281151899, 4127.250000, 9, 4127.500000, 36",
            "1675281151899, 4127.250000, 9, 4127.500000, 37",
            "1675281151899, 4127.250000, 8, 4127.500000, 37",
            "1675281151899, 4127.250000, 11, 4127.500000, 37",
            "1675281151899, 4127.250000, 10, 4127.500000, 37",
            "1675281151899, 4127.250000, 9, 4127.500000, 37",
            "1675281151899, 4127.250000, 9, 4127.500000, 38",
            "1675281151899, 4127.250000, 10, 4127.500000, 38",
            "1675281151899, 4127.250000, 11, 4127.500000, 38",
            "1675281151899, 4127.250000, 11, 4127.500000, 37",
            "1675281151899, 4127.250000, 11, 4127.500000, 36",
            "1675281151899, 4127.250000, 11, 4127.500000, 35",
            "1675281151899, 4127.250000, 11, 4127.500000, 36",
            "1675281151899, 4127.250000, 7, 4127.500000, 36",
            "1675281151899, 4127.250000, 6, 4127.500000, 36",
            "1675281151899, 4127.250000, 8, 4127.500000, 36",
            "1675281151899, 4127.250000, 8, 4127.500000, 34",
            "1675281151899, 4127.250000, 8, 4127.500000, 33",
            "1675281151899, 4127.250000, 8, 4127.500000, 32",
            "1675281151899, 4127.250000, 8, 4127.500000, 31",
            "1675281151899, 4127.250000, 9, 4127.500000, 31",
            "1675281151899, 4127.250000, 10, 4127.500000, 31",
            "1675281151899, 4127.250000, 10, 4127.500000, 13",
            "1675281151899, 4127.250000, 10, 4127.500000, 11",
            "1675281151899, 4127.250000, 12, 4127.500000, 11",
            "1675281151901, 4127.250000, 12, 4127.750000, 36",
            "1675281151901, 4127.250000, 13, 4127.750000, 36",
            "1675281151901, 4127.250000, 12, 4127.750000, 36",
            "1675281151901, 4127.500000, 4, 4127.750000, 36",
            "1675281151902, 4127.500000, 3, 4127.750000, 36",
            "1675281151902, 4127.500000, 3, 4127.750000, 35",
            "1675281151902, 4127.250000, 10, 4127.750000, 35",
            "1675281151902, 4127.250000, 11, 4127.750000, 35",
            "1675281151902, 4127.250000, 11, 4128.000000, 37",
            "1675281151902, 4127.250000, 13, 4128.000000, 37",
            "1675281151902, 4127.500000, 1, 4128.000000, 37",
            "1675281151903, 4127.250000, 14, 4128.000000, 37",
    };

    utc_start = 1675281150;
    utc_end =   1675281152;
    quote_file = write_file(quote1);
    trade_file = write_file(trade1);
    try {
        md::TickData2Bar tp(quote_file, trade_file, utc_start, utc_end, 1);
        auto bar_sp = std::make_shared<std::vector<std::string>>();
        std::vector<std::string>& bars(*bar_sp);
        EXPECT_TRUE(tp.parse(bar_sp));
        EXPECT_EQ(bars.size(), 2);
        EXPECT_EQ(bars[1], "1675281152, 4126.875, 4127.75, 4126.5, 4127.625, 809, 4127.5, 1675281151902000, -685, 14.1, 33.4, 0.690678, 63, -36, 13, -697");

        

        /*for tracing
        std::vector<std::string> book_ticks;
        tp.tickDump(book_ticks);
        write_file(book_ticks);
        */

    } catch(const std::exception& e) {
    }
    std::remove(quote_file.c_str()); 
    std::remove(trade_file.c_str());
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setupCfg();
    return RUN_ALL_TESTS();
}
