#include "md_bar.h"
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
        ofs << "    TESTQ1 = {\n";
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
        ofs << "        bbg_id = TESTQ1 COMDTY\n";
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

class BQFixture : public testing::Test {
public:
    BQFixture ():
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

    template<typename TupleType1, typename TupleType2>
    bool tuple_compare(const TupleType1& t0, const TupleType2& t1) const {
        const auto [x1, x2, x3, x4]=t0;
        const auto [y1, y2, y3, y4]=t1;
        return double_eq(x1,y1) && double_eq(x2,y2) &&
               double_eq(x3,y3) && double_eq(x4,y4);
    }

    template<typename PairType, typename DT1, typename DT2>
    bool pair_compare(const PairType& t0, const DT1& y1, const DT2& y2) const {
        const auto [x1, x2]=t0;
        return double_eq(x1,y1) && double_eq(x2,y2);
    }

    template<typename BookType, typename TupleType>
    bool checkBook(const BookType& bk, bool is_trade, const TupleType& tpl, uint64_t upd_micro) const {
        if (bk.update_ts_micro != upd_micro) return false;
        if (is_trade) {
            return tuple_compare(bk.getTradeTuple(), tpl);
        }
        return tuple_compare(bk.getBBOTuple(), tpl);
    }

protected: 
    md::BarPrice _bar;
    md::BookConfig _bcfg;
    int _barsec;
};

TEST_F (BQFixture, BidReducing) {
    std::string line = "1675198786000000, 0, 0, 101.2, 3, 0, 1, 101.2, 3, 1, 101.3, 2";
    md::BookDepotLevel<1> book(line);
    {
        // BID
        // check on bid/ask size reducing
        auto pq = md::getBidReducing(book,101.2, 1);
        EXPECT_TRUE(pair_compare(pq, 101.2, 2));

        pq = md::getBidReducing(book,101.2, 3);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        pq = md::getBidReducing(book,101.2, 5);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        // level add/remove
        pq = md::getBidReducing(book,101.3, 1);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        pq = md::getBidReducing(book,101.1, 1);
        EXPECT_TRUE(pair_compare(pq, 101.2, 3));

        // check on the 0 px/sz
        pq = md::getBidReducing(book,0, 1);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        pq = md::getBidReducing(book,101.2, 0);
        EXPECT_TRUE(pair_compare(pq, 101.2, 3));
    }
    {
        // ASK
        // check on bid/ask size reducing
        auto pq = md::getAskReducing(book,101.3, 1);
        EXPECT_TRUE(pair_compare(pq, 101.3, 1));

        pq = md::getAskReducing(book,101.3, 3);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        pq = md::getAskReducing(book,101.3, 5);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        // level add/remove
        pq = md::getAskReducing(book,101.2, 1);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        pq = md::getAskReducing(book,101.4, 1);
        EXPECT_TRUE(pair_compare(pq, 101.3, 2));

        // check on the 0 px/sz
        pq = md::getAskReducing(book,0, 1);
        EXPECT_TRUE(pair_compare(pq, 0, 0));

        pq = md::getAskReducing(book,101.3, 0);
        EXPECT_TRUE(pair_compare(pq, 101.3, 2));
    }
    {
        //BBO
        md::BBOTuple bt = {101.2, 3, 101.3, 1};
        auto tpl = md::getBBOReducing(book,bt);
        EXPECT_TRUE(tuple_compare(tpl, md::BBOTuple(0,0,101.3,1)));

        bt = {101.1, 2, 101.4, 1};
        tpl = md::getBBOReducing(book,bt);
        EXPECT_TRUE(tuple_compare(tpl, md::BBOTuple(101.2,3,101.3,2)));

        bt = {102.2, 3, 101.3, 4};
        tpl = md::getBBOReducing(book,bt);
        EXPECT_TRUE(tuple_compare(tpl, md::BBOTuple(0,0,0,0)));

        bt = {0, 0, 101.4, 4};
        tpl = md::getBBOReducing(book,bt);
        EXPECT_TRUE(tuple_compare(tpl, md::BBOTuple(0,0,101.3,2)));

        bt = {100.2, 100, 0, 0};
        tpl = md::getBBOReducing(book,bt);
        EXPECT_TRUE(tuple_compare(tpl, md::BBOTuple(101.2,3,0,0)));
    }
}

TEST_F (BQFixture, TradeDirection) {
    {
        std::string line = "1675198786000000, 0, 0, 101.2, 3, 0, 1, 101.1, 11, 1, 101.2, 2";
        md::BookDepotLevel<1> book(line);
        md::BBOTuple bt = {101.1, 11, 101.2, 2};
        EXPECT_TRUE(tuple_compare(bt, book.getBBOTuple()));
    }

    md::BookDepotLevel<1> book;
    md::TradeDirection tdir(book, _bcfg);

    // initial invalid
    EXPECT_EQ(tdir.updBBO(101.2,1,false,1675198786001000),nullptr);
    book.updBBO(101.2,1,false,1675198786001000);

    // invalid quote
    auto trd_pair = tdir.updTrade(101.2,1,1675198786003000);
    EXPECT_EQ(trd_pair.first, nullptr);
    EXPECT_EQ(trd_pair.second, nullptr);
    EXPECT_FALSE(tdir.hasPending());

    md::BBOTuple bt = {101.1, 11, 101.3, 2};
    EXPECT_EQ(tdir.updBBO(bt, 1675198786002000),nullptr);
    // now has two sided quotes
    book.updBBO(bt, 1675198786002000);

    // should have normal trade
    EXPECT_EQ(tdir.updBBO(bt,1675198786002000),nullptr);
    trd_pair = tdir.updTrade(101.3,1,1675198786003000);
    EXPECT_EQ(trd_pair.first, nullptr);
    EXPECT_FALSE(trd_pair.second==nullptr);

    // identify as a buy at 101.3
    md::TradeTuple tt = {101.3, 1, 0, 1675198786003000};
    EXPECT_TRUE(tuple_compare(*trd_pair.second, tt));

    trd_pair=tdir.updTrade(101.0,2,1675198786004000);
    EXPECT_EQ(trd_pair.first, nullptr);
    EXPECT_FALSE(trd_pair.second==nullptr);
    tt = {101.0, 2, 1, 1675198786004000};
    EXPECT_TRUE(tuple_compare(*trd_pair.second, tt));
    EXPECT_FALSE(tdir.hasPending());

    // push in mid
    trd_pair=tdir.updTrade(101.2,1,1675198786005000);
    EXPECT_EQ(trd_pair.first, nullptr);
    EXPECT_EQ(trd_pair.second, nullptr);
    EXPECT_TRUE(tdir.hasPending());

    // add one more
    trd_pair=tdir.updTrade(101.2,3,1675198786006000);
    EXPECT_EQ(trd_pair.first, nullptr);
    EXPECT_EQ(trd_pair.second, nullptr);
    EXPECT_TRUE(tdir.hasPending());

    // pending decided by the good trade
    trd_pair=tdir.updTrade(101.1,5,1675198786007000);
    EXPECT_TRUE(trd_pair.first!=nullptr);
    tt = {101.2, 4, 1, 1675198786006000};
    EXPECT_TRUE(tuple_compare(*trd_pair.first, tt));
    EXPECT_TRUE(trd_pair.second!=nullptr);
    tt = {101.1, 5, 1, 1675198786007000};
    EXPECT_TRUE(tuple_compare(*trd_pair.second, tt));
    EXPECT_FALSE(tdir.hasPending());

    tdir.updTrade(101.2,1,1675198786008000);
    EXPECT_TRUE(tdir.hasPending());
    trd_pair=tdir.updTrade(101.3,5,1675198786009000);
    EXPECT_TRUE(trd_pair.first!=nullptr);
    tt = {101.2, 1, 0, 1675198786008000};
    EXPECT_TRUE(tuple_compare(*trd_pair.first, tt));
    EXPECT_TRUE(trd_pair.second!=nullptr);
    tt = {101.3, 5, 0, 1675198786009000};
    EXPECT_TRUE(tuple_compare(*trd_pair.second, tt));
    EXPECT_FALSE(tdir.hasPending());

    // pending discarded by late trade
    tdir.updTrade(101.2,1,1675198786008000);
    EXPECT_TRUE(tdir.hasPending());
    trd_pair=tdir.updTrade(101.3,5,1675198786121000);
    EXPECT_EQ(trd_pair.first, nullptr);
    EXPECT_TRUE(trd_pair.second!=nullptr);
    tt = {101.3, 5, 0, 1675198786121000};
    EXPECT_TRUE(tuple_compare(*trd_pair.second, tt));
    EXPECT_FALSE(tdir.hasPending());

    // pending decided by good quote
    md::BBOTuple bt0={101.1, 11, 101.3, 2};
    tdir.updBBO(bt0, 1675198786012000);
    tdir.updTrade(101.2,1,1675198786012000);
    EXPECT_TRUE(tdir.hasPending());
       // ref: bt = {101.1, 11, 101.3, 2};
    bt = {101.1, 10, 101.3, 2}; // bid reducing
    auto trd0=tdir.updBBO(bt, 1675198786013000);
    tt = {101.2, 1, 1, 1675198786012000};
    EXPECT_TRUE(trd0!=nullptr);
    EXPECT_TRUE(tuple_compare(*trd0, tt));
    EXPECT_FALSE(tdir.hasPending());

    tdir.updBBO(bt0, 1675198786014000);
    tdir.updTrade(101.2,1,1675198786014000);
    EXPECT_TRUE(tdir.hasPending());
       // ref: bt = {101.1, 11, 101.3, 2};
    trd0=tdir.updBBO(101.4, 1, false, 1675198786015000);
    tt = {101.2, 1, 0, 1675198786014000};
    EXPECT_TRUE(trd0!=nullptr);
    EXPECT_TRUE(tuple_compare(*trd0, tt));
    EXPECT_FALSE(tdir.hasPending());


    // pending discarded by non-reducing quote
    tdir.updBBO(bt0, 1675198786016000);
    tdir.updTrade(101.2,1,1675198786016000);
    EXPECT_TRUE(tdir.hasPending());
       // ref: bt = {101.1, 11, 101.3, 2};
    trd0=tdir.updBBO(101.2, 1, true, 1675198786017000);
    EXPECT_TRUE(trd0==nullptr);
    EXPECT_TRUE(tdir.hasPending());
       // another side is not decreasing
    trd0=tdir.updBBO(101.3, 3, false, 1675198786017000);
    EXPECT_TRUE(trd0==nullptr);
    EXPECT_FALSE(tdir.hasPending());


    // pending discarded by 2-sided decreasing quote
    tdir.updBBO(bt0, 1675198786017001);
    tdir.updTrade(101.2,1,1675198786017001);
    EXPECT_TRUE(tdir.hasPending());
       // ref: bt = {101.1, 11, 101.3, 2};
    trd0=tdir.updBBO(101.2, 1, false, 1675198786017900);
    EXPECT_TRUE(trd0==nullptr);
    EXPECT_TRUE(tdir.hasPending());
       // giving one more chance
    bt = {101.1, 1, 101.2, 1}; // bid reducing
    trd0=tdir.updBBO(bt, 1675198786017991);
    tt = {101.2, 1, 1, 1675198786017001};  // sell trade
    EXPECT_TRUE(tuple_compare(*trd0, tt));
    EXPECT_FALSE(tdir.hasPending());


    tdir.updBBO(bt0, 1675198786018000);
    tdir.updTrade(101.2,1,1675198786018000);
    EXPECT_TRUE(tdir.hasPending());
       // ref: bt = {101.1, 11, 101.3, 2};
    bt = {101.0, 10, 101.4, 20}; // both reducing
    trd0=tdir.updBBO(bt, 1675198786019000);
    EXPECT_TRUE(trd0==nullptr);
    EXPECT_FALSE(tdir.hasPending());

    // pending discarded by late quotes
    tdir.updBBO(bt0, 1675198786020000);
    tdir.updTrade(101.2,1,1675198786020000);
    EXPECT_TRUE(tdir.hasPending());
       // ref: bt = {101.1, 11, 101.3, 2};
    trd0=tdir.updBBO(101.1, 1, true, 1675198786123000);
    EXPECT_TRUE(trd0==nullptr);
    EXPECT_FALSE(tdir.hasPending());
}

TEST_F (BQFixture, TradeDirection_GivenDir) {

    // 1. given dir
    // 2. given dir upon pending
    //    px +/-/=
    //    bbo
    // 3. given undecided dir
    // 4. given undecided dir upon pending
    //    px +/-/=
    //    bbo
    // 5. mixed given and inferred
    //    pending due to 
    
    {
        std::string line = "1675198786000000, 0, 0, 101.2, 3, 0, 1, 101.1, 11, 1, 101.2, 2";
        md::BookDepotLevel<1> book(line);
        md::BBOTuple bt = {101.1, 11, 101.2, 2};
        EXPECT_TRUE(tuple_compare(bt, book.getBBOTuple()));
    }

    md::BookDepotLevel<1> book;
    md::TradeDirection tdir(book,_bcfg);

    // TODO - to be finished here
}

TEST_F (BQFixture, BookQ) {

    md::BookQ<1> bq (_bcfg, false, true);
    auto bout = bq.theWriter(); // it's a reference
    auto br = bq.newReader();

    md::BookWriter<md::BookQ<1>::Writer, 1> bw(bout, _bcfg);

    md::BBOTuple bt0={101.1, 11, 101.3, 2};
    bw.updBBO(bt0, 1675198786000000); // 1. {101.1, 11, 101.3, 2}
    bw.updBBO(101.3,4,false,1675198786001000); // 2. ask to 101.3, 4
    bw.updTrade(101.1, 1, 1675198786002000); // 3. trade sell 1 @ 101.1
    bw.updBBO(101.1, 10, true, 1675198786002000); // 4. bid adj to 101.1,10
    bw.updTrade(101.2,1, 1675198787000000); // undecided 1 at 101.2 (middle)
    bw.updTrade(101.2,3, 1675198787001000); // add 3  more

    // decided by trade
    bw.updTrade(101.1,2, 1675198787002000); // 5. trade sell 4 @ 101.2, 
                                            // 6. sell 2 @ 101.1

    bw.updTrade(101.2,1, 1675198788000000); // undecided 1 at 101.2 (middle)
    bw.updTrade(101.4,1, 1675198788002000); // 7. trade buy 1 @ 101.2, 
                                            // 8. buy 1 @ 101.4

    // decided by quote
    bw.updTrade(101.2,1, 1675198789000000); // undecided 1 at 101.2 (middle)
    bw.updBBO(101.1, 9, true, 1675198789002000); // 9. trade sell 1 at 101.2, 
                                                 // 10. bid adj to 101.1,9

    bw.updTrade(101.2,1, 1675198791001000); // undecided 1 at 101.2 (middle)
    bt0={101.1, 11, 101.4, 2};
    bw.updBBO(bt0, 1675198791002000); // 11. trade buy 1 at 101.2, 
                                      // 12. bbo adj bid to 101.1, 11
                                      // 13. bbo adj ask to 101.4, 2

    // discarded by quote (non-decreasing)
    bw.updBBO(101.3,10,false, 1675198792000000); // 14. ask adj to 101.3, 10
    bw.updTrade(101.2,1,      1675198792000000); // undecided 1 at 101.2 (middle)
    bw.updBBO(101.1, 9, true, 1675198792103000); // discarded, too late
                                                 // more than 2-milli
                                                 // 14. bid adj to 101.1, 9

    // now recover everything from the reader's queue
    md::BookDepotLevel<1> book;
    std::vector<md::BookDepotLevel<1>> bv;
    while (br->getNextUpdate(book) ) {
        bv.push_back(book);
    }

    EXPECT_EQ(bv.size(), 15);
    md::TradeTuple tt0;

    bt0={101.1, 11, 101.3, 2};
    EXPECT_TRUE(checkBook(bv[0], false, bt0, 1675198786000000));
    bt0={101.1, 11, 101.3, 4};
    EXPECT_TRUE(checkBook(bv[1], false, bt0, 1675198786001000));
    tt0={101.1, 1, 1, 1675198786002000};
    EXPECT_TRUE(checkBook(bv[2], true, tt0, 1675198786002000));
    bt0={101.1, 10, 101.3, 4};
    EXPECT_TRUE(checkBook(bv[3], false, bt0, 1675198786002000));

    tt0={101.2, 4, 1, 1675198787001000};
    EXPECT_TRUE(checkBook(bv[4], true, tt0, 1675198787001000));
    tt0={101.1, 2, 1, 1675198787002000};
    EXPECT_TRUE(checkBook(bv[5], true, tt0, 1675198787002000));

    tt0={101.2, 1, 0, 1675198788000000};
    EXPECT_TRUE(checkBook(bv[6], true, tt0, 1675198788000000));
    tt0={101.4, 1, 0, 1675198788002000};
    EXPECT_TRUE(checkBook(bv[7], true, tt0, 1675198788002000));

    tt0={101.2, 1, 1, 1675198789000000};
    EXPECT_TRUE(checkBook(bv[8], true, tt0,  1675198789000000));
    bt0={101.1, 9, 101.3, 4};
    EXPECT_TRUE(checkBook(bv[9], false, bt0, 1675198789002000));

    tt0={101.2, 1, 0, 1675198791001000};
    EXPECT_TRUE(checkBook(bv[10], true, tt0,  1675198791001000));

    bt0={101.1, 11, 101.4, 2};
    EXPECT_TRUE(checkBook(bv[12], false, bt0, 1675198791002000));

    bt0={101.1, 11, 101.3, 10};
    EXPECT_TRUE(checkBook(bv[13], false, bt0, 1675198792000000));
    bt0={101.1, 9, 101.3, 10};
    EXPECT_TRUE(checkBook(bv[14], false, bt0, 1675198792103000));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setupCfg();
    return RUN_ALL_TESTS();
}
