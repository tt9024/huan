#include "RiskMonitor.h"
#include "stdio.h"
#include <string>
#include "gtest/gtest.h"
#include <cstdlib>
#include <fstream>
#include "PositionData.h"
#include "ExecutionReport.h"

class MockPM {
public:
    MockPM() :
        algo_qty(0), // algo+mkt position
        eng_qty(0),  // aggregated position of mkt for all algo
        algo_pnl(0), // aggregated pnl of algo for all mkt 
        mkt_pnl(0) {};  // algo+mkt pnl

    long long getPosition_Market(const std::string* algo,
                                 const std::string* mkt,
                                 double* pnl=nullptr) const {
        if (algo && mkt) {
            if (pnl) *pnl = mkt_pnl;
            return algo_qty;
        }

        if (!mkt) {
            if (pnl) *pnl = algo_pnl;
            return (long long) 1e+10;
        }

        //!algo
        return eng_qty;
    }

    template<typename ER>
    bool haveThisFill(const ER& er) const {
        return false;
    }

    void setPosition_Market(int64_t algo_qty_, int64_t eng_qty_, double algo_pnl_, double mkt_pnl_) {
        algo_qty = algo_qty_;
        eng_qty = eng_qty_;
        algo_pnl = algo_pnl_;
        mkt_pnl = mkt_pnl_;
    }

    /* need to support such operations
    algo_qty = pm.getPosition_Market(&algo, &mkt, &mkt_pnl);
    pm.getPosition_Market(&algo, nullptr, &algo_pnl);
    eng_qty =  pm.getPosition_Market(nullptr, &mkt);
    */
    int64_t algo_qty, eng_qty;
    double algo_pnl, mkt_pnl;
};

class MockFM {
public:
    bool isFloorManager() const { return true; } ;
    std::string toString() const { return "MockFM"; };
};

class RiskFixture : public testing::Test {
public:
    RiskFixture ():
    _sym_cfg_str ( \
        "tradable = {\n"
        "    CLG0 = {\n"
        "        symbol = WTI\n"
        "        exch_symbol = CL\n"
        "        venue = NYM\n"
        "        mts_contract = CL_202002\n"
        "        tick_size = 0.010000000000\n"
        "        point_value = 1000.0\n"
        "        px_multiplier = 0.010000000000\n"
        "        type = FUT\n"
        "        contract_month = 202002\n"
        "        tt_security_id = 3971113080253836646\n"
        "        tt_venue = NYM\n"
        "        bbg_id = CLM1 CMD\n"
        "        bbg_px_multiplier = 1.0\n"
        "        currency = USD\n"
        "        expiration_date = 2020-02-20\n"
        "        mts_symbol = WTI_N0\n"
        "        N = 0\n"
        "        lotspermin = 40\n"
        "        expiration_days = 20\n"
        "    }\n"
        "    CLH0 = {\n"
        "        symbol = WTI\n"
        "        exch_symbol = CL\n"
        "        venue = NYM\n"
        "        mts_contract = CL_202003\n"
        "        tick_size = 0.010000000000\n"
        "        point_value = 1000.0\n"
        "        px_multiplier = 0.010000000000\n"
        "        type = FUT\n"
        "        contract_month = 202003\n"
        "        tt_security_id = 3971113080253836647\n"
        "        tt_venue = NYM\n"
        "        bbg_id = CLK1 CMD\n"
        "        bbg_px_multiplier = 1.0\n"
        "        currency = USD\n"
        "        expiration_date = 2020-03-20\n"
        "        mts_symbol = WTI_N1\n"
        "        N = 1\n"
        "        lotspermin = 40\n"
        "        expiration_days = 20\n"

        "    }\n"
        "}"),
    _risk_fn("tests/risk.json"),
    _sym_fn("/tmp/risk_test_sym.cfg"),
    _main_cfg_fn("/tmp/main.cfg")
    {
        writeCfg();
        {
            // write the main config to /tmp/main.cfg
            std::ofstream ofs;
            ofs.open (_main_cfg_fn, std::ofstream::out | std::ofstream::trunc);
            ofs << "SymbolMap = " << _sym_fn << std::endl;
            ofs << "Logger = /tmp/risk_test_log" << std::endl;
            ofs << "MDProviders = { BPipe = /tmp/bpipe.cfg }\nMaxN = 2\nMTSVenue = { NYM = [BPipe] }\nMTSSymbol = {}" << std::endl;
        }

        // set the config path NOW!
        utils::PLCC::setConfigPath(_main_cfg_fn);
    }

    void writeCfg() const {
        FILE* fp = fopen(_sym_fn, "w");
        fprintf(fp, "%s\n", _sym_cfg_str);
        fclose(fp);
    }

    pm::ExecutionReport genER(const std::string& tag39, int qty, time_t utc, double px=1.0) const {
        return pm::ExecutionReport("CLH0", "TSC-7000-387", "", "", tag39, qty, px, "", "", utc*1000000ULL, 0);
    }

    const char *_sym_cfg_str, *_risk_fn, *_sym_fn, *_main_cfg_fn;
};

TEST_F (RiskFixture, cfg_test) {
    // load the risk and check for the numbers
    auto cfg = pm::risk::Config(std::string(_risk_fn));

    // scale
    EXPECT_DOUBLE_EQ(cfg.m_scale["TSC-7000-370"], 3.15);
    EXPECT_DOUBLE_EQ(cfg.m_scale["TSC-7000-389"], 3.99);

    const std::string mkt("WTI");

    // engine max_pos
    EXPECT_EQ(cfg.m_eng_max_pos[mkt], 1000);

    // count limit
    const auto& eng_cnt (cfg.m_eng_count[mkt]);
    EXPECT_EQ(eng_cnt.size(), 2);
    EXPECT_EQ(eng_cnt[1].first, 120);
    EXPECT_EQ(eng_cnt[1].second, 300);

    // m_eng_fat_finger
    EXPECT_EQ(cfg.m_eng_fat_finger[mkt], 100);

    // participation rates
    const auto& part_rate(cfg.m_eng_rate_limit[mkt]);
    EXPECT_EQ(part_rate.size(), 276);
    const auto& pr1 (part_rate[1]); // vector< <rate, seconds> >
    // 376.2482205324331, 584.4073690238184, 1835.117347729162, 1852.4857854276174, 6105.742482473943, 4470.49331814198, 0.01864343711618308, 0.008110143218291293

    EXPECT_DOUBLE_EQ(pr1.size(), 3);
    EXPECT_DOUBLE_EQ(pr1[0].first, 1.0*(376.2482205324331 + 3.0 * 584.4073690238184));
    EXPECT_EQ(pr1[0].second, 5*60);
    EXPECT_DOUBLE_EQ(pr1[2].first, 0.25*6105.742482473943);
    EXPECT_EQ(pr1[2].second, 60*60);

    const auto& spd(cfg.m_eng_spread_limit[mkt]);
    EXPECT_EQ(spd.size(), 276);
    EXPECT_DOUBLE_EQ(spd[1], 0.01864343711618308 + 5*_MAX_(0.008110143218291293, 0.01864343711618308));

    // strategyr
    const std::string strat ("TSC-7000-387");
    double scl (5.51);

    EXPECT_FALSE(cfg.m_strat_paper_trading[strat]);
    EXPECT_DOUBLE_EQ(cfg.m_strat_pnl_drawdown[strat], -5000000*scl);
    EXPECT_DOUBLE_EQ(cfg.m_strat_mkt_pnl_drawdown[strat][mkt], -2000000*scl);
    EXPECT_EQ((int)cfg.m_strat_max_pos[strat][mkt], (int)(100 * scl + 0.5));

    const auto& strat_cnt (cfg.m_strat_count[strat][mkt]);
    EXPECT_EQ(strat_cnt.size(), 2);
    EXPECT_EQ(strat_cnt[0].first, 10);
    EXPECT_EQ(strat_cnt[1].second, 300);

    const auto& strat_rate (cfg.m_strat_rate_limit[strat][mkt]);
    EXPECT_EQ(strat_rate.size(), 2);
    EXPECT_EQ(strat_rate[0].second, 15 * 60);
    EXPECT_DOUBLE_EQ(strat_rate[1].first, 4.0);

    const auto& strat_flip (cfg.m_strat_flip[strat][mkt]);
    EXPECT_EQ(strat_flip.size(), 1);
    EXPECT_EQ(strat_flip[0].first, 2);
    EXPECT_EQ(strat_flip[0].second, 23*3600);
}

TEST_F (RiskFixture, status_test) {
    // load the risk and check for the numbers
    auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn)));
    auto ts_csv (cfg->m_status_file);
    FILE* fp = fopen(ts_csv.c_str(), "w");
    fprintf(fp, "20221223-09:00:00, RiskMonitor, Pause, ALL, ALL, OFF\n");
    fclose(fp);

    try {
        // should throw since no operator id is set
        auto st = pm::risk::Status(cfg);
        EXPECT_TRUE(false);
    }catch (const std::exception& e) {
    }

    fp = fopen(ts_csv.c_str(), "at");
    fprintf(fp, "20221223-09:00:00, RiskMonitor, Operator, OP1\n");
    fclose(fp);
    auto st = pm::risk::Status(cfg);
    const std::string mkt("WTI");
    const std::string strat ("TSC-7000-387");
    MockFM fm;

    // trading_status.csv has initial content of 
    //      20221223-09:00:00, RiskMonitor, ALL, ALL, OFF 
    EXPECT_FALSE(st.getPause(strat, mkt));
    EXPECT_EQ(st.get_operator_id(), std::string("OP1"));
    //printf("queryPause: %s\n", st.queryPause("", "").c_str());

    st.setPause("", "", true);
    st.persist_pause(fm, "Z ,,ON");
    st.set_operator_id(fm, "OP2");
    EXPECT_TRUE(st.getPause(strat, mkt));
    EXPECT_EQ(st.get_operator_id(), std::string("OP2"));
    //printf("queryPause: %s\n", st.queryPause("", "").c_str());

    {
        auto st2 = pm::risk::Status(cfg);
        EXPECT_TRUE(st.getPause(strat, mkt));
        EXPECT_EQ(st.get_operator_id(), std::string("OP2"));
    }
    st.setPause("", "", false);
    st.persist_pause(fm, "Z ,,OFF");
    {
        auto st2 = pm::risk::Status(cfg);
        EXPECT_FALSE(st.getPause(strat, mkt));
    }
    st.setPause("TSC-7000-387", "", true);
    EXPECT_TRUE(st.getPause(strat, mkt));

    st.setPause("", "", false);
    EXPECT_FALSE(st.getPause(strat, mkt));
    st.setPause("", "WTI", true);
    EXPECT_TRUE(st.getPause(strat, mkt));

    st.setPause("", "", false);
    EXPECT_FALSE(st.getPause(strat, mkt));
    st.setPause(strat, mkt, true);
    EXPECT_TRUE(st.getPause(strat, mkt));

    st.persist_pause(fm, "Z TSC-7000-387,WTI,ON");
    {
        auto st2 = pm::risk::Status(cfg);
        EXPECT_TRUE(st.getPause(strat, mkt));
        EXPECT_EQ(st.get_operator_id(), std::string("OP2"));
    }
}

TEST_F(RiskFixture, state_test) {
    // want loggings
    utils::PLCC::ToggleTest(true);

    auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn)));
    const auto input_scale = cfg->m_scale;
    auto stt = pm::risk::State(cfg);
    const std::string mkt("WTI");
    const std::string strat ("TSC-7000-387");
    MockPM pm;
    int64_t algo_qty_, eng_qty_;
    double algo_pnl_, mkt_pnl_;

    // simple order update without PM
    const time_t utc0 = utils::TimeUtil::startUTC() + 5*60 + 1 + 24*3600;
    time_t cur_utc = utc0;
    // make sure it's at 6:05 pm
    
    // fat finger
    EXPECT_FALSE(stt.checkOrder(strat, mkt, 101, cur_utc));

    // good case
    EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc));
    EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc));
    EXPECT_TRUE(stt.reportFill(strat, mkt, 1, cur_utc));

    // order counts
    // eng:   [60 orders @60sec, 120 orders @300sec]
    // strat: [10 orders @60sec, 30 orders @300sec]
    {
        auto stt = pm::risk::State(cfg);
        time_t cur_utc = utc0;
        // hit the strat order count at 60sec
        for (int i=0; i<10; ++i) {
            EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc));
            EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc));
            cur_utc += 5;
        }
        EXPECT_FALSE(stt.checkOrder(strat, mkt, 1, cur_utc));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc));

        // cancel
        stt.reportCancel(strat, mkt); // this removes order counts on both sides
        EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc)); //because vilating order was updated
        stt.reportCancel(strat, mkt);
        stt.reportCancel(strat, mkt);
        EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc));
    };
    {
        auto stt = pm::risk::State(cfg);
        time_t cur_utc = utc0;
        // hit the strat order count at 60sec
        for (int i=0; i<30; ++i) {
            EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc));
            EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc));
            cur_utc += 8;
        }
        EXPECT_FALSE(stt.checkOrder(strat, mkt, 1, cur_utc));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc));

        // cancel
        stt.reportCancel(strat, mkt); // this removes order counts on both sides
        EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc));
        stt.reportCancel(strat, mkt);  // cancel twice
        stt.reportCancel(strat, mkt);
        EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc));
    }
    {
        // bring up the strat cnt to let the eng count to hit
        // hit 60@60
        auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn)));
        cfg->m_strat_count[strat][mkt][0].first=100;
        cfg->m_strat_count[strat][mkt][1].first=150;

        auto stt = pm::risk::State(cfg);
        time_t cur_utc = utc0;
        // hit the strat order count at 60sec
        int i=0;
        for (; i<60; ++i) {
            EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc+int(0.9*i)));
            EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc+int(0.9*i)));
        }
        EXPECT_FALSE(stt.checkOrder(strat, mkt, 1, cur_utc+int(0.9*i)));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc+int(0.9*i)));

        // cancel
        stt.reportCancel(strat, mkt); // this removes order counts on both sides
        EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc+int(0.9*i)));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc+int(0.9*i)));
        stt.reportCancel(strat, mkt);
        stt.reportCancel(strat, mkt);
        EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc+int(0.9*i)));
    }
    {
        // bring up the strat cnt to let the eng count to hit
        // hit 120@300
        auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn)));
        cfg->m_strat_count[strat][mkt][0].first=100;
        cfg->m_strat_count[strat][mkt][1].first=150;

        auto stt = pm::risk::State(cfg);
        time_t cur_utc = utc0;
        // hit the strat order count at 60sec
        int i=0;
        for (; i<120; ++i) {
            EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc+int(2*i)));
            EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc+int(2*i)));
        }
        EXPECT_FALSE(stt.checkOrder(strat, mkt, 1, cur_utc+int(2*i)));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc+int(2*i)));

        // cancel
        stt.reportCancel(strat, mkt); // this removes order counts on both sides
        EXPECT_TRUE(stt.checkOrder(strat, mkt, 1, cur_utc+int(2*i)));
        EXPECT_FALSE(stt.reportNew(strat, mkt, cur_utc+int(2*i)));
        stt.reportCancel(strat, mkt);
        stt.reportCancel(strat, mkt);
        EXPECT_TRUE(stt.reportNew(strat, mkt, cur_utc+int(2*i)));
    }

    // avg/std of volume at bar: 1
    // 376.2482205324331, 584.4073690238184, 1835.117347729162, 1852.4857854276174, 6105.742482473943, 4470.49331814198, 0.01864343711618308, 0.008110143218291293
    //
    // eng participation rate at [2129.47@5min, 1843.80@15m, 1526.43@60m]
    // strat turnover rate at [600@15m, 1200@60m]
    //
    // turnover rate: 
    // 2*100*2*1.5@15min, 4*100*2*1.5@60m
    
    {
        // hit strat turnover rate with scale 1.5
        auto scale = input_scale;
        scale[strat] = 1.5; // turnover rate will hit first - 600
        auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn), scale));
        auto stt = pm::risk::State(cfg);
        time_t cur_utc = utc0;
        // hit the turnover rate of 600
        for (int i=0; i<10; ++i) {
            EXPECT_TRUE(stt.checkOrder(strat, mkt, 59, cur_utc));
            EXPECT_TRUE(stt.reportFill(strat, mkt, 59, cur_utc));
            cur_utc += 10;
        }
        EXPECT_FALSE(stt.checkOrder(strat, mkt, 59, cur_utc));
        EXPECT_FALSE(stt.reportFill(strat, mkt, 59, cur_utc));
    }

    {
        // hit eng participation rate - 1526
        auto scale = input_scale;
        scale[strat] = 6.0; // eng part rate will hit first - 1526
        auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn), scale));
        auto stt = pm::risk::State(cfg);
        time_t cur_utc = utc0;
        // hit the turnover rate of 600
        for (int i=0; i<20; ++i) {
            EXPECT_TRUE(stt.checkOrder(strat, mkt, 75, cur_utc));
            EXPECT_TRUE(stt.reportFill(strat, mkt, 75, cur_utc));
            cur_utc += 12;
        }
        EXPECT_FALSE(stt.checkOrder(strat, mkt, 75, cur_utc));
        EXPECT_FALSE(stt.reportFill(strat, mkt, 75, cur_utc));
    }

    {
        // position checks, set scale to be 2
        auto scale = input_scale;
        scale[strat] = 2.0; // scaled to be 2 for maxpos/pnl
        auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn), scale));
        auto stt = pm::risk::State(cfg);

        // good case
        EXPECT_TRUE(stt.checkPosition(strat, mkt, 2, pm, false, 0));
        EXPECT_TRUE(stt.checkPosition(strat, mkt, 2, pm, true, 0));

        // eng_max = 1000, strat_max = 200
        algo_qty_= -192;
        eng_qty_ = -192;
        algo_pnl_ =0;
        mkt_pnl_ = 0;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -9, pm, false, 0));
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -9, pm, true, 0));

        // strat drawdown -5m*2, strat+mkt drawdown -2m*2
        algo_qty_= -192;
        eng_qty_ = -192;
        algo_pnl_ =0;
        mkt_pnl_ = -4000001;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -1, pm, false, 0));
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -1, pm, true, 0));

        algo_qty_= -92;
        eng_qty_ = -92;
        algo_pnl_ = -10000001;
        mkt_pnl_ =  0;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -1, pm, false, 0));
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -1, pm, true, 0));
    }

    {
        // the flip limit of 2 flips in 23 hours
        auto cfg (std::make_shared<pm::risk::Config>(std::string(_risk_fn)));
        auto stt = pm::risk::State(cfg);

        algo_qty_= 0;
        eng_qty_ = 0;
        algo_pnl_ = 0;
        mkt_pnl_ =  0;
        auto cur_utc = utils::TimeUtil::cur_utc();
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_TRUE(stt.checkPosition(strat, mkt, 50, pm, false, cur_utc));
        EXPECT_TRUE(stt.checkPosition(strat, mkt, 50, pm, true, cur_utc));

        algo_qty_= 50;
        eng_qty_ = 50;
        cur_utc+= 3600;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_TRUE(stt.checkPosition(strat, mkt, -100, pm, false, cur_utc));
        EXPECT_TRUE(stt.checkPosition(strat, mkt, -100, pm, true,  cur_utc));

        algo_qty_= -50;
        eng_qty_ = -50;
        cur_utc += 3600;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_TRUE(stt.checkPosition(strat, mkt, 100, pm, false, cur_utc));
        EXPECT_TRUE(stt.checkPosition(strat, mkt, 100, pm, true,  cur_utc));

        algo_qty_= 50;
        eng_qty_ = 50;
        cur_utc += 3600*10;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -100, pm, false, cur_utc));
        EXPECT_FALSE(stt.checkPosition(strat, mkt, -100, pm, true, cur_utc));
    }
}

TEST_F(RiskFixture, monitor_test) {
    // want loggings
    utils::PLCC::ToggleTest(true);
    const std::string mkt("WTI");
    const std::string sym("CLH0");

    const std::string strat ("TSC-7000-387");
    auto& mon (*pm::risk::Monitor::get(_risk_fn));
    MockPM pm;
    int64_t algo_qty_, eng_qty_;
    double algo_pnl_, mkt_pnl_;

    // simple order update without PM
    time_t cur_utc = utils::TimeUtil::startUTC() + 5*60 + 1 + 24*3600;
    // make sure it's at 6:05 pm
    
    mon.status().setPause("", "", false);

    // fat finger
    EXPECT_FALSE(mon.checkNewOrder(strat, sym, 101, pm, cur_utc));
    EXPECT_FALSE(mon.checkNewOrder(strat, sym, 101, cur_utc));

    // good case
    mon.status().setPause("", "", false);
    EXPECT_TRUE(mon.checkNewOrder(strat, sym, 1, pm, cur_utc));
    EXPECT_TRUE(mon.checkNewOrder(strat, sym, 1, cur_utc));
    EXPECT_TRUE(mon.checkReplace(strat, sym, -1, pm, cur_utc));
    EXPECT_TRUE(mon.checkReplace(strat, sym, -1));

    auto er(genER("0", 1, cur_utc));
    EXPECT_TRUE(mon.updateER(er));
    EXPECT_TRUE(mon.updateER(er, pm));

    {
        // the flip limit of 2 flips in 23 hours
        cur_utc += 3600;
        algo_qty_= 0;
        eng_qty_ = 0;
        algo_pnl_ = 0;
        mkt_pnl_ =  0;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_TRUE(mon.checkReplace(strat, sym, 50, pm, cur_utc));
        EXPECT_TRUE(mon.updateER(
                    genER("2", 50, cur_utc),
                    pm));

        cur_utc += 3600;
        algo_qty_= 50;
        eng_qty_ = 50;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_TRUE(mon.checkNewOrder(strat, sym, -100, pm, cur_utc));
        EXPECT_TRUE(mon.updateER(
                    genER("2", -100, cur_utc),
                    pm));

        cur_utc += 3600;
        algo_qty_= -50;
        eng_qty_ = -50;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_TRUE(mon.checkReplace(strat, sym, 100, pm, cur_utc));
        EXPECT_TRUE(mon.updateER(
                    genER("2", 100, cur_utc),
                    pm));

        cur_utc += 3600;
        algo_qty_= 50;
        eng_qty_ = 50;
        pm.setPosition_Market(algo_qty_, eng_qty_, algo_pnl_, mkt_pnl_);
        EXPECT_FALSE(mon.checkReplace(strat, sym, -100, pm, cur_utc));
        EXPECT_FALSE(mon.checkNewOrder(strat, sym, -100, pm, cur_utc));
        EXPECT_FALSE(mon.updateER(
                    genER("2", -100, cur_utc),
                    pm));

        EXPECT_TRUE(mon.status().getPause(strat, mkt));
        mon.status().setPause(strat, mkt, false);
    }

    // order counts
    // eng:   [60 orders @60sec, 120 orders @300sec]
    // strat: [10 orders @60sec, 30 orders @300sec]
    {
        mon.status().setPause("", "", false);
        cur_utc += 24*3600;
        // hit the strat order count at 60sec
        for (int i=0; i<10; ++i) {
            EXPECT_TRUE(mon.checkNewOrder(strat, sym, 2, pm, cur_utc));
            auto er(genER("0", 2, cur_utc));
            EXPECT_TRUE(mon.updateER(er, pm));
            cur_utc += 5;
        }
        auto er(genER("0", 2, cur_utc));
        EXPECT_FALSE(mon.checkNewOrder(strat, sym, 2, cur_utc));
        EXPECT_FALSE(mon.updateER(er, pm)); // notify pause (?)

        // should be paused for now
        EXPECT_TRUE(mon.status().getPause(strat, mkt));

        // but we can still cancel
        auto er2(genER("4", 1, cur_utc));
        mon.updateER(er2);
        mon.updateER(er2);

        // report side comes through
        auto er3(genER("0", 2, cur_utc));
        EXPECT_TRUE(mon.updateER(er3, pm));

        // order side paused
        EXPECT_FALSE(mon.checkNewOrder(strat, sym, 1, cur_utc)); //paused
        mon.status().setPause(strat, mkt, false);
        EXPECT_TRUE(mon.checkNewOrder(strat, sym, 1, cur_utc));
    };
}

int main(int argc, char** argv) {
    //utils::PLCC::ToggleTest(true);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

