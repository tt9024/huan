#include "time_util.h"
#include "stdio.h"
#include <string>
#include "gtest/gtest.h"
#include <cstdlib>

TEST (TimeTest, FracReadWrite) {
    const char* ts1[] = {
        "20201101-02:00:00",
        "20200308-03:00:00.0",
        "20200101-20:16:32.123",
        "20201030-20:16:32.124678"};

    int dec1[] = {0,1,3,6};
    char buf[32];
    for (int i=0; i<4; ++i) {
        auto ts = ts1[i];
        auto dec = dec1[i];
        uint64_t utc = utils::TimeUtil::string_to_frac_UTC(ts, dec);
        utils::TimeUtil::frac_UTC_to_string(utc, buf, sizeof(buf), dec);
        printf("%s %lld %s %d\n", ts, (long long)utc, buf, (int)strcmp(ts, buf));
        EXPECT_STREQ (ts, buf);
    }

    int dec2[] = {1,0,5,3};
    const char* ts2[] = {
        "20201101-02:00:00.0", 
        "20200308-03:00:00",
        "20200101-20:16:32.12300",
        "20201030-20:16:32.124"};

    for (int i=0; i<4; ++i) {
        auto ts = ts1[i];
        auto dec = dec2[i];
        auto tsr = ts2[i];
        uint64_t utc = utils::TimeUtil::string_to_frac_UTC(ts, dec);
        utils::TimeUtil::frac_UTC_to_string(utc, buf, sizeof(buf), dec);
        printf("%s %lld %s %d\n", ts, (long long)utc, buf, (int)strcmp(tsr, buf));
        EXPECT_STREQ (tsr, buf);
    }

    // testing the current micro
    uint64_t cur_micro = utils::TimeUtil::cur_micro();
    utils::TimeUtil::frac_UTC_to_string(cur_micro, buf, sizeof(buf), 6, "%Y%m%d%H%M%S");

    std::string ct_milli = utils::TimeUtil::frac_UTC_to_string(0, 3, "%Y%m%d%H%M%S");
    std::string ct_sec = utils::TimeUtil::frac_UTC_to_string(0, 0, "%Y%m%d%H%M%S");
    std::string ct_nano = utils::TimeUtil::frac_UTC_to_string(0, 9, "%Y%m%d%H%M%S");

    uint64_t ct_milli_ts = utils::TimeUtil::string_to_frac_UTC(ct_milli.c_str(), 3, "%Y%m%d%H%M%S");
    uint64_t ct_sec_ts = utils::TimeUtil::string_to_frac_UTC(ct_sec.c_str(), 0, "%Y%m%d%H%M%S");
    uint64_t ct_nano_ts = utils::TimeUtil::string_to_frac_UTC(ct_nano.c_str(), 9, "%Y%m%d%H%M%S");

    // demand they be less tha 1000 micro
    EXPECT_LT(std::abs((long long) cur_micro - (long long)(ct_milli_ts*1000ULL)),  1000000LL -1);
    EXPECT_LT(std::abs((long long) cur_micro - (long long)(ct_sec_ts*1000000ULL)), 1000000LL -1);
    EXPECT_LT(std::abs((long long) cur_micro - (long long)(ct_nano_ts/1000ULL)),   1000000LL -1);

    // debug
    printf("%s, %s, %s, %s\n", buf, ct_milli.c_str(), ct_sec.c_str(), ct_nano.c_str());
}

TEST (TimeTest, TradingDayCME) {
    // 20201206 is a Sunday
    std::vector<std::string> tstr 
        {"20201206-17:59:59", "20201206-18:00:00", "20201206-18:00:01", "20201207-16:59:59", "20201207-17:00:00", "20201207-17:00:01"};

    std::vector< std::vector<std::string> > tday {
        {"",                  "20201207",          "20201207",          "20201207",          "",                  ""        }, //snap 0
        {"20201204",          "20201207",          "20201207",          "20201207",          "20201207",          "20201207"}, //snap 1
        {"20201207",          "20201207",          "20201207",          "20201207",          "20201208",          "20201208"}, //snap 2
    };

    int sh = -6, sm = 0, eh = 17, em = 0;
    for (size_t i=0; i<tstr.size(); ++i) {
        time_t utc = utils::TimeUtil::string_to_frac_UTC(tstr[i].c_str(), 0);
        for (int snap=0; snap<3; ++snap) {
            const auto& t0(utils::TimeUtil::tradingDay(utc, sh, sm, eh, em, 0, snap));
            EXPECT_EQ(t0, tday[snap][i]);
        }
    }
}

TEST (TimeTest, TradingDayETF) {
    // 20201206 is a Sunday
    std::vector<std::string> tstr 
        {"20201207-09:29:59", "20201207-09:30:00", "20201207-09:30:01", "20201207-16:14:59", "20201207-16:15:00", "20201207-16:15:01"};

    std::vector< std::vector<std::string> > tday {
        {"",                  "20201207",          "20201207",          "20201207",          "",                  ""        }, //snap 0
        {"20201204",          "20201207",          "20201207",          "20201207",          "20201207",          "20201207"}, //snap 1
        {"20201207",          "20201207",          "20201207",          "20201207",          "20201208",          "20201208"}, //snap 2
    };

    int sh = 9, sm = 30, eh = 16, em = 15;
    for (size_t i=0; i<tstr.size(); ++i) {
        time_t utc = utils::TimeUtil::string_to_frac_UTC(tstr[i].c_str(), 0);
        for (int snap=0; snap<3; ++snap) {
            const auto& t0(utils::TimeUtil::tradingDay(utc, sh, sm, eh, em, 0, snap));
            EXPECT_EQ(t0, tday[snap][i]);
        }
    }
}

TEST (TimeTest, TradingDayCMEOffset) {
    // 20201206 is a Sunday
    std::vector<std::string> tstr 
        {"20201206-17:59:59", "20201206-18:00:00", "20201206-18:00:01", "20201207-16:59:59", "20201207-17:00:00", "20201207-17:00:01"};

    int snap = 2; // towards future
    std::vector< std::vector<std::string> > tday {
        {"20201204",          "20201204",          "20201204",          "20201204",          "20201207",          "20201207"}, // -1
        {"20201214",          "20201214",          "20201214",          "20201214",          "20201215",          "20201215"}, // +5
    };

    int sh = -6, sm = 0, eh = 17, em = 0;

    int offset[] = {-1, 5};
    for (size_t i=0; i<tstr.size(); ++i) {
        time_t utc = utils::TimeUtil::string_to_frac_UTC(tstr[i].c_str(), 0);
        for (int j=0; j<2; ++j) {
            const auto& t0(utils::TimeUtil::tradingDay(utc, sh, sm, eh, em, offset[j], snap));
            EXPECT_EQ(t0, tday[j][i]);
        }
    }
}

TEST (TimeTest, TradingDayETFOffset) {
    // 20201206 is a Sunday
    std::vector<std::string> tstr 
        {"20201207-09:29:59", "20201207-09:30:00", "20201207-09:30:01", "20201207-16:14:59", "20201207-16:15:00", "20201207-16:15:01"};

    int snap = 1; // towards before
    std::vector< std::vector<std::string> > tday {
        {"20201127",          "20201130",          "20201130",          "20201130",          "20201130",          "20201130"}, // -5
        {"20201207",          "20201208",          "20201208",          "20201208",          "20201208",          "20201208"}, // +1
    };

    int sh = 9, sm = 30, eh = 16, em = 15;
    int offset[] = {-5, 1};
    for (size_t i=0; i<tstr.size(); ++i) {
        time_t utc = utils::TimeUtil::string_to_frac_UTC(tstr[i].c_str(), 0);
        for (int j=0; j<2; ++j) {
            const auto& t0(utils::TimeUtil::tradingDay(utc, sh, sm, eh, em, offset[j], snap));
            EXPECT_EQ(t0, tday[j][i]);
        }
    }
}

TEST (TimeTest, StartUTC) {
    time_t utc[2][2] = { {1671652272, 1671577200}, // normal time
                       {1671577200, 1671577200}  // weekday snap forward
    };
    for (const auto& u: utc) {
        EXPECT_EQ(utils::TimeUtil::startUTC(u[0]), u[1]);
    }
}

TEST (TimeTest, FmtStr) {
    const char* td = "20201207";
    time_t utc_td = utils::TimeUtil::string_to_frac_UTC(td, 0, "%Y%m%d");
    auto td_str = utils::TimeUtil::frac_UTC_to_string(utc_td, 0, "%Y%m%d");
    EXPECT_STREQ(td, td_str.c_str());
}

TEST (TimeTest, GMTTime) {
    time_t ts0[] = { 1604214000,  1583650800 };
    const char* ts1[] = {
        "20201101-07:00:00",
        "20200308-07:00:00.126" 
    };
    EXPECT_EQ(ts0[0],
        utils::TimeUtil::string_to_frac_UTC(ts1[0],0,NULL,true));
    EXPECT_EQ(ts0[1] * 100 + 12,
        utils::TimeUtil::string_to_frac_UTC(ts1[1],2,NULL,true));

    EXPECT_STREQ(utils::TimeUtil::frac_UTC_to_string(ts0[0],0,NULL,true).c_str(),
            ts1[0]);
    EXPECT_STREQ(utils::TimeUtil::frac_UTC_to_string(ts0[1] * 1000 + 126,3,NULL,true).c_str(),
            ts1[1]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

