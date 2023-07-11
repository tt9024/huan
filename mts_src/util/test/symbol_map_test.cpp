#include "symbol_map.h"
#include "stdio.h"
#include <string>
#include "gtest/gtest.h"
#include <cstdlib>
#include "plcc/PLCC.hpp"

class SymFixture : public testing::Test {
public:
    SymFixture ():
    _cfg_str ( \
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
        "        expiration_date = 2020-01-20\n"
        "        mts_symbol = WTI_N0\n"
        "        N = 0\n"
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
        "        expiration_days = 20\n"
        "    }\n"
        "    BRN FMG0020! = {\n"
        "        symbol = Brent\n"
        "        exch_symbol = R\n"
        "        venue = IFEU\n"
        "        mts_contract = Brent_202002\n"
        "        tick_size = 0.010000000000\n"
        "        point_value = 1000.0\n"
        "        px_multiplier = 1.000000000000\n"
        "        type = FUT\n"
        "        contract_month = 202002\n"
        "        tt_security_id = 3971113080253836648\n"
        "        bbg_id = LCO CMD\n"
        "        bbg_px_multiplier = 1.0\n"
        "        tt_venue = ICE_L\n"
        "        currency = USD\n"
        "        expiration_date = 2020-02-20\n"
        "        mts_symbol = Brent_N0\n"
        "        N = 0\n"
        "        expiration_days = 5\n"
        "    }\n"
        "}"),
    _fn("/tmp/cfg_test.cfg") 
    {}

    void writeCfg(const char*fn, const char* cfg_str) const {
        FILE* fp = fopen(fn, "w");
        fprintf(fp, "%s\n", cfg_str);
        fclose(fp);
    }

    template<typename T>
    void testReader(const T& sr) const {
        const auto* trd = sr.getByMtsSymbol("WTI_N1");
        EXPECT_STREQ(trd->_tradable.c_str(), "CLH0");
        EXPECT_STREQ(trd->_mts_contract.c_str(), "CL_202003");
        EXPECT_STREQ(trd->_symbol.c_str(), "WTI");
        EXPECT_STREQ(trd->_exch_symbol.c_str(), "CL");
        EXPECT_STREQ(trd->_contract_month.c_str(), "202003");
        EXPECT_STREQ(trd->_tt_security_id.c_str(), "3971113080253836647");
        EXPECT_STREQ(trd->_currency.c_str(), "USD");
        EXPECT_STREQ(trd->_expiration_date.c_str(), "2020-03-20");
        EXPECT_STREQ(trd->_type.c_str(), "FUT");
        EXPECT_DOUBLE_EQ(trd->_px_multiplier, 0.01);

        trd = sr.getByTradable("BRN FMG0020!");
        EXPECT_STREQ(trd->_tradable.c_str(), "BRN FMG0020!");
        EXPECT_STREQ(trd->_mts_contract.c_str(), "Brent_202002");
        EXPECT_STREQ(trd->_symbol.c_str(), "Brent");
        EXPECT_STREQ(trd->_exch_symbol.c_str(), "R");
        EXPECT_STREQ(trd->_venue.c_str(), "IFEU");
        EXPECT_STREQ(trd->_mts_symbol.c_str(), "Brent_N0");
        EXPECT_STREQ(trd->_contract_month.c_str(), "202002");
        EXPECT_DOUBLE_EQ(trd->_point_value, 1000.0);
        EXPECT_DOUBLE_EQ(trd->_tick_size, 0.01);
        EXPECT_EQ(trd->_expiration_days, 5);
        EXPECT_EQ(trd->_N, 0);

        const auto& tv = sr.getAllBySymbol("WTI");
        EXPECT_EQ(tv.size(), 2);
        EXPECT_STREQ(tv[0]->_tradable.c_str(), "CLG0");
        EXPECT_STREQ(tv[1]->_tradable.c_str(), "CLH0");

        const auto& tv2 = sr.getAllByTradable("BRN FMG0020!");
        EXPECT_EQ(tv2.size(), 1);
        EXPECT_STREQ(tv2[0]->_mts_symbol.c_str(), "Brent_N0");

        EXPECT_STREQ(sr.getN1("CLG0")->_tradable.c_str(), "CLH0");
        EXPECT_STREQ(sr.getN1("CLH0")->_tradable.c_str(), "CLH0");
        EXPECT_EQ(sr.getN1("BRN FMG0020!"), nullptr);
    }

    const char* _cfg_str;
    const char* _fn;
};

TEST_F (SymFixture, test) {
    // load the file and write to it
    writeCfg(_fn, _cfg_str);
    const auto& sr (utils::SymbolMapReader::getFile(_fn));
    testReader(sr);

    // update it and write and load it and compare
    //
    utils::SymbolMapWriter writer(_fn);
    writer.toConfigFile(_fn);
    const auto& sr2 = utils::SymbolMapReader::getFile(_fn);
    testReader(sr2);

    writer.clear();
    writer.addTradable(sr2);
    writer.toConfigFile(_fn);
    const auto& sr3 = utils::SymbolMapReader::getFile(_fn);
    testReader(sr3);

    utils::SymbolMapWriter writer2;
    writer2.addTradable(sr2.getByMtsSymbol("Brent_N0"));
    writer2.addTradable(writer);
    writer2.toConfigFile(_fn);
    const auto& sr4 = utils::SymbolMapReader::getFile(_fn);
    testReader(sr4);
}

TEST_F (SymFixture, sub) {
    const char* cfn = "/tmp/main.cfg";
    const char* main_cfg1 = 
"SymbolMap = /tmp/cfg_test.cfg\n"
"MDProviders = {\n"
"                  BPipe = config/bpipe.cfg\n"
"                  TTFix = Config = config/legacy/feed_TT.xml\n"
"              }\n"
"MaxN = 2\n"
"MTSVenue = {\n"
"               IFEU= [TTFix , BPipe]\n"
"           }\n"
"MTSSymbol = {\n"
"              WTI=   [BPipe, TTFix ]\n"
"            }\n";
    writeCfg(cfn, main_cfg1);
    utils::PLCC::setConfigPath(cfn);

    //BPipe
    const auto& sr = utils::SymbolMapReader::getFile(_fn);
    const auto& bp = sr.getSubscriptions("BPipe");
    const auto& tt = sr.getSubscriptions("TTFix");

    EXPECT_EQ(bp.first.size(), 2);
    std::set<std::string> bps (bp.first.begin(), bp.first.end());
    EXPECT_TRUE(bps.find("WTI_N1") != bps.end());
    EXPECT_TRUE(bps.find("WTI_N0") != bps.end());
    EXPECT_EQ(bp.second.size(), 1);
    EXPECT_STREQ(bp.second[0].c_str(), "Brent_N0");

    EXPECT_EQ(tt.second.size(), 2);
    std::set<std::string> tts (tt.second.begin(), tt.second.end());
    EXPECT_TRUE(tts.find("WTI_N1") != tts.end());
    EXPECT_TRUE(tts.find("WTI_N0") != tts.end());
    EXPECT_EQ(tt.first.size(), 1);
    EXPECT_STREQ(tt.first[0].c_str(), "Brent_N0");

    const char* main_cfg2 =
"SymbolMap = /tmp/cfg_test.cfg\n"
"MDProviders = {\n"
"                  BPipe = config/bpipe.cfg\n"
"                  TTFix = Config = config/legacy/feed_TT.xml\n"
"              }\n"
"MaxN = 2\n"
"MTSVenue = {\n"
"             NYM = [BPipe]\n"
"           }\n"
"MTSSymbol = {\n"
"               #Brent = []\n"
"            }\n";
    writeCfg(cfn, main_cfg2);
    const auto& bp2 = sr.getSubscriptions("BPipe");
    const auto& tt2 = sr.getSubscriptions("TTFix");

    EXPECT_EQ(bp2.first.size(), 2);
    bps.clear();
    bps.insert(bp2.first.begin(), bp2.first.end());
    EXPECT_TRUE(bps.find("WTI_N1") != bps.end());
    EXPECT_TRUE(bps.find("WTI_N0") != bps.end());
    EXPECT_EQ(bp2.second.size(), 0);

    EXPECT_EQ(tt2.second.size(), 0);
    EXPECT_EQ(tt2.first.size(), 0);

    const char* main_cfg3 =
"SymbolMap = /tmp/cfg_test.cfg\n"
"MDProviders = {\n"
"                  BPipe = config/bpipe.cfg\n"
"                  TTFix = Config = config/legacy/feed_TT.xml\n"
"              }\n"
"MaxN = 2\n"
"MTSVenue = {\n"
"             IFEU= [BPipe,TTFix]\n"
"             NYM = [BPipe]\n"
"           }\n"
"MTSSymbol = {\n"
"            }\n";
    writeCfg(cfn, main_cfg3);
    const auto& bp3 = sr.getSubscriptions("BPipe");
    const auto& tt3 = sr.getSubscriptions("TTFix");

    EXPECT_EQ(bp3.first.size(), 3);
    bps.clear();
    bps.insert (bp3.first.begin(), bp3.first.end());
    EXPECT_TRUE(bps.find("WTI_N1") != bps.end());
    EXPECT_TRUE(bps.find("WTI_N0") != bps.end());
    EXPECT_TRUE(bps.find("Brent_N0") != bps.end());
    EXPECT_EQ(bp3.second.size(), 0);

    EXPECT_EQ(tt3.second.size(), 1);
    EXPECT_STREQ(tt3.second[0].c_str(), "Brent_N0");
    EXPECT_EQ(tt3.first.size(), 0);

    const char* main_cfg4 =
"SymbolMap = /tmp/cfg_test.cfg\n"
"MDProviders = {\n"
"                  BPipe = config/bpipe.cfg\n"
"                  TTFix = Config = config/legacy/feed_TT.xml\n"
"              }\n"
"MaxN = 2\n"
"MTSVenue = {\n"
"             IFEU= [BPipe,TTFix]\n"
"             NYM = [BPipe]\n"
"           }\n"
"MTSSymbol = {\n"
"            Brent = [TTFix]\n"
"            }\n";
    writeCfg(cfn, main_cfg4);
    try {
        const auto& bp4 = sr.getSubscriptions("BPipe");
        EXPECT_EQ(1, 2);
    } catch (const std::exception & e) {
        // WTI 
    }

    const char* main_cfg5 =
"SymbolMap = /tmp/cfg_test.cfg\n"
"MDProviders = {\n"
"                  BPipe = config/bpipe.cfg\n"
"                  TTFix = Config = config/legacy/feed_TT.xml\n"
"              }\n"
"MaxN = 2\n"
"MTSVenue = {\n"
"             IFEU= [BPipe,BPipe]\n"
"             NYM = [BPipe]\n"
"           }\n"
"MTSSymbol = {\n"
"            Brent = [BPipe]\n"
"            }\n";
    writeCfg(cfn, main_cfg5);
    const auto& bp5 = sr.getSubscriptions("BPipe");
    const auto& tt5 = sr.getSubscriptions("TTFix");

    EXPECT_EQ(bp5.first.size(), 3);
    bps.clear();
    bps.insert (bp5.first.begin(), bp5.first.end());
    EXPECT_TRUE(bps.find("WTI_N1") != bps.end());
    EXPECT_TRUE(bps.find("WTI_N0") != bps.end());
    EXPECT_TRUE(bps.find("Brent_N0") != bps.end());
    EXPECT_EQ(bp5.second.size(), 0);

    EXPECT_EQ(tt5.second.size(), 0);
    EXPECT_EQ(tt5.first.size(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

