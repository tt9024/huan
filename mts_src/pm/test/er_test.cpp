#include "ExecutionReport.h"
#include <iostream>
#include <cmath>
#include "gtest/gtest.h"
#include "plcc/PLCC.hpp"

void setupConfig() {
    const std::string cfgstr ("SymbolMap = /tmp/symbol_map.cfg\n");
    const std::string mapstr (
            "tradable = {\n"
            "    sym1 = {\n"
            "        symbol = WTI\n"
            "        exch_symbol = CL\n"
            "        venue = NYM\n"
            "        currency = USD\n"
            "        tick_size = 0.010000000000\n"
            "        point_value = 1.0\n"
            "        px_multiplier = 0.010000000000\n"
            "        type = FUT\n"
            "        mts_contract = WTI_202106\n"
            "        contract_month = 202106\n"
            "        mts_symbol = WTI_N1\n"
            "        N = 1\n"
            "        expiration_days = 5\n"
            "        tt_security_id = 12258454998215387229\n"
            "        tt_venue = NYM\n"
            "        currency = USD\n"
            "        expiration_date = 2021-05-20\n"
            "        bbg_id = CLM1 COMDTY\n"
            "        bbg_px_multiplier = 1.0\n"
            "        tickdata_id = CLM21\n"
            "        tickdata_px_multiplier = 1.000000000000\n"
            "        tickdata_timezone = America/New York\n"
            "    }\n"
            "}\n");

    std::ofstream cfg_file("/tmp/main.cfg");
    cfg_file << cfgstr;
    cfg_file.close();

    std::ofstream map_file("/tmp/symbol_map.cfg");
    map_file << mapstr;
    map_file.close();
}

TEST (ExecutionReportTest, TestER) {
    const std::string erfname = "/tmp/test_er.csv";
    utils::CSVUtil::FileTokens erlines = {
        {"sym1", "algo1","cid1", "eid1", "0","-10","2.1218", "20201004-18:33:02","", "1601850782023138"},
        {"sym1", "algo1","cid1", "eid2", "1","-3" ,"2.1218", "20201004-18:33:02","", "1601850782823033"},
        {"sym1", "algo1","cid2", "eid3", "0","5"  ,"2.1219", "20201004-18:33:08","", "1601850788504031"},
        {"sym1", "algo1","cid2", "eid4", "4","0"  ,"0"     , "20201004-18:33:08","", "1601850788591032"},
        {"sym1", "algo1","cid1", "eid5", "2","-7" ,"2.13"  , "20201004-18:33:09","", "1601850789504031"}
    };

    bool verbose = true;
    utils::CSVUtil::write_file(erlines, erfname, false);
    const auto& lines = utils::CSVUtil::read_file(erfname);

    std::vector<pm::ExecutionReport> erlist;
    for (const auto& line : lines) {
        erlist.push_back(pm::ExecutionReport::fromCSVLine(line));
    }

    if (verbose) {
        std::cout << "read er from csv: " << std::endl;
        for (const auto& er: erlist) {
            std::cout << er.toString() << std::endl;
        }
    }

    // write back to csv lines
    utils::CSVUtil::FileTokens erlines_derived;
    for (const auto& er : erlist) {
        erlines_derived.push_back(er.toCSVLine());
    }

    // load it differently
    std::vector<pm::ExecutionReport> erlist_derived;
    for (const auto& line: erlines_derived) {
        erlist_derived.emplace_back(
                line[0],
                line[1],
                line[2],
                line[3],
                line[4],
                std::stoi(line[5]),
                std::stod(line[6]),
                line[7],
                line[8],
                std::stoull(line[9]),
                0);
    };

    if (verbose) {
        std::cout << "derived from write back: " << std::endl;
        for (const auto& er: erlist_derived) {
            std::cout << er.toString() << std::endl;
        }
    }

    // compare the two
    std::string difflog;
    for (size_t i=0; i<erlist.size(); ++i) {
        EXPECT_TRUE(erlist[i].compareTo(erlist_derived[i], &difflog)) << difflog;
    }

    // test persistence
    FILE* fp = fopen(erfname.c_str(), "w");
    const auto& er0(erlist[0]);
    utils::CSVUtil::write_file(er0.toCSVLine(), fp);
    fclose(fp);
    const auto& erline = utils::CSVUtil::read_file(erfname); 
    EXPECT_EQ(1, erline.size());
    EXPECT_TRUE(er0.compareTo(pm::ExecutionReport::fromCSVLine(erline[0]), &difflog)) << difflog;
}

int main(int argc, char** argv) {
    utils::PLCC::setConfigPath("/tmp/main.cfg");
    setupConfig();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

