#include "PositionManager.h"
#include "csv_util.h"
#include "time_util.h"
#include <iostream>
#include <cmath>
#include "gtest/gtest.h"
#include "plcc/PLCC.hpp"
#include <fstream>

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
            "    sym2 = {\n"
            "        symbol = WTI\n"
            "        exch_symbol = CL\n"
            "        venue = NYM\n"
            "        currency = USD\n"
            "        tick_size = 0.010000000000\n"
            "        point_value = 1.0\n"
            "        px_multiplier = 0.010000000000\n"
            "        type = FUT\n"
            "        mts_contract = WTI_202107\n"
            "        contract_month = 202107\n"
            "        mts_symbol = WTI_N2\n"
            "        N = 2\n"
            "        expiration_days = 35\n"
            "        tt_security_id = 12258454998215387223\n"
            "        tt_venue = NYM\n"
            "        currency = USD\n"
            "        expiration_date = 2021-06-20\n"
            "        bbg_id = CLN1 COMDTY\n"
            "        bbg_px_multiplier = 1.0\n"
            "        tickdata_id = CLN21\n"
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

utils::CSVUtil::FileTokens erlines = {
    {"sym1", "algo1","cid1", "eid1", "0","-10","3.0", "20201004-18:33:02","", "1601850782023138"},
    {"sym1", "algo1","cid1", "eid2", "1","-5" ,"3.0", "20201004-18:33:02","", "1601850782823033"},
    {"sym1", "algo1","cid2", "eid3", "0","5"  ,"2", "20201004-18:33:08","", "1601850788504031"},
    {"sym1", "algo1","cid2", "eid4", "4","0"  ,"0"     , "20201004-18:33:08","", "1601850788591032"},
    {"sym1", "algo1","cid1", "eid5", "2","-5" ,"1.0"  , "20201004-18:33:09","", "1601850789504031"},
    {"sym1", "algo2","cid3", "eid6", "0","5"  ,"3.0", "20201004-18:33:08","", "1601850788504031"},
    {"sym1", "algo2","cid3", "eid7", "1","1"  ,"3.0", "20201004-18:33:08","", "1601850788504031"},
    {"sym2", "algo1","cid4", "eid8", "0","-5"  ,"10.1", "20201004-18:33:08","", "1601850788504031"},
    {"sym2", "algo1","cid4", "eid9", "1","-1"  ,"10.1", "20201004-18:33:08","", "1601850788504031"}
    //,{"sym1", "algo2","cid3", "eid10", "4","0"  ,"0", "20201004-18:33:08","", "1601850788504031"}
};

const char* RecoveryCSV="recovery.csv";
const char* EODCSV = "eod_pos.csv";
const std::string RecoverPath = "/tmp";

class PMTest : public ::testing::Test {
public:
    const utils::CSVUtil::FileTokens loadEoD_CSVLines(const std::string& eod_file, std::string* latest_day_ptr, const pm::PositionManager& pmgr) const {
        return pmgr.loadEoD_CSVLines(eod_file, latest_day_ptr);
    }
};

TEST_F(PMTest,  LoadSave) {
    utils::CSVUtil::FileTokens erlines0;
    utils::CSVUtil::write_file(erlines0, RecoverPath+"/"+EODCSV, false);

    pm::PositionManager pmgr("testpm", RecoverPath);
    utils::CSVUtil::write_file(erlines, RecoverPath+"/"+RecoveryCSV, false);
    pmgr.loadRecovery(RecoveryCSV);

    pmgr.persist();
    pm::PositionManager pmgr2("testpm2", RecoverPath);

    std::cout << pmgr.toString() << std::endl;
    std::cout << pmgr2.toString() << std::endl;

    std::string difflog = pmgr.diff(pmgr2);
    EXPECT_FALSE (difflog.size() != 0) << "save-load mismatch!" << std::endl << difflog;

    utils::CSVUtil::FileTokens erlines_update = {
    {"sym1", "algo1","cid5", "eid11", "0","10","3.0", "20201004-18:33:02","", "1601850782023138"},
    {"sym1", "algo1","cid5", "eid12", "1","5" ,"3.0", "20201004-18:33:02","", "1601850782823033"}
    };

    for (const auto& line : erlines_update) {
        pmgr.update(pm::ExecutionReport::fromCSVLine(line));
    }

    // get position of sym1 algo1
    double vap, pnl;
    int64_t oqty;
    int64_t qty = pmgr.getPosition("algo1", "sym1", &vap, &pnl, &oqty);
    EXPECT_FALSE ((qty != -5) || (std::fabs(vap-2.0)>1e-10) || (std::fabs(pnl+5.0)>1e-10) || (oqty != 5)) << "getPosition algo1 sym1 mismatch!" << std::endl << pmgr.toString();

    // get Position of sym1
    // algo1: short 5 @ 2.0, 
    // algo2: long 1 @ 3.0
    oqty=0;
    qty = pmgr.getPosition("sym1", &vap, &pnl, &oqty);
    EXPECT_FALSE ( (qty!=-4) || (std::fabs(vap-2.0)>1e-10) || (std::fabs(pnl+6)>1e-10) || (oqty != 9) ) << "getPosition sym1 mismatch!" << std::endl << pmgr.toString();

    // list Positions of algo1
    // sym1: -5, 2.0 -5.0  oo(5, 3.0)
    // sym2: -1  10.1 0.0  oo(-4, 10.1)
    const std::string algo1 ("algo1");
    auto idpvec = pmgr.listPosition(&algo1);
    std::string idpvec_str;
    for (auto& idp:idpvec) {
        idpvec_str += (idp->toString() + "\n");
    }
    EXPECT_FALSE ((idpvec.size()!=2) || 
        (idpvec[0]->getPosition() != -5) || 
        (idpvec[1]->getPosition() != -1)) << "listPoisition mismatch!" << std::endl << idpvec_str;

    // list OO (2)
    
    auto oovec = pmgr.listOO(&algo1);
    std::string oovec_str;
    for (auto& oo : oovec) {
        oovec_str += (oo->toString() + "\n");
    }
    EXPECT_FALSE ( (oovec.size()!= 2)  ||
         (oovec[0]->m_open_qty != 5) ||
         (oovec[1]->m_open_qty != -4) ||
         std::fabs((oovec[1]->m_open_px - 10.1) > 1e-10) ) << "listOO mismatch!" << std::endl << oovec_str;


    // reconcile
    utils::CSVUtil::FileTokens erlines_update2 = {
    {"sym1", "algo1","cid5", "eid12", "1","5" ,"3.0", "20201004-18:33:02","", "1601850782823033"}, // a duplicate fill
    {"sym2", "algo1","cid4", "eid13", "2","-4","10.1", "20201004-18:33:02","", "1601850782023138"}
    };

    utils::CSVUtil::write_file(erlines_update, RecoverPath+"/"+RecoveryCSV, false);
    utils::CSVUtil::write_file(erlines_update2, RecoverPath+"/"+RecoveryCSV, true);


    // expect reconcile to fail
    bool ret=pmgr.reconcile(RecoveryCSV, difflog, true);
    std::cout << "reconcile diff: " << difflog << std::endl;
    EXPECT_FALSE (ret) << "reconcile mismatch!";

    // list Positions of algo1
    // sym1: -5, 2.0 -5.0  oo(5, 3.0)
    // sym2: -5  10.1 0.0  oo()
    idpvec = pmgr.listPosition(&algo1);
    EXPECT_FALSE ((idpvec.size()!=2) || 
        (idpvec[0]->getPosition() != -5) || 
        (idpvec[1]->getPosition() != -5)) << "reconciled listPoisition mismatch!";

    for (auto& idp:idpvec) {
        std::cout << idp->toString() << std::endl;
    }

    // 
    // list OO (1)
    
    oovec = pmgr.listOO(&algo1);
    EXPECT_FALSE ( (oovec.size()!= 1)  ||
         (oovec[0]->m_open_qty != 5) ||
         std::fabs((oovec[0]->m_open_px - 3.0) > 1e-10) ) << "listOO mismatch!";

    for (auto& oo : oovec) {
        std::cout << oo->toString() << std::endl;
    }

    // get pnl
    pnl = pmgr.getPnl(&algo1);
    EXPECT_FALSE (std::fabs(pnl+5.0)>1e-10) << "reconcile pnl mismatch! " << pnl;

    idpvec = pmgr.listPosition();
    EXPECT_FALSE (idpvec.size()!= 3) << "reconcile position mismatch! ";

    for (auto& idp : idpvec) {
        std::cout << idp->toString() << std::endl;
    }

    // loading eod mtm
    utils::CSVUtil::FileTokens erlines_mtm = {
        {"20220811-17:00:03","TSD-7000-370","WTI_202209","-27","92.71451178","51757.46","31385.65","1660328703517184","USD"},
        {"20220812-17:00:03","TSC-7000-380","Gilt_202209","-98","116.39671666","52498.75","23420.52","1660317309727922","GBP"}
    };
    utils::CSVUtil::write_file(erlines_mtm, RecoverPath+"/"+RecoveryCSV, false);
    std::string latest_day;
    const auto& line_vec(loadEoD_CSVLines(RecoverPath+"/"+RecoveryCSV, &latest_day, pmgr));
    EXPECT_TRUE( latest_day == "20220812-17:00:03" );
    EXPECT_EQ  ( line_vec.size(), 1 );
    for (unsigned int i=0; i < line_vec[0].size(); ++i) {
        EXPECT_TRUE  ( line_vec[0][i] == erlines_mtm[1][i+1] );
    }
};

int main(int argc, char** argv) {
    utils::PLCC::setConfigPath("/tmp/main.cfg");
    setupConfig();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
