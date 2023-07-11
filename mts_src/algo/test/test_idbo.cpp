#include "idbo/idbo_tf.h"
#include <iostream>
#include <cmath>
#include "gtest/gtest.h"
#include "FloorBase.h"
#include <cstdlib>

const char* main_cfg = "/tmp/main.cfg";
const char* strat_cfg = "/tmp/idbo.cfg";

void setupSymMap() {
    const std::string cfgstr ("SymbolMap = /tmp/symbol_map.cfg\n");
    const std::string mapstr (
            "tradable = {\n"
            "    CLM1 = {\n"
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
            "    CLN1 = {\n"
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

    FILE* fp = fopen(main_cfg, "a+");
    fprintf(fp, "%s", cfgstr.c_str());
    fclose(fp);

    fp = fopen("/tmp/symbol_map.cfg", "w");
    fprintf(fp, "%s", mapstr.c_str());
    fclose(fp);
}

void setupStratConfig() {
    FILE* fp = fopen(main_cfg, "a+");
    fprintf(fp, "RecoveryPath = /tmp\n");
    fclose(fp);

    std::system("mkdir -p /tmp/strat/idbo_test > /dev/null 2>&1");
    std::system("rm -fR /tmp/strat/idbo_test/*.csv > /dev/null 2>&1");

    const std::string cfgstr = 
"update_date = 20210506\n"
"strategy_code = 7000-335\n"
"strategy_weight = 0.5\n"
"symbols = {\n"
"    WTI_N1 = {\n"
"        signal_tf = 0\n"
"        time_from = 0800\n"
"        time_to = 1500\n"
"        thres_h = 66.76\n"
"        thres_l = 64.92\n"
"        sar_ds = 0.668792\n"
"        sar_ic = 0.01\n"
"        sar_cl = 0.2\n"
"        pos_n = 30\n"
"        inactive_bars = 191\n"
"    }\n"
"}\n";

    fp = fopen(strat_cfg, "w");
    fprintf(fp, "%s", cfgstr.c_str());
    fclose(fp);
}

namespace algo {
class IDBO_TF_Mock : public IDBO_TF {
public:

    IDBO_TF_Mock(const std::string& name, const std::string& strat_cfg_file, pm::FloorBase::ChannelType& channel, uint64_t cur_micro) :
        IDBO_TF(name, strat_cfg_file, channel)
    {
        onReload(cur_micro, m_cfg); // the derived
    }

    void onReload(uint64_t cur_micro, const std::string& config_file) {
        removeAllSymbol();
        m_param.clear(); // remnove all subscriptions

        try {
            const auto& reader = utils::ConfigureReader(config_file.c_str()).getReader("symbols");
            for (const auto& key: reader.listKeys()) {
                auto new_param( std::make_shared<Param> (key, reader.getReader(key),  (time_t)(cur_micro/1000000ULL)));
                new_param->_symid = 0;
                m_param[new_param->_symid] = new_param;
            }
        } catch (const std::exception& e) {
            logError("%s failed to load parameter file %s: %s",
                    m_name.c_str(), m_cfg.c_str(), e.what());
            return;
        }
        initState(cur_micro/1000000);
    }


    bool updateState(State& state) {
        state._pos += 10;
        state._tgt_pos -= 20;
        state._h += 1.01;
        state._stop -= 2.02;
        state.persist();
        return true;
    }

    bool checkState(const State& state) const {
        const auto& state0 (m_state.find(0)->second);
        return (state0->_pos == state._pos) &&
               (state0->_tgt_pos == state._tgt_pos) &&
               (state0->_stop == state._stop) &&
               (state0->_h == state._h);
    }

    State get_state(int symid) const {
        return *m_state.find(symid)->second;
    }
};
}

TEST (IDBO_StateTest, LoadSave) {
    //pm::FloorBase::ChannelType = std::make_unique<utils::Floor::Channel>();
    auto channel = std::make_unique<utils::Floor::Channel>();

    // retrieve empty state
    std::system("rm -f /tmp/strat/idbo_test/WTI_N1_state.csv > /dev/null 2>&1");
    auto idbo = algo::IDBO_TF_Mock("idbo_test", strat_cfg, channel, utils::TimeUtil::cur_micro());

    auto state = idbo.get_state(0);
    EXPECT_EQ(0, state._pos);
    EXPECT_EQ(0, state._tgt_pos);

    // state is updated and presisted back 
    idbo.updateState(state);

    EXPECT_EQ(10, state._pos);
    EXPECT_EQ(-20, state._tgt_pos);
    EXPECT_DOUBLE_EQ(1.01, state._h);
    EXPECT_DOUBLE_EQ(-2.02, state._stop);

    // update and put a second line
    idbo.updateState(state);

    std::system("mkdir -p /tmp/strat/idbo_test2 > /dev/null 2>&1");
    std::system("cp -fR /tmp/strat/idbo_test/*.csv /tmp/strat/idbo_test2 > /dev/null 2>&1");
    auto idbo2 = algo::IDBO_TF_Mock("idbo_test2", strat_cfg, channel, utils::TimeUtil::cur_micro());
    EXPECT_TRUE(idbo2.checkState(state));
    std::system("rm -fR /tmp/strat > /dev/null 2>&1");
}

int main(int argc, char** argv) {
    utils::PLCC::setConfigPath(main_cfg);
    setupSymMap();
    setupStratConfig();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

