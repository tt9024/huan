#include "AlgoBase.h"
#include "gtest/gtest.h"
#include "FloorBase.h"
#include "strat/AR1.h"
#include "plcc/PLCC.hpp"
#include <fstream>
#include <string>

TEST(AlgoBase, GetSetPosition) {
    const char* main_cfg_fn = "/tmp/main.cfg";
    {
        std::ofstream ofs;
        ofs.open (main_cfg_fn, std::ofstream::out | std::ofstream::trunc);
        ofs << "Logger = ." << std::endl;
    }
    utils::PLCC::setConfigPath(main_cfg_fn);

    pm::FloorBase m_floor("algobase_test",false);
    pm::FloorBase::MsgType msgreq(pm::FloorBase::GetPositionReq, nullptr, 0), msgresp;
    const pm::FloorBase::PositionRequest pr("CL001", "CLF1", 0, 0);
    msgreq.copyData((const char*)&pr, sizeof(pm::FloorBase::PositionRequest));
    msgresp.copyString("");
    EXPECT_TRUE(m_floor.m_channel->request(msgreq, msgresp));
    std::cout <<((pm::FloorBase::PositionRequest*)msgresp.buf)->qty_done << std::endl;

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
