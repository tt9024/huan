#include "stdio.h"
#include <string>
#include "gtest/gtest.h"
#include <cstdlib>
#include "thread_utils.h"
#include "time_util.h"

// this needs to be run with at least 2 CPUs
#define VOTERS 10

namespace spin {
    class SpinVoter : public utils::Runnable {
    public: 
        SpinVoter(int voter_id, int voters, long long* buf):_id(voter_id), _voters(voters), _buf(buf) {
        }

        void run(void*param) {
            // take turn to write voter id and 
            utils::SpinLock::LockType*lock = (utils::SpinLock::LockType*) param;
            _should_run = true;
            _stopped = false;
            try {
                while (true) {
                    utils::SpinLock slock(*lock);
                    if (!_should_run) {
                        throw std::runtime_error("stopped!");
                    }
                    auto & next_id = _buf[_voters];
                    if ( int(next_id%_voters) == _id ) {
                        // my turn
                        _buf[_id] = next_id++;
                    }
                }
            } catch(const std::exception& e) {
                printf("%s\n", e.what());
            }
            _stopped = true;
        }

        void stop() {
            _should_run = false;
            // spin to wait stopped
            while (!_stopped) ;
        }

    private:
        const int _id;
        const int _voters;
        long long* _buf;
        volatile bool _should_run;
        volatile bool _stopped;
    };
}

TEST (ThreadTest, Spin) {
    auto lock (utils::SpinLock::CreateSpinLock(false));
    long long buf[VOTERS+1];
    memset(buf, 0, sizeof(buf));
    std::vector<std::shared_ptr<utils::ThreadWrapper<spin::SpinVoter> > > voters;

    // create VOTERS and start them
    for (int i=0; i<VOTERS; ++i) {
        auto* voter (new spin::SpinVoter(i, VOTERS, buf));
        auto vthread (std::make_shared<utils::ThreadWrapper<spin::SpinVoter>>(*voter));
        vthread->run((void*) lock.get());
        voters.push_back(vthread);
    };

    // check buffer consistency and progress every 10 milli for 1 second
    while (buf[VOTERS] < VOTERS);
    long long idx = buf[VOTERS];

    for (int i=0; i<10; ++i) {
        utils::TimeUtil::micro_sleep(100000);
        const auto& next_ix = buf[VOTERS ];
        EXPECT_TRUE (next_ix > idx);
        idx = next_ix;
        {
            utils::SpinLock slock(*lock);
            for (int i=0; i<VOTERS; ++i) {
                EXPECT_EQ(buf[i]%VOTERS, i);
            }
        }
    }
    printf("done basic\n");

    // create more VOTER
    for (int i=0; i<VOTERS; ++i) {
        auto* voter (new spin::SpinVoter(i, VOTERS, buf));
        auto vthread (std::make_shared<utils::ThreadWrapper<spin::SpinVoter>>(*voter));
        vthread->run((void*) lock.get());
        voters.push_back(vthread);
    };

    // check again
    idx = buf[VOTERS];
    for (int i=0; i<100; ++i) {
        utils::TimeUtil::micro_sleep(1000);
        const auto& next_ix = buf[VOTERS];
        //EXPECT_TRUE (next_ix > idx);
        idx = next_ix;
        {
            utils::SpinLock slock(*lock);
            EXPECT_FALSE(utils::SpinLock::TryLock(*lock));
            for (int i=0; i<VOTERS; ++i) {
                EXPECT_EQ(buf[i]%VOTERS, i);
            }
        }
    }

    printf("done extended\n");

    // kill the first VOTERS
    for (int i=0; i<VOTERS; ++i) {
        voters[i]->stop();
    }

    // check again
    idx = buf[VOTERS];
    for (int i=0; i<10; ++i) {
        utils::TimeUtil::micro_sleep(100000);
        const auto& next_ix = buf[VOTERS];
        EXPECT_TRUE (next_ix > idx);
        idx = next_ix;
        while (true) {
            auto sl (utils::SpinLock::TryLock(*lock));
            if (!sl) {
                continue;
            }
            EXPECT_FALSE(utils::SpinLock::TryLock(*lock));
            for (int i=0; i<VOTERS; ++i) {
                EXPECT_EQ(buf[i]%VOTERS, i);
            }
            break;
        }
    }

    printf("done killing baisc\n");

    // kill 1 of them, wait and start 2
    for (int i=VOTERS; i<VOTERS+1; ++i) {
        voters[i]->stop();
    }

    // should now stop
    utils::TimeUtil::micro_sleep(100000);
    idx = buf[VOTERS];
    utils::TimeUtil::micro_sleep(100000);
    EXPECT_EQ (buf[VOTERS], idx);
    {
        utils::SpinLock slock(*lock);
        for (int i=0; i<VOTERS; ++i) {
            EXPECT_EQ(buf[i]%VOTERS, i);
        }
    }

    // start one more
    auto* voter (new spin::SpinVoter(0, VOTERS, buf));
    auto vthread (std::make_shared<utils::ThreadWrapper<spin::SpinVoter>>(*voter));
    vthread->run((void*) lock.get());

    // check again
    idx = buf[VOTERS];
    for (int i=0; i<10; ++i) {
        utils::TimeUtil::micro_sleep(100000);
        const auto& next_ix = buf[VOTERS];
        EXPECT_TRUE (next_ix > idx);
        idx = next_ix;
        {
            utils::SpinLock slock(*lock);
            EXPECT_FALSE(utils::SpinLock::TryLock(*lock));
            for (int i=0; i<VOTERS; ++i) {
                EXPECT_EQ(buf[i]%VOTERS, i);
            }
        }
    }

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

