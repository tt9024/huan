#include "stdio.h"
#include <string>
#include "gtest/gtest.h"
#include <cstdlib>
#include "rate_limiter.h"
#include "rate_estimator.h"
#include "plcc/PLCC.hpp"

/*
TEST (RateTest, Check) {
    int a = 1;
    EXPECT_EQ(1 , a);
}
*/

TEST (RateTest, Check) {
    for(int hist_multiple = 0; hist_multiple<3;++hist_multiple)
    {
        {
            // 2 orders in 3 seconds
            // normal case 1
            utils::RateLimiter rc(2, 3, hist_multiple);
            EXPECT_EQ(rc.check(11000000), 0);
            EXPECT_EQ(rc.check(12000000), 0);
            EXPECT_EQ(rc.check(14000000), 0);
        };

        {
            // normal case 2
            utils::RateLimiter rc(2, 3, hist_multiple);
            EXPECT_EQ(rc.check(11000000), 0);
            EXPECT_EQ(rc.check(13000000), 0);
            EXPECT_EQ(rc.check(15000000), 0);
        };

        {
            // bad case 1
            utils::RateLimiter rc(2, 3, hist_multiple);
            EXPECT_EQ(rc.check(11000000), 0);
            EXPECT_EQ(rc.check(12000000), 0);
            EXPECT_EQ(rc.check(13000000), 1000000);
        };

        {
            // bad case 2
            utils::RateLimiter rc(2, 3, hist_multiple);
            EXPECT_EQ(rc.check(11000000), 0);
            EXPECT_EQ(rc.check(12000000), 0);
            EXPECT_EQ(rc.check(13000000), 1000000);
            EXPECT_EQ(rc.check(13500000), 500000);
            EXPECT_EQ(rc.check(14000000), 0);
        };

        {
            printf("hist_mult=%d\n", hist_multiple);
            // checkOnly and updateOnly
            utils::RateLimiter rc(2, 3, hist_multiple);
            EXPECT_EQ(rc.updateOnly(11000000,2), 0);
            EXPECT_EQ(rc.updateOnly(12000000,1), 2000000);
            EXPECT_EQ(rc.updateOnly(13000000,1), 1000000);
            EXPECT_EQ(rc.updateOnly(14000000,1), 1000000);
            EXPECT_EQ(rc.updateOnly(15000000,1), 1000000);
            EXPECT_EQ(rc.updateOnly(17000000,1), 0);
            EXPECT_EQ(rc.checkOnly(17000000,1),  1000000);
            EXPECT_EQ(rc.checkOnly(18000000,1),  0);
            EXPECT_EQ(rc.updateOnly(18000000,1),  0);
            EXPECT_EQ(rc.updateOnly(19000000,2),  2000000);
            EXPECT_EQ(rc.updateOnly(20000000,3),  3000000);
            EXPECT_EQ(rc.check(21000000), 2000000);
            EXPECT_EQ(rc.check(23000000), 0);
        };

        {
            // update and remove
            utils::RateLimiter rc(2, 3, hist_multiple);
            EXPECT_EQ(rc.updateOnly(11000000,10), 3000000);
            EXPECT_EQ(rc.checkOnly(13000000,10), 3000000);
            EXPECT_EQ(rc.checkOnly(13000000,1), 1000000);
        }
    }
}

TEST (RateTest, Check2) {

    // check the exception cases
    try {
        utils::RateLimiter rc(2, 3, -1);
        EXPECT_EQ(0,1);
    } catch (const std::exception& e) {
        // good case
    }

    utils::RateLimiter rc(2, 3, 1);
    try {
        rc.checkOnly(0, 0);
        EXPECT_EQ(0,1);
    } catch (const std::exception& e) {
        // good case
    }

    try {
        rc.updateOnly(0, 0);
        EXPECT_EQ(0,1);
    } catch (const std::exception& e) {
        // good case
    }

    try {
        rc.removeOnly(-1);
        EXPECT_EQ(0,1);
    } catch (const std::exception& e) {
        // good case
    }

    // with at least 1 _count of additional history
    for(int hist_multiple = 1; hist_multiple<3;++hist_multiple)
    {
        {
            // 2 orders in 3 seconds
            // update more than _count
            // update more than _hist_count
            utils::RateLimiter rc(2, 3, hist_multiple);
            EXPECT_EQ(rc.check(11000000, 3), 3000000);
            EXPECT_EQ(rc.updateOnly(11000000, 4), 3000000);
            EXPECT_EQ(rc.checkOnly(12000000, 1), 2000000);
            rc.removeOnly(1);
            EXPECT_EQ(rc.checkOnly(13000000, 1), 1000000);
            rc.removeOnly(1);
            EXPECT_EQ(rc.checkOnly(13000000, 1), 1000000);

            auto rc0 (utils::RateLimiter::loads(rc.toStringDump()));
            std::vector<utils::RateLimiter*> rc_vec {&rc, rc0.get()};
            for (auto rc_ptr: rc_vec) {
                rc_ptr->removeOnly(1);
                EXPECT_EQ(rc_ptr->check(13000000, 1), 0);
                EXPECT_EQ(rc_ptr->checkOnly(13000000, 1), 1000000);
                EXPECT_EQ(rc_ptr->check(14000000, 1), 0);
                // update and remove multiples, we have 4:
                // 1 at 13, 1 at 14, 2 at 15
                EXPECT_EQ(rc_ptr->updateOnly(15000000, 2), 2000000);

                // remove 3, should leave 1 at 13
                rc_ptr->removeOnly(3);
                EXPECT_EQ(rc_ptr->check(15000000, 1), 0);
                EXPECT_EQ(rc_ptr->check(15000000, 2), 3000000);
                EXPECT_EQ(rc_ptr->updateOnly(16000000, 1), 0);
            }

            // dump/load
            std::string s0 = rc0->toStringDump();
            auto rc1 (utils::RateLimiter::loads(s0));
            EXPECT_EQ(s0, rc1->toStringDump());
        };
    }
}

TEST (RateTest, CheckEns) {
    for(int hist_multiple = 1; hist_multiple<3;++hist_multiple)
    {
        {
            // 2 orders in 3 seconds and 4 orders in 4 seconds Ens
            // note the second rl should never hit.
            std::vector< std::tuple<int, time_t, int> > rlv = { 
                std::make_tuple(2,3,hist_multiple), 
                std::make_tuple(4,4,hist_multiple)};
            utils::RateLimiterEns rc(rlv);


            EXPECT_EQ(rc.check(11000000, 3), 3000000);
            EXPECT_EQ(rc.updateOnly(11000000, 4), 3000000);
            EXPECT_EQ(rc.checkOnly(12000000, 1), 3000000); // the 4/4 is longer
            rc.removeOnly(1);
            EXPECT_EQ(rc.checkOnly(13000000, 1), 1000000);
            rc.removeOnly(1);
            EXPECT_EQ(rc.checkOnly(13000000, 1), 1000000);

            auto rc0 (utils::RateLimiterEns::loads(rc.toStringDump()));
            utils::RateLimiterEns rc1 (*rc0);
            // copy 
            std::vector<utils::RateLimiterEns*> rc_vec {&rc, &rc1};

            for (auto rc_ptr: rc_vec) {
                rc_ptr->removeOnly(1);
                EXPECT_EQ(rc_ptr->check(13000000, 1), 0);
                EXPECT_EQ(rc_ptr->checkOnly(13000000, 1), 1000000);
                EXPECT_EQ(rc_ptr->check(14000000, 1), 0);
                // update and remove multiples, we have 4:
                // 1 at 13, 1 at 14, 2 at 15
                EXPECT_EQ(rc_ptr->updateOnly(15000000, 2), 2000000);

                // assignment
                utils::RateLimiterEns rc2 (*rc0);
                rc2 = *rc_ptr;
                rc_ptr = &rc2;

                // remove 3, should leave 1 at 13
                rc_ptr->removeOnly(3);
                EXPECT_EQ(rc_ptr->check(15000000, 1), 0);
                EXPECT_EQ(rc_ptr->check(15000000, 2), 3000000);
                EXPECT_EQ(rc_ptr->updateOnly(16000000, 1), 0);
            }

            {
                // dump/load
                std::string s0 = rc0->toStringDump();
                std::shared_ptr<utils::RateLimiterEns> rle;
                {
                    auto rc_ptr (utils::RateLimiterEns::loads(s0));
                    EXPECT_EQ(s0, rc_ptr->toStringDump());
                    rc_ptr->updateOnly(16000000, 2);
                    s0 = rc_ptr->toStringDump();
                    // copy constructor
                    rle = std::make_shared<utils::RateLimiterEns>(*rc_ptr);
                }
                EXPECT_EQ(s0, rle->toStringDump());
                {
                    auto rc_ptr (utils::RateLimiterEns::loads(s0));
                    rc_ptr->updateOnly(19000000, 2);
                    s0 = rc_ptr->toStringDump();
                    // assignment
                    *rle = *rc_ptr;
                }
                EXPECT_EQ(s0, rle->toStringDump());
            }
        };
    }
}


TEST (RateTest, RateEstimator1) {
    std::vector<std::pair<double, int>> rate_seconds;
    rate_seconds.push_back(std::make_pair<double, int>(1, 10));
    int bucket_seconds = 1;
    // rate estimator on 1/sec for trailing 10 seconds
    {
        // good case, basic simple forms
        utils::RateEstimator re(rate_seconds, bucket_seconds);
        EXPECT_EQ(re.check(), 0);
        for (int i=0; i<9; ++i) {
            EXPECT_EQ(re.check(), 0);
        }
        re.updateOnly();
        EXPECT_EQ(re.check()>=1000000ULL, true);
    }
    {
        // one-by-one cases
        utils::RateEstimator re(rate_seconds, bucket_seconds);
        auto cur_utc = utils::TimeUtil::cur_utc();

        EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);
        for (int i=1; i<10; ++i) {
            EXPECT_EQ(re.checkUTC(cur_utc+i, 1), 0);
        }
        cur_utc += 9;  // utc+9
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 1);

        cur_utc+= 1; // utc+10
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc, 2), 2);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc, 2), 4);
        cur_utc += 1;
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 4);
        re.removeOnly(1);
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 3);
        cur_utc += 1;
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 2);
        cur_utc += 1;
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 1);
        cur_utc += 1;
        EXPECT_EQ(re.checkOnlyUTC(cur_utc, 1), 0);
        EXPECT_EQ(re.checkUTC(cur_utc, 2), 1);
        EXPECT_EQ(re.checkUTC(cur_utc, 3), 2);
        EXPECT_EQ(re.checkUTC(cur_utc, 4), 3);
        EXPECT_EQ(re.checkUTC(cur_utc, 5), 4);
        EXPECT_EQ(re.checkUTC(cur_utc, 6), 5);
        EXPECT_EQ(re.checkUTC(cur_utc, 7), 6);
        EXPECT_EQ(re.checkUTC(cur_utc, 8), 6);
        EXPECT_EQ(re.checkUTC(cur_utc, 9), 6);
        EXPECT_EQ(re.checkUTC(cur_utc, 10), 6);
        EXPECT_EQ(re.checkUTC(cur_utc-1, 10), 7);
        EXPECT_EQ(re.checkUTC(cur_utc+1, 10), 5);
        EXPECT_EQ(re.checkUTC(cur_utc+2, 10), 4);
        EXPECT_EQ(re.checkUTC(cur_utc+3, 10), 3);
        EXPECT_EQ(re.checkUTC(cur_utc+4, 10), 2);
        EXPECT_EQ(re.checkUTC(cur_utc+5, 10), 1);
        EXPECT_EQ(re.checkUTC(cur_utc+6, 9), 0);
        EXPECT_EQ(re.checkUTC(cur_utc+6, 1), 0);
        EXPECT_EQ(re.checkUTC(cur_utc+6, 1), 10);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+7, 1), 9);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+8, 1), 8);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+9, 1), 7);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+10, 1), 6);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+11, 1), 5);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+12, 1), 4);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+13, 1), 3);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+14, 1), 2);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+15, 1), 1);
        EXPECT_EQ(re.updateOnlyUTC(cur_utc+16, 1), 0);

        // sometime later...
        cur_utc += 16;
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+1, 1), 0);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+1, 2), 1);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+1, 3), 2);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+1, 4), 3);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+2, 5), 3);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+3, 6), 3);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+4, 7), 3);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+5, 8), 3);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+6, 9), 3);
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+7, 10), 3);

        // this update_cnt exceed the limit of 10, taken
        // as the limit
        EXPECT_EQ(re.checkOnlyUTC(cur_utc+7, 11), 3);
    }
}

TEST (RateTest, RateEstimator2) {
    std::vector<std::pair<double, int>> rate_seconds;
    rate_seconds.push_back(std::make_pair<double, int>(1, 10));
    {
        // run this one with bucket and multiple rates
        std::vector<std::pair<double, int>> rate_seconds;

        rate_seconds.push_back(std::make_pair<double, int>(1, 10));
        rate_seconds.push_back(std::make_pair<double, int>(0.5, 20));
        //rate_seconds.push_back(std::make_pair<double, int>(1, 10));
        int bucket_seconds = 2;

        // this gonna be fun
        utils::RateEstimator re(rate_seconds, bucket_seconds);
        auto cur_utc = utils::TimeUtil::cur_utc()/bucket_seconds*bucket_seconds; //starts from 0 (even)
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);
        for (int i=1; i<10; ++i) {
            EXPECT_EQ(re.checkUTC(cur_utc+i, 1), 0);
        }
        cur_utc += 9; 
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 11);
        cur_utc+= 1;
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 10);
        cur_utc+= 1;
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 9);
        cur_utc+= 1;
        EXPECT_EQ(re.checkUTC(cur_utc, 1), 8);
        cur_utc+= 8;  //cur_utc = 20

        auto re1 (re);
        utils::RateEstimator re2(rate_seconds, bucket_seconds);
        re2 = re;

        std::vector<utils::RateEstimator> rev = {re1, re2};
        auto cur_utc_=cur_utc;

        for (auto re_ : rev) {
            re = re_;
            cur_utc = cur_utc_;
            // we get one free bucket worth of 2

            EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);
            cur_utc+= 1;  // cur_utc = 21
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);

            // need one more bucket, credit of 2
            // wait time is 1 since we are 1 second 
            // away from the next bucket of 22
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 1);
            EXPECT_EQ(re.checkUTC(cur_utc, 2), 1);

            re.removeOnly(1);
            EXPECT_EQ(re.updateOnlyUTC(cur_utc, 2), 1);
            re.removeOnly(1);
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 1);
            EXPECT_EQ(re.checkUTC(cur_utc, 2), 1);

            // now we allow 12 for 20 seconds
            std::vector<double> new_rates = {1, 0.6};
            re.updateRateLimit(new_rates);

            // can stick 2 more to the current bucket
            EXPECT_EQ(re.checkUTC(cur_utc, 2), 0);

            // but no more
            EXPECT_EQ(re.checkUTC(cur_utc, 2), 1);

            cur_utc += 1;  // cur_utc = 22
            EXPECT_EQ(re.checkUTC(cur_utc, 2), 0);
            cur_utc += 2;  // cur_utc = 24
            EXPECT_EQ(re.checkUTC(cur_utc, 2), 0);
            cur_utc += 2; // 26
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);
            cur_utc += 1; // 27
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);

            // now we have 4 at 20, 2 at 22, 2 at 24, 
            // 2 at 26, this already full for (1,10)
            cur_utc += 1; // 28, violates the (1,10)
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 2);
            re.removeOnly(2);  // removed 26/27
            EXPECT_EQ(re.updateOnlyUTC(cur_utc, 3), 2);
            re.removeOnly(1);  // now 28 has 2
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 2);

            cur_utc += 2; // 30, hit 2,  (0.6,20) is full
            EXPECT_EQ(re.checkUTC(cur_utc, 2), 0);

            cur_utc += 1; // 31, violates (0.6, 20)
            EXPECT_EQ(re.updateOnlyUTC(cur_utc, 1), 9);
            re.removeOnly(1);
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 9);

            cur_utc += 9; // 40
            EXPECT_EQ(re.updateOnlyUTC(cur_utc, 4), 0);
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 2);
            cur_utc += 1; // 41
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 1);
            cur_utc += 2; // 42
            EXPECT_EQ(re.checkUTC(cur_utc, 1), 0);

            re.resetAll();
            // normalize cur_utc w.r.t. bucket
            cur_utc = cur_utc/bucket_seconds*bucket_seconds;
            for (int i=0; i<10; ++i) {
                EXPECT_EQ(re.checkUTC(cur_utc+i, 1), 0);
            }
            cur_utc += 9;
            // create a copy from the toString
            std::string str = re.toString(cur_utc);
            auto re0 = utils::RateEstimator::loads(str);
            std::vector<utils::RateEstimator*>re_list = {&re, re0.get()};

            auto cur_utc0 = cur_utc;
            for (auto re_ptr: re_list) {
                cur_utc = cur_utc0;
                // hit 10 1.0
                EXPECT_EQ(re_ptr->checkUTC(cur_utc, 1), 1);
                cur_utc += 1;  // cur_utc = 10
                EXPECT_EQ(re_ptr->checkUTC(cur_utc, 1), 0);
                EXPECT_EQ(re_ptr->checkUTC(cur_utc, 1), 0);
            }
            // check the rates should be all 0.6 and 1.0
            std::vector<double> rates;
            rates = re.checkRates(cur_utc);
            EXPECT_DOUBLE_EQ(rates[0], 1.0);
            EXPECT_DOUBLE_EQ(rates[1], 0.6);
            rates = re0->checkRates(cur_utc);
            EXPECT_DOUBLE_EQ(rates[0], 1.0);
            EXPECT_DOUBLE_EQ(rates[1], 0.6);

            cur_utc += 1; // 11
            rates = re.checkRates(cur_utc);
            EXPECT_DOUBLE_EQ(rates[0], 1.0);
            EXPECT_DOUBLE_EQ(rates[1], 0.6);
            rates = re0->checkRates(cur_utc);
            EXPECT_DOUBLE_EQ(rates[0], 1.0);
            EXPECT_DOUBLE_EQ(rates[1], 0.6);

            cur_utc0 = cur_utc;
            for (auto re_ptr: re_list) {
                cur_utc = cur_utc0;
                EXPECT_EQ(re_ptr->checkUTC(cur_utc, 1), 9);
                cur_utc += 9;
                EXPECT_EQ(re_ptr->checkUTC(cur_utc, 1), 0);
                EXPECT_EQ(re_ptr->checkUTC(cur_utc, 1), 0);
                cur_utc += 1;
                EXPECT_EQ(re_ptr->checkUTC(cur_utc, 1), 1);
            };
        }
    }
}

int main(int argc, char** argv) {
    utils::PLCC::ToggleTest(true);
    ::testing::InitGoogleTest(&argc, argv);
    //utils::PLCC::instance().loggerStdoutON();
    return RUN_ALL_TESTS();
    //utils::PLCC::instance().loggerStdoutOFF();
}

