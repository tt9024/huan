#include "csv_util.h"
#include <iostream>
#include "gtest/gtest.h"
#include <cstdio>
#include <stdio.h>

class CSVFixture : public testing::Test {
public:
    CSVFixture ():
    _lines( {"a, 1, 2.0, , ", 
             "b, 2, 0.2, a ,c ",
             "c, 3, 2 2 , ,"
            } 
          ),
    _ftr( { {"a", "1", "2.0", "", ""}, 
            {"b", "2", "0.2", "a", "c"},
            {"c", "3", "2 2", "", ""}
         }
       )
    { 
        _tmpfile = "/tmp/csv_test";
    }

    void TearDown() {
        if (_tmpfile.size()>0) {
            remove (_tmpfile.c_str());
        }
    }

protected:
    std::string _tmpfile;
    std::vector<std::string> _lines;
    utils::CSVUtil::FileTokens _ftr;
};

TEST_F (CSVFixture, Read) {
    std::ofstream filecsv(_tmpfile);
    for (const auto& l : _lines) {
        filecsv << l << std::endl;
    }
    auto ft = utils::CSVUtil::read_file(_tmpfile);
    EXPECT_EQ (ft, _ftr);

    // debug
    {
        for (const auto& l: ft) {
            for (const auto& t: l) {
                std::cout << "[" << t <<"]";
            }
            std::cout << std::endl;
        }
    }

    // append
    utils::CSVUtil::write_file(ft, _tmpfile, true);
    auto ft2 = utils::CSVUtil::read_file(_tmpfile);
    EXPECT_EQ(ft2[0], ft2[3]);
    EXPECT_EQ(ft2[1], ft2[4]);

};

TEST_F (CSVFixture, Double) {
    std::vector<double> da {0, 1.0, 1.000001, 1.66666666, 12.6, 123.0000001, 1234.66666666, 1e+12};
    std::vector<std::vector<std::string> > das { 
                           {"0",   "1",   "1",        "2",        "13",   "123",   "1235",        "1000000000000"}, 
                           {"0.0", "1.0", "1.0",      "1.7",      "12.6", "123.0", "1234.7",      "1000000000000.0"},
                           {"0.0", "1.0", "1.000001", "1.666667", "12.6", "123.0", "1234.666667", "1000000000000.0"} };
    std::vector<int> decimals {0,1,6};
    // positive number
    for (size_t di=0; di<decimals.size(); ++di) {
        int d = decimals[di];
        for (size_t i = 0; i < da.size() ; ++i) {
            double dv = da[i];
            EXPECT_EQ(utils::CSVUtil::printDouble(dv, d), das[di][i]);
            if (i>0) {
                // test for negative
                std::string ns = std::string("-") + das[di][i];
                EXPECT_EQ(utils::CSVUtil::printDouble(-dv, d), ns);
            }
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

