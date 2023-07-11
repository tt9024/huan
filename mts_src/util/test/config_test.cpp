#include "plcc/ConfigureReader.hpp"
#include "stdio.h"
#include <string>
#include "gtest/gtest.h"
#include <cstdlib>

TEST (ConfigTest, ReadWrite) {
    const char* cfgstr = 
        "#test\n"
        "k1 = v1\n"
        "    # [ problem ] \n"
        "    # [ #aa #aa ] \n"
        "k2 = [ 2, 3 ]\n"
        "k3 = { \n"
        "       k31 = [2.1] \n"
        "       k32 =  abc  \n"
        "      }\n"
        "# this is a comment\n"
        "k4 = [ { k41 = [ { k411 = v411},{k412=[1,2,3]}]\n"
        "         k42 = 0.2},\n"
        "       { k43 = { k431 = { k4311 = [ { k43111 = 0 }, { k43112 = [2,3]} #comments here\n"
        "           ] } # more comments\n"
        "                 k432 = [ [\\=,\\,],\\[,\\}] }\n"
        "          \n"
        "    # \n"
        "          k44 = \\ abc\\\\d\\  \n"
        "        }]\n"
        "\n"
        "#k1 = [ v1 ]\n"
        "k5 = {\n"
        "  #aaa\n"
        "a=b\n"
        "  #bbb\n"
        "  #ccc\n"
        "}\n"
        "#k1 = [ v1 ]\n"
        "#kk;";
    const char* cfg = "/tmp/cfgtest.cfg";

    FILE* fp = fopen(cfg, "w");
    fprintf(fp, "%s", cfgstr);
    fclose(fp);

    utils::ConfigureReader r(cfg);
    fp = fopen(cfg, "w");
    fprintf(fp, "%s", r.toString().c_str());
    fclose(fp);

    utils::ConfigureReader r2(cfg);
    EXPECT_EQ(r, r2);

    // do some query
    EXPECT_STREQ(r2.get<std::string>("k1").c_str(), "v1");
    EXPECT_TRUE(r2.get<long long>("k2[1]") == (long long)3);
    EXPECT_DOUBLE_EQ(r2.get<double>("k3.k31[0]"), 2.1);
    EXPECT_TRUE(r2.get<int>("k4[0].k41[1].k412[2]") ==  3); 
    EXPECT_STREQ(r2.get<std::string>("k4[1].k43.k432[ 0][ 1]").c_str(), ",");

    const auto v = r2.getReader("k4[1].k43");
    const auto ks = v.listKeys();
    EXPECT_STREQ(ks[0].c_str(), "k431");
    EXPECT_STREQ(ks[1].c_str(), "k432");
    EXPECT_EQ(v.getReader("k431.k4311").arraySize(), 2);

    auto arr = v.getArr<int>("k431.k4311[1].k43112");
    std::vector<int> arr2 = {2, 3};
    EXPECT_EQ(arr, arr2);

    // get non-exist keys
    try {
        r2.get<int>("k1");
        EXPECT_TRUE(false);
    } catch (const std::exception & e) {
    }
    try {
        r2.get<int>("Q1");
        EXPECT_TRUE(false);
    } catch (const std::exception & e) {
        bool found = true;
        EXPECT_EQ(r2.get<long long>("k2[2]", &found, 1LL), 1LL);
        EXPECT_FALSE(found);
    }

    // set some values 
    std::vector<std::string> arr3 = {" { = ] ", " \\\nmore \n bad char, "};
    r2.set("k3.k31[1].k313[1]", 1);
    r2.set("k3.k31[1].k313[0]", 0);
    r2.setArr("k3.k31[ 2 ][0]", arr3);

    // set some values with escaped values
    fp = fopen(cfg, "w");
    fprintf(fp, "%s", r2.toString().c_str());
    fclose(fp);
    utils::ConfigureReader r3(cfg);
    EXPECT_EQ(r2, r3);

    // test json
    const char* cfg_json = 
    "{\n"
    "\"k1\" : \"v1\",\n"
    "\"k2\" : [ 2, 3 ],\n"
    "\"k3\" : { \n"
    "       \"k31\" : [2.1], \n"
    "       \"k32\" : false  \n"
    "         },\n"
    "\"k4\" : [ { \"k41\" : [ { \"k411\" : \"val\\\"411\\\"\"},{\"k412\":[1,2,3]}]\n,\n \n"
    "             \"k42\" : 0.2, \"k42_bad\": true,\n},"
    "           { \"k43\" : { \"k431\" : { \"k4311\" : [ { \"k43111\" : 0 }, { \"k43112\" : [2,3]} \n"
    "           ] }, \n"
    "                \"k432\" : [ [\"=\", \":,{}[]\n\\\"\"],\"[\", \"\\\\\"]}\n"
    "          \n, \"k43_bad\":true},"
    "           {\"k44\" : \"abc\\\\d  \n\"}, {\"all_good\": false}"
    "        ],\n"
    "\n"
    "\"k5\" : {\n\"empty_value\":\"\",     \"a\":\"\'b \""
    "}\n}";

    fp = fopen(cfg, "w");
    fprintf(fp, "%s", cfg_json);
    fclose(fp);
    auto jr = utils::ConfigureReader::getJson(cfg);

    fp = fopen(cfg, "w");
    fprintf(fp, "%s", jr->toString().c_str());
    fclose(fp);
    auto jr2 = utils::ConfigureReader::getJson(cfg);

    EXPECT_EQ(*jr, *jr2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

