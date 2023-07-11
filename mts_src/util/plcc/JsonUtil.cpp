#include "ConfigureReader.hpp"
#include "csv_util.h"
#include <iostream>
#include <string.h>

// This is a command line wrapper for the C++ type of utils::ConfigureReader
// to make it interative. 

void print_usage(const char* cmd) {
    printf("Usage: %s action file key value\n"
           "    action: help/check/get/getArr/count/list/set/setArr \n"
           "        get/getArr: get a value given the key/type\n"
           "        count: count number of elements given key (for an array type)\n"
           "        list: list sub-keys given key (for an dict type)\n"
           "        set/setArr: set a value given key/type\n"
           "    file: path to the configuration file\n"
           "    key/value: optional when needed\n"
           "        key: can be specified with dot(.) or squre brackets ([])\n"
           "        value: array values in double quotes and comma separated\n"
           "Example:\n"
           "to get a value given a key: \n"
           "    get models.ES01.parameters[0]\n"
           "    0.6\n"
           "\n"
           "to get an array given a key: \n"
           "    getArr models.ES01.parameters\n"
           "    0.6, 0.12, -1.2\n"
           "\n"
           "    count models.ES01.parameters\n"
           "    3\n"
           "\n"
           "to list sub keys given a key:\n"
           "    list models\n"
           "    ES01, CL01, IDBO_TF\n"
           "    ** Note set key to be '' lists root level keys\n"
           "\n"
           "to set array value given a key:\n"
           "    setArr models.ES01.subscriptions \"ES, HG, GC\"\n"
           "    the updated config is written back to the given file\n"
           , cmd);
}

void printVec(const std::vector<std::string>& vec) {
    if (vec.size() > 0) {
        std::cout << vec[0];
    }
    for (size_t i=1; i<vec.size(); ++i) {
        std::cout << ", " << vec[i];
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    const char* action = argv[1];
    if (strcmp(action, "help")==0) {
        print_usage(argv[0]);
        return 0;
    }

    if (strcmp(action, "check")==0) {
        const char* file = argv[2];
        const auto cr = utils::ConfigureReader::getJson(file);
        std::cout << "OK" << std::endl;
        std::cout << cr->toString() << std::endl;
        return 0;
    }

    if ((strcmp(action, "get")==0) ||
        (strcmp(action, "getArr")==0) ||
        (strcmp(action, "count")==0) ||
        (strcmp(action, "list")==0) ||
        (strcmp(action, "set") == 0) ||
        (strcmp(action, "setArr")==0))
    {
        const char* file = argv[2];
        const char* key = argv[3];
        auto& cr = *utils::ConfigureReader::getJson(file);

        if (strcmp(action, "get")==0) {
            std::cout << cr.get<std::string>(key) << std::endl;
        } else if (strcmp(action, "getArr")==0) {
            printVec(cr.getArr<std::string>(key));
        } else if (strcmp(action, "count")==0) {
            std::cout << cr.getReader(key).arraySize() << std::endl;
        } else if (strcmp(action, "list")==0) {
            printVec(cr.getReader(key).listKeys());
        } else {
            const char* val = argv[4];
            if (strcmp(action, "set")==0) {
                cr.set<std::string>(key,std::string(val));
            } else if (strcmp(action, "setArr")==0) {
                // vals is a comma delimited csv line
                // i.e. 1, 2
                cr.setArr<std::string>(key, utils::CSVUtil::read_line(std::string(val)));
            }
            cr.toFile(file);
        }
    } else {
        throw std::runtime_error(std::string("unknown action ") + action);
    }
    return 0;
}

