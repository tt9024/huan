#include "plcc/PLCC.hpp"
#include <iostream>


int main(int argc, char** argv) {
    if (argc == 3 && strcmp(argv[2], "json")==0) {
        std::cout << utils::ConfigureReader::getJson(argv[1])->toString() << std::endl;
        return 0;
    }

    utils::ConfigureReader cr(argv[1]);
    std::cout << cr.toString() << std::endl;
    return 0;
}
