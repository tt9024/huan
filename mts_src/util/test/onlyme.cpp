#include "instance_limiter.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s check_only(0/1) inst_names... (space delimitered names)\n", argv[0]);
        return 0;
    }
    bool check_only = argv[1][0] == '1';
    auto& om (utils::OnlyMe::get());
    for (int i=2; i<argc; ++i) {
        om.add_name(argv[i]);
    }
    while (true) {
        bool yes;
        if (check_only) {
            yes = om.check_only();
        } else {
            yes = om.check();
        }
        printf("%s: %s!\n", check_only?"check only":"check+update", yes?"Only Me":"Someone Else");
        utils::TimeUtil::micro_sleep(500000);
    }
    return 0;
}
