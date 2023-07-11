#include "flr.h"
#include "async_io.h"

static const char* Version = "1.1";

int main(int argc, char**argv) {
    pm::FloorClientUser fcu ("user");
    if (argc == 1) {
        // an interative user interface
        pm::FloorClientUser fcu ("user");
        std::cout << "Floor Client Version "<< Version << std::endl << "Doesn't support Delete and Arrow keys, use CTRL-C instead."<< std::endl;
        std::string resp_;
        char cmd[MAX_LINE_LEN+1];
        size_t cmd_len;
        int resp_cnt = 1;
        std::cout << ">>> " << std::flush ;
        while (true) {
            if (utils::getline_nb(cmd, sizeof(cmd), &cmd_len)) {
                if (cmd_len > 1) {
                    std::cout << fcu.sendReq(std::string(cmd)) << std::endl;
                    resp_cnt = 1;
                }
                std::cout << ">>> " << std::flush;
            }
            utils::TimeUtil::micro_sleep(100*1000);
            resp_ = "";
            if (fcu.checkPrevResp(resp_)) {
                resp_cnt++;
                std::cout << std::endl << "*** the " << resp_cnt << " response ***" << std::endl << resp_ << std::endl << std::flush;
            }
        }
    } else {
        // an one-time command, 
        // could have multiple space delimitered parts
        std::string line (argv[1]);
        for (int i = 2; i < argc; ++i) {
            line += " ";
            line += std::string(argv[i]);
        }
        std::istringstream iss(line);
        std::string cmd;
        std::getline(iss >> std::ws, cmd);
        std::cout << fcu.sendReq(cmd.c_str()) << std::endl;
    }
    return 0;
}
