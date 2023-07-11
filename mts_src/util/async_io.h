/* Linux (POSIX) implementation of _kbhit(), with extension to
 * non-blocking get line
 */

#include <stdio.h>
#include <sys/select.h>
#include <termios.h>
//#include <stropts.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <cstring>

#define MAX_LINE_LEN 2048
namespace utils {
    int _kbhit() {
        static const int STDIN = 0;
        static bool initialized = false;
        if (! initialized) {
            // use termios to turn off line buffering
            termios term;
            tcgetattr(STDIN, &term);
            term.c_lflag &= ~ICANON;
            term.c_lflag |= ECHO;


            tcsetattr(STDIN, TCSANOW, &term);
            setbuf(stdin, NULL);
            initialized = true;
        }
        int bytesWaiting;
        ioctl(STDIN, FIONREAD, &bytesWaiting);
        return bytesWaiting;
    }

    bool getline_nb(char*line_buf, const size_t line_buf_len, size_t* line_len) {
        // get a line input, copy to the buffer upto the buflen
        static char buf[MAX_LINE_LEN+1];
        static int char_cnt = 0;
        int ready_cnt = _kbhit();
        while (ready_cnt > 0) {
            char ch = getchar();
            --ready_cnt;
            buf[char_cnt++] = ch;
            if (ch == '\n') {
                int cnt = char_cnt;
                *line_len = char_cnt;
                if ((int)cnt > (int)line_buf_len-1) {
                    cnt = line_buf_len-1;
                }
                memcpy(line_buf, buf, cnt);
                line_buf[cnt] = 0;
                char_cnt = 0;
                return true;
            }
        }
        return false;
    }
}
