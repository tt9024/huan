#include <cstdarg>
#include <string>
#include <cstring>

#include "time_util.h"
#include "rate_limiter.h"

#define MAX_LOG_ENTRY 1024*4
#define MAX_LOG_PER_SECOND 100

/*
 * TODO Add Line Number and File
 */
namespace utils {
  enum LogLevel {
    Error = 0,
    Warning = 1,
    Info = 2,
    Trace = 3,
    Debug = 4,

    TotalLevels
  };

  inline
  static const char* getLevelStr(LogLevel level) {
    switch (level) {
    case Error: return "ERR";
    case Warning: return "WAR";
    case Info: return "INF";
    case Trace: return "TRC";
    case Debug: return "DBG";
    default:
        return "???";
    }
  }

  class Logger {
  public:
    void logInfo(const char* file, int line, const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        log(Info, file, line, fmt, ap);
    }

    void logInfo(const char* file, int line, const std::string& str) {
        logInfo(file, line, "%s", str.c_str());
    }

    void logDebug(const char* file, int line, const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        log(Debug, file, line, fmt, ap);
    }

    void logError(const char* file, int line, const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        log(Error, file, line, fmt, ap);
    }

    void logError(const char* file, int line, const std::string& str) {
        logError(file, line, "%s", str.c_str());
    }

    virtual void flush() = 0;
    Logger()
#ifndef NUMPY_COMPILE
    :rl(MAX_LOG_PER_SECOND,1)
#endif
    {}
    virtual ~Logger() {
    };

  protected:
    void log(LogLevel level, const char* file, int line, const char* fmt, va_list ap) {
#ifndef NUMPY_COMPILE
      if (__builtin_expect(rl.check() != 0,0)) {
          return;
      }
#endif
      char char_buffer[MAX_LOG_ENTRY];
      int len = Logger::prepare_log_string(level, file, line, char_buffer, MAX_LOG_ENTRY-1);
      int len2 = vsnprintf(char_buffer+len, MAX_LOG_ENTRY-len, fmt, ap);
      if (__builtin_expect(len2 >= MAX_LOG_ENTRY-len,0)) {
          strcpy(char_buffer+MAX_LOG_ENTRY-11, "<CROPPED!>");
          len = MAX_LOG_ENTRY-1;
      } else {
          len += len2;
      }
      char_buffer[len++] = '\n';
      writeLog(level, char_buffer, len);
    }

    void log_string(LogLevel level, const char* str, int str_len) {
       writeLog(level, str, str_len);
    };

    static int prepare_log_string(LogLevel level, const char* file, int line, char* char_buffer, int buf_size) {
      int len = TimeUtil::frac_UTC_to_string(0, char_buffer, buf_size, 3);
      len += sprintf(char_buffer+len, ",%s,%s:%d,", getLevelStr(level),file,line);
      return len;
    }

    virtual void writeLog(int level, const char* str, int size) = 0;

#ifndef NUMPY_COMPILE
    RateLimiter rl;
#endif

  };

  class FileLogger : public Logger {
  public:
    FileLogger(const char* filepath): fp(nullptr), fp_save(nullptr) {
        if (strncmp(filepath, "stdout", 6)==0) {
            fp = stdout;
        } else {
            fp = fopen(filepath, "at+");
        }
        if (!fp) {
            throw std::runtime_error("cannot open log file to write!");
        }
        fp_save = fp;
    }
    ~FileLogger() {
        // don't close stdout yet
        if (fp_save && (fp_save != stdout)) {
            fclose(fp_save);
        }
        fp_save = nullptr;
    }
    void flush() {
       fflush(fp);
    }

    void loggerStdoutON() {
        fp = stdout;
    }

    void loggerStdoutOFF() {
        fp = fp_save;
    }

  private:
    FILE* fp;
    FILE* fp_save;
    void writeLog(int /* level */ , const char* str, int size) {
        fwrite(str, 1, size, fp);
        flush();
    };
  };
};
