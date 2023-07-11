#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace utils {
    class CSVUtil {
    public:
        using LineTokens = std::vector<std::string>;
        using FileTokens = std::vector<LineTokens>;
        static const char Delimiter = ',';

        static LineTokens read_line(const std::string& line, char delimiter = Delimiter) {
            LineTokens vec;
            if (line.size()==0) 
                return vec;

            std::string line0 = line;
            if (line0.back()==delimiter) {
                //in case last column skipped
                line0.append(1, ' ');
            }
            std::istringstream iss(line0);
            std::string token;
            while (getline(iss, token, delimiter)) {
                vec.push_back(trim(token));
            }
            return vec;
        }

        static FileTokens read_file(const std::string& csv_file, char delimiter = Delimiter, int /* skip_head_lines */ = 0) {
            std::string line;
            FileTokens vec;
            try {
                std::ifstream csv(csv_file);
                while (std::getline(csv, line)) {
                    const auto v = read_line(line, delimiter);
                    if (v.size()>0) {
                        vec.push_back(v);
                    }
                }
            } catch(const std::exception& e) {
                fprintf(stderr, "Error getting csv file %s: %s\n", csv_file.c_str(), e.what());
            }
            return vec;
        }

        template<typename OStream>
        static void write_line(const LineTokens& token_vec, OStream& csvfile, char delimiter = Delimiter) {
            if (token_vec.size()==0) 
                return;
            csvfile << token_vec[0];
            for (size_t i=1; i<token_vec.size();++i) {
                csvfile << delimiter << token_vec[i];
            }
            csvfile << std::endl;
        }

        static bool write_file(const FileTokens& line_vec, const std::string& filename, bool append=true,  char delimiter=Delimiter) {
            std::ofstream csvfile;
            try {
                if (append) 
                    csvfile.open(filename, std::ios_base::app);
                else {
                    csvfile.open(filename, std::ios_base::trunc);
                }
                 
                for (const auto& line : line_vec) {
                     write_line(line, csvfile, delimiter);
                }
                return true;
            } catch (const std::exception& e) {
                fprintf(stderr, "csv file %s write failed: %s\n", filename.c_str(), e.what());
            } catch (...) {
                fprintf(stderr, "csv file %s write failed for unknown reason\n", filename.c_str());
            };
            return false;
        };

        static bool write_line_to_file(const LineTokens& token_vec, const std::string& filename, bool append=true,  char delimiter=Delimiter) {
            FileTokens ft;
            ft.push_back(token_vec);
            return write_file(ft, filename, append, delimiter);
        };

        static bool write_file(const LineTokens& token_vec, FILE*fp) {
            if (!fp) return false;
            try {
                std::ostringstream csvoss;
                write_line(token_vec, csvoss);
                std::string str = csvoss.str();
                size_t sz = str.size();
                if (sz == 0) {
                    return true;
                }
                fwrite(str.c_str(), 1, sz,fp);
                fflush(fp);
                return true;
            } catch (const std::exception& e) {
                fprintf(stderr, "csv file write failed: %s\n", e.what());
            }
            return false;
        }

        static std::string write_line_to_string(const LineTokens& token_vec) {
            char buf[2048];
            buf[0] = 0;
            size_t len = token_vec.size(), cnt=0;
            if (len > 0) {
                cnt = snprintf(buf, sizeof(buf), "%s", token_vec[0].c_str());
            }
            for (size_t i=1; (i<len) && (cnt<sizeof(buf)-16) ; ++i) {
                cnt+= snprintf(buf+cnt, sizeof(buf)-cnt, ",%s", token_vec[i].c_str());
            }
            return std::string(buf);
        }

        // string utilities
        static std::string ltrim(const std::string& s) {
            static const std::string WhiteSpace = " \n\r\t\f\v";
            size_t start = s.find_first_not_of(WhiteSpace);
            return (start == std::string::npos) ? "" : s.substr(start);
        }

        static std::string rtrim(const std::string& s) {
            static const std::string WhiteSpace = " \n\r\t\f\v";
            size_t end = s.find_last_not_of(WhiteSpace);
            return (end == std::string::npos) ? "" : s.substr(0, end+1);
        }

        static std::string trim(const std::string& s) {
            return rtrim(ltrim(s));
        }

        static std::string printDouble(double d, int max_decimal) {
            // convert a double to string, with maximum number of decimals in fraction precision
            // note it rounds the last decimal if needed, similar as %g in printf
            // i.e. 
            // printDouble(-2.5678, 2) --> -2.57
            // printDouble(-2.5678, 6) --> -2.5678
            // printDouble(-2.5678, 0) --> -3.0
            // printDouble(0, 0)       --> 0

            if (__builtin_expect((max_decimal>20) || (max_decimal < 0),0)) {
                // 10**20 is almost 2**60, upto a long long to hold the fraction part 
                throw std::runtime_error(std::string("printDouble got max_decimal too high ") + std::to_string(max_decimal));
            }

            char strbuf[128];
            size_t cnt = 0;
            if (d < (double)0.0) {
                strbuf[cnt++] = '-';
                d = -d;
            }
            double mul10 = pow(10,max_decimal), intpart, fracpart;
            d = (double)((unsigned long long)(d*mul10 + 0.5))/mul10; // normalize d w.r.t. max_decimal
            fracpart = modf(d, &intpart);

            // write the sign and integer part
            cnt += snprintf(strbuf+cnt, sizeof(strbuf)-cnt, "%llu",  (unsigned long long) (intpart+0.5));
            if (max_decimal > 0) {
                // write the fraction upto max_decimal
                strbuf[cnt++]='.';
                unsigned long long fpart = (unsigned long long) (fracpart*mul10 + 0.5);
                if (fpart == 0) {
                    // no fraction, put "0" and done
                    strbuf[cnt++]='0';
                    strbuf[cnt++]=0;
                } else {
                    char* ptr = strbuf + (cnt+max_decimal);
                    *ptr--=0;
                    const char* ptr0 = strbuf+cnt;
                    // skipping trailing zeros
                    while (fpart%10==0) {
                        fpart/=10;
                        *ptr--=0;
                    };
                    while (ptr>=ptr0) {
                        *ptr--= (char)((fpart%10) + '0');
                        fpart/=10;
                    }
                }
            }
            return std::string(strbuf);
        }

        static std::string printPnl(double d) {
            auto s = printDouble(d, 0);
            // add a sign place in the front
            return d>=0? (std::string(" ")+s):s;
        }

        static std::string printPnlColor(double d) {
            static const std::string NORM = "\x1B[0m";
            static const std::string RED = "\x1B[31m";
            static const std::string GRN = "\x1B[32m";
            
            const std::string ret = printPnl(d) + NORM;
            if (d>=0) {
                return GRN+ret;
            }
            return RED+ret;
        }
    };
}
