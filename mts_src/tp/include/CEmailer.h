#ifndef CEMAILER_HEADER

#define CEMAILER_HEADER

#include <vector>
#include <string>

namespace Mts {
    namespace Mail {
        class CEmailer
        {
        public:
            void check(int iStatus, char *szFunction) const;

            bool sendEmailStd(const std::string &strToEmailAddr,
                    const std::string &strSubject,
                    const std::string &strBody) const;

            bool sendEmail(const std::string &strToEmailAddr,
                    const std::string &strSubject,
                    const std::string &strMsgLine) const;

            bool sendEmail(const std::string &strSubject,
                    const std::vector<std::string> &strMsgLine) const;

            bool sendEmail(const std::string &strSubject,
                    const std::string &strMsgLine) const;

            bool sendEmail(const std::string &strSmtpServerName,
                    const std::string &strToAddr,
                    const std::string &strFromAddr,
                    const std::string &strSubject,
                    const std::vector<std::string> &strMsgLine) const;
        };
    }
}

#endif


