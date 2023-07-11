#include "CExchangeRecovery.h"
#include "CApplicationLog.h"

namespace Mts {
    namespace Exchange {
        // TODO- this should be a local struct of CDropCopyFIX_TT
        CExchangeRecovery::CExchangeRecovery() {};

        void CExchangeRecovery::init(const std::string &startLocalTime, const std::string &endLocalTime) {
            // convert to utc
            m_sStartUTC = utils::TimeUtil::string_local_to_gmt(startLocalTime.c_str());
            m_sEndUTC = utils::TimeUtil::string_local_to_gmt(endLocalTime.c_str());
            m_er.clear();
        }

        void CExchangeRecovery::addExecutionReport(const std::shared_ptr<pm::ExecutionReport> &er) {
            m_er.push_back(er);
        }

        bool CExchangeRecovery::persistToFile(const std::string &filename) const {
            for (const auto &er: m_er) {
                utils::CSVUtil::write_line_to_file(er->toCSVLine(), filename, true);
            }
            return true;
        }
    }
}
