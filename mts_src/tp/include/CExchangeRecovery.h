#pragma once

#include <vector>
#include <map>
#include <tuple>

#include "ExecutionReport.h"

namespace Mts
{
    namespace Exchange
    {
        class CExchangeRecovery 
        {
        public:
            CExchangeRecovery();
            void init(const std::string& startLocalTime, const std::string& endLocalTime);

            void addExecutionReport(const std::shared_ptr<pm::ExecutionReport>& er);
            std::string startUTC() const { return m_sStartUTC; };
            std::string endUTC() const { return m_sEndUTC; };

            bool persistToFile(const std::string& filename) const;
            bool retrieveFromFile(const std::string& filename) { return false; };

        protected:
            // in UTC Format, i.e. YYYYDDMM-HH:MM:DD GMT 
            std::string m_sStartUTC;
            std::string m_sEndUTC;
            std::vector< std::shared_ptr<pm::ExecutionReport> > m_er;
        };
    };
};
