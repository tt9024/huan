#pragma once

#include "CDropCopyFIX_TT.h"
#include "CFIXEngine.h"
#include "CExchangeFactory.h"

namespace Mts
{
    namespace Exchange
    {
        class CDropCopyRecovery
        {
        public:
            CDropCopyRecovery(
                const std::string& dropcopy_fix_config,
                const std::string& dropcopy_session_config
            );

            const CExchangeRecovery& get(
                const std::string& startLocalTime,
                const std::string& endLocalTime
            );

            void getToFile(
                const std::string& startLocalTime,
                const std::string& endLocalTime,
                const std::string& filename
            );

            template<typename FillType, typename StatusType, typename PositionType>
            bool reconcile(
                const std::vector<FillType>& fills,
                const std::map<std::string, StatusType>& stats,
                const std::vector<PositionType>& positions
            ) const {
                return false;
            }

        protected:
            const std::string m_sFixConfig;
            const std::string m_sSessionConfig;
            boost::shared_ptr<CDropCopyFIX_TT> m_pSession;
        };
    }
};

