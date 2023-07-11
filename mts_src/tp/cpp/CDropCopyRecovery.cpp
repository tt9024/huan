#include "CDropCopyRecovery.hxx"
#include "CApplicationLog.h"

namespace Mts {
    namespace Exchange {
        CDropCopyRecovery::CDropCopyRecovery(
            const std::string& dropcopy_fix_config,
            const std::string& dropcopy_session_config
        ) : m_sFixConfig(dropcopy_fix_config),
            m_sSessionConfig(dropcopy_session_config),
            m_pSession(CExchangeFactory::getInstance().createDropCopy(dropcopy_session_config))
        {}

        const CExchangeRecovery& CDropCopyRecovery::get(
            const std::string& startLocalTime,
            const std::string& endLocalTime
        ) {
            m_pSession->setRequest(startLocalTime, endLocalTime);
            try {
                auto fixEngine = Mts::FIXEngine::CFIXEngine(m_sFixConfig);
                fixEngine.addSession(m_pSession);
                AppLog(std::string("Starting DropCopy Fix Connection"));
                fixEngine.run();
                while (!m_pSession->isDone()) {
                    sleep_milli(200);
                }
                AppLog(std::string("Recovery received, waiting for disconnect."));
                fixEngine.stopFIXEngine();
                int waitSec = 5;
                while ((waitSec > 0) && (!fixEngine.isShutdown())) {
                    sleep_milli(1000);
                    --waitSec;
                }
                AppLog(std::string("Done!"));
            }
            catch (const std::exception& e) {
                AppLog(std::string("Error Getting Dropcopy Recovery: ") + std::string(e.what()));
            }
            return m_pSession->getRecoveryFills();
        }
        void CDropCopyRecovery::getToFile(
            const std::string& startLocalTime,
            const std::string& endLocalTime,
            const std::string& filename
        ) {
            get(startLocalTime, endLocalTime).persistToFile(filename);
        };
    }
}
