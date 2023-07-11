#ifndef CAPPLICATION_HEADER

#define CAPPLICATION_HEADER

#include <string>
#include <memory>
#include "CEngine.h"

namespace Mts {
    namespace Application {
        class CApplication
        {
        public:
            explicit CApplication(const std::string &VersionString);

            ~CApplication();

            void run(const std::string &strEnginePath);

            void stop();

            void addAlgo(const std::string &algoName);

        private:
            const std::string m_version_string;

            void run_(const std::string &strEnginePath);

            std::shared_ptr<Mts::Engine::CEngine> m_ptrEngine;
            volatile bool m_should_run;
        };
    }
}

#endif

