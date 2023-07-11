#ifndef CSQLALGORITHM_HEADER

#define CSQLALGORITHM_HEADER

#include <string>

namespace Mts {
    namespace SQL {
        class CSQLAlgorithm
        {
        public:
            CSQLAlgorithm(unsigned int iAlgoID,
                    const std::string &strAlgoName)
                    : m_iAlgoID(iAlgoID),
                      m_strAlgoName(strAlgoName) {}

            unsigned int getAlgoID() const { return m_iAlgoID; }

            const std::string &getAlgoName() const { return m_strAlgoName; }

        private:
            unsigned int m_iAlgoID;
            std::string m_strAlgoName;
        };
    }
}

#endif

