#ifndef CCONSOLIDATEDPOSITIONMANAGER_HEADER

#define CCONSOLIDATEDPOSITIONMANAGER_HEADER

#include <boost/shared_ptr.hpp>
#include "CPositionManager.h"
#include "CAlgorithm.h"
#include "COrderFill.h"

// This is mainly used by GUI
namespace Mts {
    namespace Accounting {
        class CConsolidatedPositionManager
        {
        public:

            // contruction
            CConsolidatedPositionManager();

            // accessors
            const CPosition &getPosition(unsigned int iAlgoID,
                    unsigned int iSymbolID);

            double getCcyPosition(const Mts::Core::CCurrncy &objCcy) const;

            const CPosition &getPosition(boost::shared_ptr<CAlgorithm> ptrAlgorithm,
                    const Mts::Core::CSymbol &objSymbol);

            // operations
            void updatePosition(const Mts::Order::COrderFill &objFill);

        private:
            // each manager will manage all the positions for a particular algo
            CPositionManager m_PositionManagers[Mts::Core::CConfig::MAX_NUM_ALGOS];

            // risk broken down by ccy
            double m_CcyPosition[Mts::Core::CConfig::MAX_NUM_CCYS];
        };
    }
}

#endif

