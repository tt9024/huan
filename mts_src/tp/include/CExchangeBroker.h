#ifndef CEXCHANGEBROKER_HEADER

#define CEXCHANGEBROKER_HEADER

#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>
#include "CExchange.h"
#include "CBidAsk.h"

namespace Mts {
    namespace Exchange {
        class CExchangeBroker
        {
        public:
            static CExchangeBroker &getInstance();

            CExchangeBroker() = default;

            CExchangeBroker(const CExchangeBroker &objRhs) = delete;

            void connectToAllExchanges();

            void disconnectFromAllExchanges();

            void connectToExchange(int iProviderID);

            void disconnectFromExchange(int iProviderID);

            void initiateAllExchanges();

            void feedTestExchanges(const Mts::OrderBook::CBidAsk &objBidAsk);

            void addExchange(int iProviderID,
                    boost::shared_ptr<CExchange> ptrExchange);

            boost::optional<std::reference_wrapper<CExchange> > getExchange(int iProviderID);

            void dump();

        private:
            // provider ID -> exchange
            typedef boost::unordered_map<int, boost::shared_ptr<CExchange> > ExchangeMap;

            ExchangeMap m_Exchanges;
        };
    }
}

#endif

