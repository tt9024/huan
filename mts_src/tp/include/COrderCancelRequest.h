#ifndef CORDERCANCELREQUEST_HEADER

#define CORDERCANCELREQUEST_HEADER

#include "COrder.h"
#include "CDateTime.h"

namespace Mts {
    namespace Order {
        class COrderCancelRequest
        {
        public:
            COrderCancelRequest();

            COrderCancelRequest(unsigned int iOriginatorAlgoID,
                    const char *pszMtsCancelReqID,
                    const Mts::Core::CDateTime &dtCreateTimestamp,
                    const COrder &objOrigOrder);

            const Mts::Core::CDateTime &getCreateTimestamp() const;

            void setCreateTimestamp(const Mts::Core::CDateTime &dtCreateTimestamp);

            const char *getMtsCancelReqID() const;

            void setMtsCancelReqID(const char *pszMtsCancelReqID);

            unsigned int getOriginatorAlgoID() const;

            void setOriginatorAlgoID(unsigned int iOriginatorAlgoID);

            const COrder &getOrigOrder() const;

            void setOrigOrder(const Mts::Order::COrder &objOrigOrder);

            void cancelAccepted();

            void cancelRejected();

            std::string toString() const;

            std::string getDescription() const;

        private:
            Mts::Core::CDateTime m_dtCreateTimestamp;
            char m_szMtsCancelReqID[32];
            unsigned int m_iOriginatorAlgoID;
            COrder m_OrigOrder;
        };
    }
}

#endif

