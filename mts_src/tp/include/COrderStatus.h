#ifndef CORDERSTATUS_HEADER

#define CORDERSTATUS_HEADER

#include "COrder.h"
#include "CEvent.h"

namespace Mts {
    namespace Order {
        class COrderStatus : public Mts::Event::CEvent
        {
        public:
            COrderStatus();

            COrderStatus(const COrder &objOrder,
                    const Mts::Core::CDateTime &dtCreateTime);

            const COrder &getOrigOrder() const;

            void setOrigOrder(const COrder &objOrigOrder);

            Mts::Core::CDateTime getCreateTimestamp() const;

            void setCreateTimestamp(const Mts::Core::CDateTime &dtCreateTimestamp);

            COrder::OrderState getStatus() const;

            void setStatus(Mts::Order::COrder::OrderState iOrderState);

            std::string toString() const;

        private:
            COrder m_OrigOrder;
            Mts::Core::CDateTime m_dtCreateTimestamp;
        };
    }
}

#endif

