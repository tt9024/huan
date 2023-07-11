#ifndef CORDERFILL_HEADER

#define CORDERFILL_HEADER

#include "COrder.h"
#include "CDateTime.h"
#include "CEvent.h"

namespace Mts {
    namespace Order {
        class COrderFill : public Mts::Event::CEvent
        {
        public:
            COrderFill();

            COrderFill(const COrder &objOrder,
                    const Mts::Core::CDateTime &dtFillTimestamp,
                    unsigned int uiFilledQuantity,
                    double dFilledPrice,
                    const Mts::Core::CDateTime &dtSettDate);

            const COrder &getOrigOrder() const;

            void setOrigOrder(const COrder &objOrigOrder);

            Mts::Core::CDateTime getFillTimestamp() const;

            void setFillTimestamp(Mts::Core::CDateTime &dtFillTimestamp);

            Mts::Order::COrder::BuySell getBuySell() const;

            unsigned int getFillQuantity() const;

            void setFillQuantity(unsigned int uiFillQuantity);

            double getFillPrice() const;

            void setFillPrice(double dFillPrice);

            Mts::Core::CDateTime getSettDate() const;

            void setSettDate(Mts::Core::CDateTime &dtSettDate);

            void createBroadcastMessage(char *pszBuffer) const;

            std::string toString() const;

            bool isRecoveryFill() const;

            void setRecoveryFill(bool bRecoveryFill);

        private:
            COrder m_OrigOrder;
            Mts::Core::CDateTime m_dtFillTimestamp;
            unsigned int m_uiFillQuantity;
            double m_dFillPrice;
            Mts::Core::CDateTime m_dtSettDate;
            bool m_bRecoveryFill;
        };
    }
}

#endif

