#ifndef CORDER_HEADER

#define CORDER_HEADER

#include <cstring>
#include "CDateTime.h"
#include "CSymbol.h"
#include "CProvider.h"

namespace Mts {
    namespace Order {
        class COrder
        {
        public:
            enum OrderType
            {
                IOC, GTC
            };

            enum BuySell
            {
                BUY, SELL
            };

            enum OrderState
            {
                PENDING_NEW,
                NEW,
                CANCEL_PENDING,
                REPLACE_PENDING,
                REPLACED,
                PARTIALLY_FILLED,
                FILLED,
                CANCELLED,
                REJECTED,
                LIMIT_VIOLATION
            };

        public:
            COrder();

            COrder(
                    unsigned int m_iOriginatorAlgoID,
                    const char *szMtsOrderID,
                    const Mts::Core::CDateTime &dtCreateTimestamp,
                    unsigned int iSymbolID,
                    unsigned int iProviderID,
                    BuySell iDirection,
                    unsigned int uiQuantity,
                    OrderType iOrderType,
                    double dPrice,
                    const char *pszExecBrokerCode
            );

            unsigned int getOriginatorAlgoID() const;

            void setOriginatorAlgoID(unsigned int iOriginatorAlgoID);

            const char *getMtsOrderID() const;

            void setMtsOrderID(const char *pszMtsOrderID);

            const char *getExcOrderID() const;

            void setExcOrderID(const char *pszExcOrderID);

            const char *getExcExecID() const;

            void setExcExecID(const char *pszExcExecID);

            const char *getOrderTag() const;

            void setOrderTag(const char *pszOrderTag);

            Mts::Core::CDateTime getCreateTimestamp() const;

            void setCreateTimestamp(const Mts::Core::CDateTime &dtCreateTimestamp);

            Mts::Core::CDateTime getMtsTimestamp() const;

            void setMtsTimestamp(const Mts::Core::CDateTime &dtMtsTimestamp);

            Mts::Core::CDateTime getExcTimestamp() const;

            void setExcTimestamp(const Mts::Core::CDateTime &dtExcTimestamp);

            Mts::Core::CDateTime getLastFillTimestamp() const;

            void setLastFillTimestamp(const Mts::Core::CDateTime &dtLastFillTimestamp);

            unsigned int getSymbolID() const;

            void setSymbolID(unsigned int iSymbolID);

            unsigned int getProviderID() const;

            void setProviderID(unsigned int iProviderID);

            std::string getDirectionString() const;

            Mts::Order::COrder::BuySell getDirection() const;

            void setDirection(Mts::Order::COrder::BuySell iDirection);

            unsigned int getQuantity() const;

            void setQuantity(unsigned int uiQuantity);

            double getPrice() const;

            void setPrice(double dPrice);

            std::string getOrderTypeString() const;

            OrderType getOrderType() const;

            void setOrderType(OrderType iOrderType);

            OrderState getOrderState() const;

            void setOrderState(OrderState iOrderState);

            const char *getExecBrokerCode() const;

            void setExecBrokerCode(const char *pszExecBrokerCode);

            unsigned int getTotalFilledQty() const;

            void updateTotalFilledQty(unsigned int iFillQty);

            std::string getOrderStateDescription() const;

            std::string getDescription() const;

            std::string toString() const;

            static std::string orderstate2String(OrderState iOrdState);

        private:
            Mts::Core::CDateTime m_dtCreateTimestamp;
            Mts::Core::CDateTime m_dtMtsTimestamp;
            Mts::Core::CDateTime m_dtExcTimestamp;
            Mts::Core::CDateTime m_dtLastFillTimestamp;

            char m_szMtsOrderID[32];
            char m_szExcOrderID[64];
            char m_szExcExecID[64];
            char m_szOrderTag[64];                        // free text field
            char m_szExecBrokerCode[16];
            unsigned int m_iOriginatorAlgoID;
            unsigned int m_uiTotalFilledQty;
            unsigned int m_iSymbolID;
            unsigned int m_iProviderID;
            BuySell m_iDirection;
            unsigned int m_uiQuantity;
            double m_dPrice;
            OrderType m_iOrderType;
            OrderState m_iOrderState;
        };
    }
}

#include "COrder.hxx"

#endif

