#ifndef CEXECREPORT_HEADER

#define CEXECREPORT_HEADER

#include <string>
#include "CDateTime.h"
#include "CEvent.h"
#include "COrder.h"


namespace Mts {
    namespace Order {
        class CExecReport : public Mts::Event::CEvent
        {
        public:
            CExecReport();

            CExecReport(unsigned int iOriginatorAlgoID,
                    const char *pszParentOrderID,
                    const char *pszTicker,
                    const char *pszBuySell,
                    unsigned int iOrigQty,
                    const Mts::Core::CDateTime &dtCreateTime,
                    const char *pszAlgorithm,
                    const char *pszParameters,
                    unsigned int iLastFillQty,
                    double dLastFillPx,
                    const char *pszLastMkt,
                    unsigned int iTotalFillQty,
                    double dAvgFillPx,
                    unsigned int iRemainingQty,
                    COrder::OrderState iMtsOrderState,
                    const char *pszStatus,
                    const Mts::Core::CDateTime &dtUpdateTime);

            unsigned int getOriginatorAlgoID() const;

            const char *getParentOrderID() const;

            const char *getTicker() const;

            const char *getBuySell() const;

            unsigned int getOrigQty() const;

            const Mts::Core::CDateTime &getCreateTime() const;

            const char *getAlgorithm() const;

            const char *getParameters() const;

            unsigned int getLastFillQty() const;

            double getLastFillPx() const;

            const char *getLastMkt() const;

            unsigned int getTotalFillQty() const;

            double getAvgFillPx() const;

            unsigned int getRemainingQty() const;

            COrder::OrderState getMtsOrderState() const;

            const char *getStatus() const;

            const Mts::Core::CDateTime &getUpdateTime() const;

            void setOriginatorAlgoID(unsigned int iOriginatorAlgoID);

            void setParentOrderID(const char *pszParentOrderID);

            void setTicker(const char *pszTicker);

            void setBuySell(const char *pszBuySell);

            void setOrigQty(unsigned int iOrigQty);

            void setCreateTime(const Mts::Core::CDateTime &dtCreateTime);

            void setAlgorithm(const char *pszAlgorithm);

            void setParameters(const char *pszParameters);

            void setLastFillQty(unsigned int iLastFillQty);

            void setLastFillPx(double dLastFillPx);

            void setLastMkt(const char *pszLastMkt);

            void setTotalFillQty(unsigned int iTotalFillQty);

            void setAvgFillPx(double dAvgFillPx);

            void setRemainingQty(unsigned int iRemainingQty);

            void setMtsOrderState(COrder::OrderState iMtsOrderState);

            void setStatus(const char *pszStatus);

            void setUpdateTime(const Mts::Core::CDateTime &dtUpdateTime);

            std::string toString() const;

        private:
            char m_szParentOrderID[32];
            char m_szTicker[16];
            char m_szAlgorithm[32];
            char m_szParameters[96];
            char m_szStatus[32];
            char m_szBuySell[16];
            char m_szLastMkt[16];
            unsigned int m_iOriginatorAlgoID;
            unsigned int m_iOrigQty;
            Mts::Core::CDateTime m_dtCreateTime;
            unsigned int m_iLastFillQty;
            double m_dLastFillPx;
            unsigned int m_iTotalFillQty;
            double m_dAvgFillPx;
            unsigned int m_iRemainingQty;
            COrder::OrderState m_iMtsOrderState;
            Mts::Core::CDateTime m_dtUpdateTime;
        };
    }
}

#include "CExecReport.hxx"

#endif

