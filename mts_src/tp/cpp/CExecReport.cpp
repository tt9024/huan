#include <cstring>
#include "CExecReport.h"


using namespace Mts::Order;


CExecReport::CExecReport()
        : CEvent(Mts::Event::CEvent::EXEC_REPORT),
          m_iOriginatorAlgoID(0),
          m_iOrigQty(0),
          m_dtCreateTime(0),
          m_iLastFillQty(0),
          m_dLastFillPx(0.0),
          m_iTotalFillQty(0),
          m_dAvgFillPx(0.0),
          m_iRemainingQty(0),
          m_dtUpdateTime(0),
          m_iMtsOrderState(COrder::NEW) {

    strcpy(m_szParentOrderID, "");
    strcpy(m_szTicker, "");
    strcpy(m_szBuySell, "");
    strcpy(m_szAlgorithm, "");
    strcpy(m_szParameters, "");
    strcpy(m_szStatus, "");
    strcpy(m_szLastMkt, "");
}


CExecReport::CExecReport(unsigned int iOriginatorAlgoID,
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
        const Mts::Core::CDateTime &dtUpdateTime)
        : CEvent(Mts::Event::CEvent::EXEC_REPORT),
          m_iOriginatorAlgoID(iOriginatorAlgoID),
          m_iOrigQty(iOrigQty),
          m_dtCreateTime(dtCreateTime),
          m_iLastFillQty(iLastFillQty),
          m_dLastFillPx(dLastFillPx),
          m_iTotalFillQty(iTotalFillQty),
          m_dAvgFillPx(dAvgFillPx),
          m_iRemainingQty(iRemainingQty),
          m_iMtsOrderState(iMtsOrderState),
          m_dtUpdateTime(dtUpdateTime) {

    strcpy(m_szParentOrderID, pszParentOrderID);
    strcpy(m_szTicker, pszTicker);
    strcpy(m_szBuySell, pszBuySell);
    strcpy(m_szAlgorithm, pszAlgorithm);
    strcpy(m_szParameters, pszParameters);
    strcpy(m_szStatus, pszStatus);
    strcpy(m_szLastMkt, pszLastMkt);
}


std::string CExecReport::toString() const {
    char szBuffer[1024];
    snprintf(szBuffer, sizeof(szBuffer),
             "Oid=%s, Tick=%s, Algo=%s, Stat=%s, BuySell=%s, Mkt=%s, "
             "CreatTime=%s, OrigQty=%d, LastQty=%u, LastPx=%.7lf"
             "TotFillQty=%u, AvgFillPx=%.7lf, RemainQty=%u",
             "State=%s, UpdTime=%s",

             m_szParentOrderID,
             m_szTicker,
             m_szAlgorithm,
             m_szStatus,
             m_szBuySell,
             m_szLastMkt,

             m_dtCreateTime.toString().c_str(),
             m_iOrigQty,
             m_iLastFillQty,
             m_dLastFillPx,

             m_iTotalFillQty,
             m_dAvgFillPx,
             m_iRemainingQty,

             COrder::orderstate2String(m_iMtsOrderState).c_str(),
             m_dtUpdateTime.toString().c_str());

    return szBuffer;
}


