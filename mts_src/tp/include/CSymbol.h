#ifndef CSYMBOL_HEADER

#define CSYMBOL_HEADER

#include <string>
#include <unordered_map>
#include "CCurrncy.h"
#include "CDateTime.h"
#include "CConfig.h"
#include "CMtsException.h"

namespace Mts {
    namespace Core {
        class CSymbol
        {
        public:
            enum AssetClass
            {
                UNKNOWN = 0, FX = 1, NONFX = 2
            };

        public:
            static bool load(const std::string &strXML);

            static const CSymbol &getSymbol(const std::string &strSymbol);

            static const CSymbol &getSymbol(unsigned int iSymbolID);

            static const CSymbol &getSymbolFromExchSymbol(const std::string &strExchSymbol);

            static const CSymbol &getSymbolFromExchContract(const std::string &strExchContract);

            static const CSymbol &getSymbolFromTTSecID(const std::string &strTTSecID);

            static const CSymbol &getSymbolFromTDSymbol(const std::string &strTDSymbol);

            static const std::string getExchContractByMTSSymbol(const std::string &sym);

            static const std::string getMTSSymbolByExchContract(const std::string &exch_contract);

            static std::vector<boost::shared_ptr<const CSymbol> > getGUISymbols();

            static std::vector<boost::shared_ptr<const CSymbol> > getSymbols(AssetClass iAssetClass);

                                      static std::vector<boost::shared_ptr<const CSymbol> > getSymbols();

                                      static bool isSymbol(const std::string &strSymbol);

                                      static bool isSymbol(unsigned int iSymbolID);

                                      static void clear();

                                      static bool isHomogenous(Mts::Core::CSymbol::AssetClass iAssetClass);

                                      static void mapTTSecID(unsigned int iSymbolID,
                                              const std::string &strTTSecID);

            CSymbol();

            CSymbol(unsigned int iSymbolID,
                    const std::string &strSymbol,
                    const CCurrncy &objBaseCcy,
                    const CCurrncy &objRefCcy,
                    double dPointValue,
                    double dTickSize,
                    AssetClass iAssetClass,
                    bool bTransmitToGUI,
                    const std::string &strExchSymbol,
                    const std::string &strExchange,
                    const std::string &strTTExchange,
                    const std::string &strSecurityType,
                    const std::string &strTTSymbol,
                    double dTTScaleMultiplier,
                    const std::string &strTDSymbol,
                    double dTDScaleMultiplier,
                    unsigned int iLotsPerMin,
                    unsigned int iTimeSliceSecs,
                    unsigned int iDiscVal,
                    unsigned int iVariance,
                    unsigned int iLimitTicksAway,
                    const std::string &strContractExchSymbol,
                    const std::string &strContractMonth,
                    const std::string &strTTSecID);

            void setDefaults(const std::string &strDefContract,
                    const Mts::Core::CDateTime &dtDefContractExpiry,
                    const std::string strDefContractExchSymbol,
                    unsigned int iDefStartMin,
                    unsigned int iDefEndMin,
                    double dDefTradingStartDayFrac,
                    double dDefTradingEndDayFrac,
                    unsigned int iDefSettStartSec,
                    unsigned int iDefSettEndSec,
                    double dDefSettStartDayFrac,
                    double dDefSettEndDayFrac);

            unsigned int getSymbolID() const;

            std::string getSymbol() const;

            const CCurrncy &getBaseCcy() const;

            const CCurrncy &getRefCcy() const;

            double getPointValue() const;

            double getTickSize() const;

            AssetClass getAssetClass() const;

            bool transmitToGUI() const;

            std::string getExchSymbol() const;

            std::string getExchange() const;

            std::string getTTExchange() const;

            std::string getSecurityType() const;

            std::tuple<std::string, std::string> getCcy() const;

            std::string getTTSymbol() const;

            double getTTScaleMultiplier() const;

            std::string getTDSymbol() const;

            double getTDScaleMultiplier() const;

            bool isWithinTradingWindow(const Mts::Core::CDateTime &dtNow) const;

            bool isWithinSettWindow(const Mts::Core::CDateTime &dtNow) const;

            std::string getContractExchSymbol() const;

            std::string getContractTicker() const;

            std::string getContractMonth() const;

            // these two are used after the security definition matched
            void setTTSecID(const std::string &strTTSecID);

            std::string getTTSecID() const;

            // TWAP
            unsigned int getLotsPerMin() const;

            unsigned int getTimeSliceSecs() const;

            // Iceberg
            unsigned int getDiscVal() const;

            unsigned int getVariance() const;

            unsigned int getLimitTicksAway() const;

            std::string toString() const;

        private:
            using SymbolMapByName = std::unordered_map<std::string, boost::shared_ptr<CSymbol> >;
            using SymbolMapByID = std::unordered_map<unsigned int, boost::shared_ptr<CSymbol> >;

            // symbols stored in duplicate with different indices for faster lookup
            static SymbolMapByName m_SymbolsByName;
            static SymbolMapByID m_SymbolsByID;
            static SymbolMapByName m_SymbolsByExchContract;

            // internal MTS symbology
            unsigned int m_iSymbolID;
            std::string m_strSymbol;

            // contract specification
            CCurrncy m_objBaseCcy;
            CCurrncy m_objRefCcy;
            double m_dPointValue;  // pnl = point value * price delta
            double m_dTickSize;        // default pip/tick size, may be overridden per provider
            AssetClass m_iAssetClass;

            bool m_bTransmitToGUI;

            // exchange symbolology
            std::string m_strExchSymbol;
            std::string m_strExchange;
            std::string m_strTTExchange;
            std::string m_strSecurityType;

            // TT symbolology
            std::string m_strTTSymbol;
            double m_dTTScaleMultiplier;
            std::string m_strTTSecID;
            std::string m_strContractExchSymbol;
            std::string m_strContractTicker;
            std::string m_strContractMonth;

            // tickdata.com symbolology
            std::string m_strTDSymbol;
            double m_dTDScaleMultiplier;

            // TWAP parameters
            unsigned int m_iLotsPerMin;
            unsigned int m_iTimeSliceSecs;

            // Iceberg parameters
            unsigned int m_iDiscVal;
            unsigned int m_iVariance;
            unsigned int m_iLimitTicksAway;

        };
    }
}

#include "CSymbol.hxx"

#endif


