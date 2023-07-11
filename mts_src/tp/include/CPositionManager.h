#ifndef CPOSITIONMANAGER_HEADER

#define CPOSITIONMANAGER_HEADER

#include <unordered_map>
#include "CPosition.h"
#include "CBidAsk.h"
#include "CConfig.h"

namespace Mts {
    namespace Accounting {
        class CPositionManager
        {
        public:
            // construction
            CPositionManager();

            CPositionManager(const std::string &strName);

            // accessors
            const std::string &getName() const;

            const CPosition &getPosition(unsigned int iSymbolID) const;

            const CPosition &getPosition(const Mts::Core::CSymbol &objSymbol) const;

            void setPosition(
                    const Mts::Core::CSymbol &objSymbol,
                    const CPosition &objPosition
            );

            void printCurrentPositions();

            // operational
            void updatePnL(const Mts::OrderBook::CBidAsk &objBidAsk);

            double getTotalUSDPnL() const;

            void updatePosition(const Mts::Order::COrderFill &objFill);

            void updatePosition(
                    const Mts::Core::CSymbol &objSymbol,
                    const Mts::Order::COrder::BuySell iDirection,
                    unsigned int iQuantity,
                    double dPrice
            );

            unsigned int getNumFills(unsigned int iSymbolID) const;

            std::vector<unsigned int> getTradedInstruments() const;

            std::vector<std::pair<Mts::Core::CDateTime, double>> getDailyPnL() const;

            std::vector<std::pair<Mts::Core::CDateTime, unsigned int>> getDailyBuyCount() const;

            std::vector<std::pair<Mts::Core::CDateTime, unsigned int>> getDailySellCount() const;

        private:
            void initialize();

        private:
            using Symbol2PositionMap = std::unordered_map<unsigned int, Mts::Accounting::CPosition>;
            using Symbol2FillsMap = std::unordered_map<unsigned int, std::vector<Mts::Order::COrderFill>>;
            using DailyPNLArray = std::vector<std::pair<Mts::Core::CDateTime, double>>;
            using DailyCountArray = std::vector<std::pair<Mts::Core::CDateTime, unsigned int>>;

            // identifer for reporting
            std::string m_strName;

            // symbol ID -> position
            Symbol2PositionMap m_SymbolPositions;

            // symbol ID -> chronological list of fills
            Symbol2FillsMap m_SymbolFills;

            // end of day PnL for this model
            DailyPNLArray m_DailyPnL;

            // daily count of buys and sells
            DailyCountArray m_DailyBuyCount;
            DailyCountArray m_DailySellCount;

            // timestamp of last update (market data or fill)
            Mts::Core::CDateTime m_dtLastUpdate;

            // trade count (reset daily)
            unsigned int m_iNumBuys;
            unsigned int m_iNumSells;
        };
    }
}

#endif

