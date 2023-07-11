#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include "CApplicationLog.h"

/*
Type Algo need to implement the following interface to
service the command. 

bool sendOrder(bool isBuy, const std::string& symbol,
    long long qty, const std::string& priceStr);
bool cancelOrder(const std::string & symbol);
bool getBidAsk(const std::string& symbol);
bool getPosition(const std::string& symbol);

*/

namespace Mts {
    namespace Algorithm {
        class CAlgorithmCommand
        {
        public :
            template<class AlgoType, class LoggerType>
            static bool runCommand(const std::string &command, AlgoType &algo, LoggerType &logger);

            // utilities for parsing price strings using state information from algo
            static bool parsePriceStr(const std::string priceStr,
                    double bidPx, double askPx, double tickSize,
                    double &price, bool &isMarketOrder);

            static std::string getHelpString();


        private:
            template<class AlgoType, class LoggerType>
            static bool runOrderCommand(const std::vector<std::string> &tokens, AlgoType &algo, LoggerType &logger);

            template<class AlgoType, class LoggerType>
            static bool runCancelCommand(const std::vector<std::string> &tokens, AlgoType &algo, LoggerType &logger);

            static bool parseOrderCommand(
                    const std::vector<std::string> &tokens,
                    bool &isBuy,
                    std::string &symbol,
                    long long &qty,
                    std::string &priceStr);

            static std::vector<std::string>
            splitString(const std::string &strDelimitedData, const std::string &strDelimiter);
        };
    };
};

namespace Mts {
    namespace Algorithm {

        template<class AlgoType, class LoggerType>
        bool CAlgorithmCommand::runCommand(const std::string &command, AlgoType &algo, LoggerType &logger) {
            LogInfo(logger, std::string("AlgorithmCommand Recieved: ") + command);
            std::vector<std::string> tokens;
            try {
                tokens = splitString(command, ",");
            }
            catch (std::exception &e) {
                LogError(logger, e.what());
            }
            if (tokens.size() == 0) {
                LogError(logger, "Recieved Empty Command!");
                return false;
            }
            const std::string &cmd(tokens[0]);
            bool bRet = false;
            if ((cmd == "B") || (cmd == "S")) {
                bRet = runOrderCommand(tokens, algo, logger);
            } else if (cmd == "C") {
                bRet = runCancelCommand(tokens, algo, logger);
            } else if (cmd == "O") {
                // list open orders
                std::string symbol;
                if (tokens.size() == 2) symbol = tokens[1];
                LogInfo(logger, algo.getOpenOrders(symbol));
            } else if (cmd == "P") {
                // get positions
                std::string symbol;
                if (tokens.size() == 2) symbol = tokens[1];
                LogInfo(logger, algo.getPosition(symbol));
            } else if (cmd == "D") {
                // show best bid/ask
                std::string symbol;
                if (tokens.size() == 2) symbol = tokens[1];
                LogInfo(logger, algo.getBidAsk(symbol));
            } else if (cmd == "H") {
                LogInfo(logger, std::string(getHelpString()));
            } else {
                LogError(logger,
                         std::string("Unknown command received: ") + command + std::string(" Enter \"H\" for help"));
            }
            return bRet;
        };

        bool CAlgorithmCommand::parsePriceStr(const std::string priceStr,
                double bidPx, double askPx, double tickSize,
                double &price, bool &isMarketOrder) {
            // take a format of either a floating point number or 
            // ['a'|'b']['+'|'-']X, where X is an integer in tick size or
            // 'm', suggests a market order, and price is not given 
            // returns true if parsing is successful with price an isMarketOrder populated

            double px = 0.0;
            const char *ps = priceStr.c_str();
            if (*ps == 'm') {
                isMarketOrder = true;
                price = 0.0;
                return true;
            }
            isMarketOrder = false;
            if (*ps == 'a' || *ps == 'b') {
                px = (*ps == 'b') ? bidPx : askPx;
                try {
                    // enforce a sign 
                    if (ps[1] == '+' || ps[1] == '-') {
                        price = px + (atoi(ps + 1) * tickSize);
                        return true;
                    }
                }
                catch (const std::exception &) {}
                return false;
            }
            try {
                price = atof(ps);
                return true;
            }
            catch (const std::exception &) {}
            return false;
        };

        bool CAlgorithmCommand::parseOrderCommand(
                const std::vector<std::string> &tokens,
                bool &isBuy,
                std::string &symbol,
                long long &qty,
                std::string &priceStr) {
            //[B|S] symbol qty priceStr
            if (tokens.size() == 4) {
                try {
                    isBuy = tokens[0] == "B" ? true : false;
                    symbol = tokens[1];
                    qty = atoll(tokens[2].c_str());
                    priceStr = tokens[3];
                    return true;
                }
                catch (const std::exception &) {};
            };
            return false;
        }

        std::string CAlgorithmCommand::getHelpString() {
            // help
            char helpstr[1024];
            size_t sz = snprintf(helpstr, sizeof(helpstr), "MTS Algo Commands:\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Buy or Sell: [B|S], symbol, qty, priceStr\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Cancel:      C, symbol\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Cancel All:  X\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Open Orders: O, symbol\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Positions:   P, symbol\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Dump Book:   D, symbol\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Help:        H\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "-----\n");
            sz += snprintf(helpstr + sz, sizeof(helpstr) - sz, "Note: the priceStr is given as a double for"\
                "limit order, or could be given as 'm' for market order.  "\
                "In particular, a string with form of \"['a'|'b']['+'|'-']cnt\" can be used to specify relative"\
                "tick count of the current best bid or ask.\n"\
                "The position string has the format of:\n"\
                "Symbol, LongQty, LongWap, ShortQty, ShortWap, LastPx, LastPnl\n");
            return std::string(helpstr);
        }

        template<class AlgoType, class LoggerType>
        bool
        CAlgorithmCommand::runOrderCommand(const std::vector<std::string> &tokens, AlgoType &algo, LoggerType &logger) {
            bool isBuy;
            std::string symbol;
            long long qty;
            std::string priceStr;
            if (!parseOrderCommand(tokens, isBuy, symbol, qty, priceStr)) {
                LogError(logger, "Failed to parse the order command!");
                return false;
            }
            if (!algo.sendOrder(isBuy, symbol, qty, priceStr)) {
                LogError(logger, "Failed to send order!");
                return false;
            }
            return true;
        }

        template<class AlgoType, class LoggerType>
        bool CAlgorithmCommand::runCancelCommand(const std::vector<std::string> &tokens, AlgoType &algo,
                LoggerType &logger) {
            if (tokens.size() != 2) {
                LogError(logger, "Invalid Cancel command!");
                return false;
            }
            if (!algo.cancelOrder(tokens[1])) {
                LogError(logger, "Failed to send cancel request!");
                return false;
            }
            return true;
        };

        inline std::vector<std::string>
        CAlgorithmCommand::splitString(const std::string &strDelimitedData, const std::string &strDelimiter) {
            std::istringstream ss(strDelimitedData);
            std::string strToken;
            std::vector<std::string> objTokens;

            while (std::getline(ss, strToken, strDelimiter[0])) {
                objTokens.push_back(strToken);
            }

            return objTokens;
        };

    };
}