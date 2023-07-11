#pragma once

#include <string>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <map>
#include <set>

#include "csv_util.h"
#include "floor.h"
#include "symbol_map.h"
#include "md_snap.h"
#include "ExecutionReport.h"

namespace pm {

    class FloorBase {
    public:
        enum EventType {
            NOOP = 0,
            ExecutionReport = 1,
            SetPositionReq = 2,
            SetPositionAck = 3,
            GetPositionReq = 4,
            GetPositionResp = 5,
            SendOrderReq = 6,
            SendOrderAck = 7,
            UserReq = 8,
            UserResp = 9,
            ExecutionReplayReq = 10,
            ExecutionReplayAck = 11,
            ExecutionReplayDone = 12,
            ExecutionOpenOrderReq = 13,
            ExecutionOpenOrderAck = 14,
            AlgoUserCommand = 15,
            AlgoUserCommandResp = 16,
            TradingStatusNotice = 17,
            TotalTypes 
        };

        struct PositionServerDetails {
            std::string market{};
            std::string symbol{};
            std::string algo{};
            double value{};
            int64_t quantity{};

            uint64_t order_id{};
            uint32_t version{};
            double profit_and_loss{};
            double mark_to_market{};

            // Aggregates
            double total_profit_and_loss{};
            double total_mark_to_market{};
            bool has_aggregate{};
        };

        struct PositionRequest {
            char algo[32];
            char symbol[32];
            int64_t qty_done;
            int64_t qty_open;
            PositionRequest() 
            : qty_done(0), qty_open(0) {
                algo[0] = 0;
                symbol[0] = 0;
            }
            PositionRequest(const std::string& algo_, 
                            const std::string& symbol_,
                            int64_t qty_done_, int64_t qty_open_)
            : qty_done(qty_done_), qty_open(qty_open_) {
                if ( (algo_.size() > sizeof(algo)-1) ||
                     (symbol_.size() > sizeof(symbol)-1) ){
                    logError("algo or symbol string size too long! %s, %s", 
                            algo_.c_str(), symbol_.c_str());
                    throw std::runtime_error("algo or symbol string size too long!");
                }
                std::strcpy(algo, algo_.c_str());
                std::strcpy(symbol, symbol_.c_str());
            }
        };

        struct PositionInstructionStat {
            std::string symbol;
            std::string algo;
            md::PriceEntry enter_bbo[2];
            double avg_px;
            int64_t cum_size;  // sign significant position
            std::vector<std::tuple<double, int64_t, int64_t>> fills;
            
            explicit PositionInstructionStat(const std::string& sym, const std::string& alg): symbol(sym), algo(alg), avg_px(0), cum_size(0) {
                setEnterBBO();
            };

            void addFill(double px, int64_t size, int64_t fill_micro=0) {
                // usually the sign should be the same
                if (size+cum_size==0) {
                    avg_px = 0;
                    cum_size = 0;
                } else {
                    avg_px = (cum_size*avg_px+(px*size))/(size+cum_size);
                    cum_size += size;
                }

                if (fill_micro==0) {
                    fill_micro = utils::TimeUtil::cur_micro();
                }
                fills.emplace_back(px, size, fill_micro);
            }

            std::string toString(bool show_each_fills) const {
                char buf[2048];
                size_t bcnt = snprintf(buf, sizeof(buf), \
                        "Fill Stat: symbol: %s, algo: %s, enter_bid: %s, enter_ask: %s, total_fill_size: %lld, avg_fill_px: %s",\
                        symbol.c_str(), algo.c_str(), enter_bbo[0].toString().c_str(), enter_bbo[1].toString().c_str(), (long long) cum_size, PriceCString(avg_px));

                if (show_each_fills) {
                    bcnt += snprintf(buf, sizeof(buf)-bcnt, "\nFills:");
                    for (const auto& f : fills) {
                        double px;
                        int64_t sz, micro;
                        std::tie(px,sz,micro) = f;
                        bcnt += snprintf(buf+bcnt, sizeof(buf)-bcnt, " [%lld,%s,%lld]", (long long) sz, PriceCString(px), (long long)micro);
                    }
                }
                return std::string(buf);
            }

            void operator+(const PositionInstructionStat& pis) {
                avg_px = ((avg_px*cum_size)+(pis.avg_px*pis.cum_size))/(cum_size+pis.cum_size);
                cum_size += pis.cum_size;
                for (const auto& f : pis.fills) {
                    fills.push_back(f);
                }
            }

            void setEnterBBO() {
                md::getBBOPriceEntry(symbol, enter_bbo[0], enter_bbo[1]);
            }
        };

        struct PositionInstruction {
            char algo[32];
            char symbol[32];

            // for PI in ordMap, the order qty
            // for PI in piMap, the target qty from strategy
            int64_t qty;

            // for PI in ordMap, the limit price used for entering order
            // for PI in piMap, the price used for specifying PI
            double px; // price upon entry

            // for PI in ordMap, only set upon order sent, cleared on any updates, and stay cleared, used to scan for unack'ed orders
            // for PI in piMap, the last time checked for processing
            int last_utc;
            int target_utc; // latest time before cancel all, in utc second
            int type; // trading style etc
            int64_t reserved;  // lpm, etc

            // bid/ask on creation time
            std::shared_ptr<PositionInstructionStat> pis; // statistics of the fills so far

            enum TYPE {
                INVALID = 0,
                MARKET = 1,
                LIMIT = 2,
                PASSIVE = 3,
                TWAP = 4,
                TWAP2 = 5,
                TRADER_WO = 6,
                TOTAL_TYPES = 7
            };

            PositionInstruction()
            : qty(0), px(0),target_utc(0), type(-1), reserved(0) {
                algo[0] = 0;
                symbol[0] = 0;
                last_utc = 0;
            }
            PositionInstruction(const std::string& algo_,
                                const std::string& symbol_,
                                int64_t qty_desired_,
                                double px_limit_ = 0,
                                int target_utc_ = 0,
                                TYPE type_ = MARKET) 
            : qty (qty_desired_), px(px_limit_), target_utc(target_utc_), type((int)type_),
              reserved(0),pis(std::make_shared<PositionInstructionStat>(symbol_, algo_))
            {
                if ( (algo_.size() > sizeof(algo)-1) ||
                     (symbol_.size() > sizeof(symbol)-1) ) {
                    logError("algo or symbol string size too long! %s, %s", 
                            algo_.c_str(), symbol_.c_str());
                    throw std::runtime_error("algo or symbol string size too long!");
                }
                std::strcpy(algo, algo_.c_str());
                std::strcpy(symbol, utils::SymbolMapReader::get().getTradableSymbol(symbol_).c_str());
                last_utc = 0;
            }

            /* the default copy constructor should work
            PositionInstruction(const PositionInstruction& pi) {
                memcpy(this, &pi, sizeof(PositionInstruction));
                pis = std::make_shared<PositionInstructionStat>();
            }*/

            PositionInstruction(const char* pi_str) :
            px(0), target_utc(0), reserved(0) {
                // format of algo, symbol, qty [, px_str|twap_str]
                // the twap_str starts with 'T', followed by a number and a 's|m|h'
                // for example T5m
                // px_str is a md parsable string
                std::string respstr;

                type = FloorBase::PositionInstruction::MARKET;
                auto tk = utils::CSVUtil::read_line(pi_str);
                if (!(tk.size() == 3) && !(tk.size() == 4)) {
                    respstr = std::string("Set position not understood, (")  + std::string(pi_str) + std::string(") check help str!");
                    logError("%s", respstr.c_str());
                    throw std::runtime_error(respstr);
                }
                std::strcpy(algo, tk[0].c_str());
                std::strcpy(symbol, utils::SymbolMapReader::get().getTradableSymbol(tk[1]).c_str());


                qty = std::stoll(tk[2]);
                if (tk.size() == 4) {
                    // get the price from price str
                    if ((tk[3][0] == 'T') || (tk[3][0] == 'Y')) {
                        // expect twap_str such as T5m
                        type = (tk[3][0] == 'T')? 
                            FloorBase::PositionInstruction::TWAP:
                            FloorBase::PositionInstruction::TWAP2;

                        int dur = std::stoi(tk[3].substr(1, tk[3].size()-2));
                        char unit = tk[3][tk[3].size()-1];
                        if (unit == 'm' || unit=='M') {
                            dur *= 60;
                        } else if (unit == 'h' || unit == 'H') {
                            dur *= 3600;
                        } else if (unit == 's' || unit == 'S') {
                        } else {
                            respstr = "unknow twap unit" + tk[3];
                            logError("%s", respstr.c_str());
                            throw std::runtime_error(respstr);
                        }
                        target_utc = utils::TimeUtil::cur_utc() + dur;
                    } else {
                        // expect px_str such as a+t1
                        type = FloorBase::PositionInstruction::LIMIT;
                        if (!md::getPriceByStr(symbol, tk[3].c_str(), px)) {
                            respstr = std::string("Error getting price str: ") + tk[3];
                            logError("%s",respstr.c_str());
                            throw std::runtime_error(respstr);
                        }
                    }
                } else {
                    // a market order
                    type = FloorBase::PositionInstruction::MARKET;
                }
                pis = std::make_shared<PositionInstructionStat>(symbol, algo);
                last_utc = 0;
            }

            std::string toString() const {
                const auto& tradable = utils::SymbolMapReader::get().getTradableSymbol(symbol);
                const auto& mts_contract = utils::SymbolMapReader::get().getByTradable(tradable)->_mts_contract;

                char buf[256];
                snprintf(buf, sizeof(buf), "PositionInstruction: %s, %s, qty(%lld), px(%s), target_utc(%s), last_utc(%s), type(%s), reserved(%lld)", 
                        algo, mts_contract.c_str(), (long long)qty, PriceCString(px),
                        target_utc==0?"":utils::TimeUtil::frac_UTC_to_string(target_utc, 0).c_str(),
                        last_utc==0?"":utils::TimeUtil::frac_UTC_to_string(last_utc, 0).c_str(),
                        TypeString((TYPE)type).c_str(),(long long)reserved);
                return std::string(buf);
            }

            std::string dumpFillStat() const {
                if (pis) {
                    return pis->toString(false);
                }
                logError("no stats to dump for %s",toString().c_str());
                return "";
            }

            void addFill(const pm::ExecutionReport& er) {
                if (__builtin_expect(utils::SymbolMapReader::get().isMLegSymbol(symbol) &&
                    (!utils::SymbolMapReader::get().isMLegSymbol(er.m_symbol)),0)) {
                    return;
                }
                pis->addFill(er.m_px, er.m_qty, er.m_recv_micro);
            }

            static std::string TypeString(TYPE type) {
                switch (type) {
                case INVALID: 
                    return "INVALID";
                case MARKET: 
                    return "MARKET";
                case LIMIT:
                    return "LIMIT";
                case PASSIVE:
                    return "PASSIVE";
                case TWAP:
                    return "TWAP";
                case TWAP2:
                    return "TWAP2";
                case TRADER_WO:
                    return "TraderWorkOrder";
                default :
                    return "Unknown";
                }
            }

            static std::string TypeString(int type) {
                return TypeString((TYPE)type);
            }

            static TYPE typeFromString(const std::string& type_name) {
                for (int i=0 ; i<(int) TYPE::TOTAL_TYPES; ++i) {
                    if (type_name == std::string(TypeString((TYPE)i))){
                        return (TYPE) i;
                    }
                }
                throw std::runtime_error(std::string("PositionInstruction Type ") + type_name + " not found!");
            }
        };

        using MsgType = utils::Floor::Message;
        using ChannelType = std::unique_ptr<utils::Floor::Channel>;

        explicit FloorBase(const std::string& name, bool is_server);
        ~FloorBase() {};

        template<typename ServerType>
        bool run_one_loop(ServerType& server); 
        // For service provider to check incoming requests
        // ServerType is expected to implement a function
        // void handleMessage(MsgType& msg_in)
        // returns true if processed a request, false if idle

        void subscribeMsgType(const std::set<int> type_set) {
            m_channel->addSubscription(type_set);
        }

        const std::string m_name;
        ChannelType m_channel;
        MsgType m_msgin, m_msgout;

        static std::shared_ptr<FloorBase> getFloor(const std::string& name, bool is_server, const std::string& floor_name);
        // this operates on a different floor by the name of floor_name.  Usually used as simulation
        // clients/server can only communicates on the same floor.
        // Use the public FloorBase constructor for default floor
        
    protected:

        FloorBase(const std::string& name, bool is_server, const std::string& floor_name);
        bool parseKeyValue (const std::string& cmd, std::map<std::string, std::string>& key_map) const;
    };

    // function definitions
    
    inline
    FloorBase::FloorBase(const std::string& name, bool is_server)
    : m_name(name), 
      m_channel(is_server? utils::Floor::get().getServer() :
                           utils::Floor::get().getClient())
    {
        logDebug("%s created as %s", name.c_str(),  (is_server?"Server":"Client"));
    };

    inline
    FloorBase::FloorBase(const std::string& name, bool is_server,  const std::string& floor_name)
    : m_name(name), 
      m_channel(is_server? utils::Floor::getByName(floor_name.c_str())->getServer() :
                           utils::Floor::getByName(floor_name.c_str())->getClient())
    {
        logDebug("%s (Name %s) created as %s",
                name.c_str(), floor_name.c_str(), (is_server?"Server":"Client"));
    };

    inline
    std::shared_ptr<FloorBase> FloorBase::getFloor(
            const std::string& name,
            bool is_server,
            const std::string& floor_name
    ) {
        return std::shared_ptr<FloorBase>( new FloorBase(name, is_server, floor_name) );
    }

    inline
    bool FloorBase::parseKeyValue (const std::string& cmd, std::map<std::string, std::string>& key_map) const {
        // parse key=val,... replace ALL to empty string
        try {
            key_map.clear();
            auto tokens = utils::CSVUtil::read_line(cmd);
            for (const auto& tk: tokens) {
                auto fields = utils::CSVUtil::read_line(tk, '=');
                if (fields.size() == 2)
                    key_map[fields[0]] = ((fields[1]=="ALL") ? "" : fields[1]);
            }
        } catch (const std::exception& e) {
            logError("problem parsing key value: %s\n%s", cmd.c_str(), e.what());
            return false;
        }
        return true;
    };

    template<typename ServerType>
    inline
    bool FloorBase::run_one_loop(ServerType& server) {
        if (m_channel->nextMessage(m_msgin)) {
            server.handleMessage(m_msgin);
            return true;
        }
        return false;
    };
};

