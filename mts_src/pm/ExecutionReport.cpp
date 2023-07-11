#include "ExecutionReport.h"
#include "time_util.h"
#include <cmath>
#include <random>
#include "plcc/PLCC.hpp"
#include "md_snap.h"

namespace pm {
    ExecutionReport::ExecutionReport() {
        memset((void*)this, 0, sizeof(ExecutionReport));
    }

    ExecutionReport::ExecutionReport (
           const std::string& symbol, // exchange symbol
           const std::string& algo,   // fully qualified for position
           const std::string& clOrdId,
           const std::string& execId,
           const std::string& tag39,
           int qty,   // + buy, - sell
           double px,
           const std::string& utcTime, //YYYYMMDD-HH:MM:SS[.sss] GMT
           const std::string& optionalTag, // reserved 
           uint64_t recv_micro,
           int64_t reserved
    )
    :  m_qty(qty), m_px(px), 
       m_utc_milli(utils::TimeUtil::string_to_frac_UTC(utcTime.c_str(), 3, NULL, true)),
       m_recv_micro(recv_micro), m_reserved(reserved)
    {
        snprintf(m_symbol, sizeof(m_symbol),"%s", ((symbol == "")?"":utils::SymbolMapReader::get().getByTradable(symbol)->_tradable.c_str()));
        snprintf(m_algo, sizeof(m_algo), "%s", algo.c_str());
        snprintf(m_clOrdId, sizeof(m_clOrdId), "%s", clOrdId.c_str());
        snprintf(m_execId, sizeof(m_execId), "%s", execId.c_str());
        snprintf(m_tag39, sizeof(m_tag39), "%s", tag39.c_str());
        snprintf(m_optional, sizeof(m_optional), "%s", optionalTag.c_str());
        if (m_recv_micro==0) {
            m_recv_micro=utils::TimeUtil::cur_micro();
        }
    }

    ExecutionReport ExecutionReport::fromCSVLine(const utils::CSVUtil::LineTokens& token_vec) {
        int64_t reserved=0;
        if (token_vec.size() > 10) {
            reserved = std::stoll(token_vec[10]);
        }
        return ExecutionReport(token_vec[0], token_vec[1], token_vec[2], token_vec[3],
                token_vec[4], std::stoi(token_vec[5]), std::stod(token_vec[6]),
                token_vec[7], token_vec[8], std::stoull(token_vec[9]),reserved);
    }

    utils::CSVUtil::LineTokens ExecutionReport::toCSVLine() const {
        std::vector<std::string> token_vec;
        token_vec.push_back(m_symbol);
        token_vec.push_back(m_algo);
        token_vec.push_back(m_clOrdId);
        token_vec.push_back(m_execId);
        token_vec.push_back(m_tag39);
        token_vec.push_back(std::to_string(m_qty));
        token_vec.push_back(PriceString(m_px));
        token_vec.push_back(utils::TimeUtil::frac_UTC_to_string(m_utc_milli, 3, NULL, true));
        token_vec.push_back(m_optional);
        token_vec.push_back(std::to_string(m_recv_micro));
        token_vec.push_back(std::to_string(m_reserved));
        return token_vec;
    }

    utils::CSVUtil::LineTokens ExecutionReport::toFillsCSVLine() const {
        const int symbol_idx = 0;
        std::vector<std::string> token_vec (toCSVLine());
        token_vec[symbol_idx] = utils::SymbolMapReader::get().getByTradable(m_symbol)->_mts_contract;
        return token_vec;
    }

    std::string ExecutionReport::toString() const {
        char buf[512];
        snprintf(buf, sizeof(buf), 
                "Execution Report [symbol=%s, algo=%s, clOrdId=%s, execId=%s, tag39=%s, qty=%d, px=%s, execTime=%s, tag=%s, recvTime=%s, reserved=%lld]",
                ((m_symbol[0]==0)?"":utils::SymbolMapReader::get().getByTradable(m_symbol)->_mts_contract.c_str()),
                m_algo, m_clOrdId, m_execId, m_tag39, m_qty, PriceCString(m_px),
                utils::TimeUtil::frac_UTC_to_string(m_utc_milli, 3).c_str(), m_optional, 
                utils::TimeUtil::frac_UTC_to_string(m_recv_micro, 6).c_str(), 
                (long long) m_reserved);
        return std::string(buf);
    }

    bool ExecutionReport::isFill() const {
        return (m_tag39[0] == '1') || (m_tag39[0] == '2');
    }

    bool ExecutionReport::isNew() const {
        return (m_tag39[0] == '0');
    }

    bool ExecutionReport::isCancel() const {
        return (m_tag39[0] == '4');
    }

    bool ExecutionReport::isReject() const {
        return (m_tag39[0] == '8');
    }

    bool ExecutionReport::compareTo(const ExecutionReport& er, std::string* difflog)  const {
        const auto& l1 = toCSVLine();
        const auto& l2 = er.toCSVLine();
        if (l1.size() != l2.size()) {
            if (difflog) {
                *difflog = "size mismatch!";
                return false;
            }
        }
        const size_t px_col = 5;
        for (size_t i=0; (i<l1.size()) && (i!=px_col); ++i) {
            if (l1[i] != l2[i]) {
                if (difflog) {
                    *difflog = "column " + std::to_string(i) + " mismatch!";
                }
                return false;
            }
        }
        if (std::fabs(m_px-er.m_px)>1e-10) {
            if (difflog) {
                *difflog = "column " + std::to_string(px_col) + " mismatch!";
            }
            return false;
        }
        return true;
    }

    std::string ExecutionReport::ERPersistFile() {
        return plcc_getString("ERPersistFile");
    }

    bool ExecutionReport::loadFromPersistence(const std::string& start_time_local, const std::string& end_time_local, const std::string& out_file, const std::string& er_persist_file) {
        // read fills from the fill file and dump lines to out_file since start_utc.
        // start_time_local is specified with a local time string in YYYYMMDD-HH:MM:SS (inclusive)
        // out_file is path to the output file (includes the path)
        // end_time_local, if not empty, has same time format for end time (exclusive)

        const std::string& erfile (er_persist_file==""? ERPersistFile():er_persist_file);
        const auto lines = utils::CSVUtil::read_file(erfile);
        utils::CSVUtil::FileTokens line_vec;

        uint64_t smicro = utils::TimeUtil::string_to_frac_UTC(start_time_local.c_str(), 0) * 1000000;
        uint64_t emicro = (
                end_time_local.size() > 0 ? 
                (utils::TimeUtil::string_to_frac_UTC(end_time_local.c_str(), 0) * 1000000) : 
                (uint64_t)(-1));

        for (const auto& line : lines ) {
            try {
                auto er (pm::ExecutionReport::fromCSVLine(line));
                if ((er.m_recv_micro >= smicro) &&
                    (er.m_recv_micro < emicro)) 
                {
                    line_vec.emplace_back(er.toCSVLine());
                }
            } catch (const std::invalid_argument& e) {
                logDebug("Invalid argument - failed to load execution report: %s", e.what());
            } catch (const std::exception& e) {
                logError("failed to load execution report: %s", e.what());
            }
        }
        return utils::CSVUtil::write_file(line_vec, out_file, false);
    }

    const char* ExecutionReport::syntheticOrdIdPrefix() {
        return "Syn";
    }

    ExecutionReport ExecutionReport::genSyntheticFills(const std::string& symbol,
                                                 const std::string& algo,
                                                 int qty, // sign significant
                                                 double px,
                                                 const std::string& clOrdId_str,
                                                 const std::string& optional_tag) {
        // normalize the px
        md::getPriceByStr(symbol, std::to_string(px).c_str(), px);
        int64_t cur_micro = utils::TimeUtil::cur_micro();
        const std::string clOrdId (clOrdId_str);
        const std::string execId (clOrdId + "ExId");
        const std::string tag39 (std::to_string(2)); // filled
        const std::string utcTime (utils::TimeUtil::frac_UTC_to_string(cur_micro, 3, NULL, true));
        const auto er = ExecutionReport ( symbol,
                                 algo,
                                 clOrdId,
                                 execId,
                                 tag39,
                                 qty,
                                 px,
                                 utcTime,
                                 optional_tag,
                                 cur_micro,
                                 0);
        logInfo("Generating a synthetic fill: %s", er.toString().c_str());
        return er;
    }

    std::string ExecutionReport::genClOrdId() {
        static thread_local std::mt19937 generator;
        std::uniform_int_distribution<int> nano(0,1000);
        uint64_t cur_micro = utils::TimeUtil::cur_micro()*1000ULL + (uint64_t)nano(generator);
        return std::to_string(cur_micro);
    }
    
    std::string ExecutionReport::genClOrdId(int pi_type, uint32_t woid) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d-%u-%s", (int)pi_type, (unsigned int) woid, genClOrdId().c_str());
        return std::string(buf);
    }

    std::string ExecutionReport::genReplaceClOrdId(const std::string& clOrdId) {
        int pi_type=0;
        uint32_t woid = 0;
        const auto& rcid (parseClOrdId(clOrdId.c_str(), pi_type, woid));
        if (pi_type != -1) {
            return genClOrdId(pi_type, woid);
        }
        return genClOrdId();
    }

    std::string ExecutionReport::parseClOrdId(const char* clOrdId) {
        // parse out the ':', 
        // In case a synthetic fill, the format is Syn:ClOrdId:cur_micro_str
        // In case a real fill, the format is simply ClOrdId
        const auto& tk = utils::CSVUtil::read_line(clOrdId, ':');
        std::string ord_id = tk[0];
        if (__builtin_expect(tk.size() == 3, 0)) {
            ord_id = tk[1];
        }
        return ord_id;
    }

    std::string ExecutionReport::parseClOrdId(const char* clOrdId, int& pi_type, uint32_t& woid) {
        // clOrdId in format of "pi_type-woid-cur_micro_str"
        const auto& ord_id = parseClOrdId(clOrdId);
        const auto& tk = utils::CSVUtil::read_line(ord_id, '-');
        if (__builtin_expect(tk.size() == 3, 1)) {
            pi_type = std::stoi(tk[0]);
            woid = (uint32_t) std::stoul(tk[1]);
        } else {
            pi_type = -1;
            woid = (uint32_t)-1;
        }
        return ord_id;
    }
}
