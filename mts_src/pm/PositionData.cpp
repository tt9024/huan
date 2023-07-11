#include "PositionData.h"
#include "time_util.h"
#include <cmath>
#include <cstdlib>
#include "md_snap.h"

namespace pm {

    IntraDayPosition::IntraDayPosition() : m_contract_size(1) { resetPositionUnsafe(0,0,0); };

    IntraDayPosition::IntraDayPosition (
            const std::string& symbol,
            const std::string& algo,
            int64_t qty,
            double vap,
            int64_t last_micro 
    )
    : m_algo(algo), m_symbol(symbol), m_contract_size(getPointValue())
    {
        resetPositionUnsafe(qty, vap, last_micro);
    }

    IntraDayPosition::IntraDayPosition(const utils::CSVUtil::LineTokens& tokens) {
        // token sequence: algo, symbol, qty, vap, pnl, last_utc, ccy read from a csv file
        m_algo = tokens[0];
        m_symbol = utils::SymbolMapReader::get().getTradableSymbol(tokens[1]);
        if (m_symbol == "") {
            throw std::runtime_error(tokens[1] + ": symbol not found!");
        }
        m_contract_size = getPointValue();
        int64_t qty = std::stoll(tokens[2]);
        double vap = std::stod(tokens[3]);
        // intra-day realized pnl starts with 0
        //double pnl = std::stod(tokens[4]);
        m_last_micro = std::stoll(tokens[5]);
        resetPositionUnsafe(qty, vap, m_last_micro);
    }

    IntraDayPosition::~IntraDayPosition() {}

    void IntraDayPosition::update(const ExecutionReport& er) {
        switch(er.m_tag39[0]) {
            case '0': // new
            {
                addOO(er);
                break;
            }

            case '1': // Partial Fill
            case '2': // Fill
            {
                int64_t qty = (int64_t) er.m_qty;
                double px = er.m_px;

                // add fill - qty to be multiplied by contract size
                addFill(qty, px, er.m_recv_micro);

                // update open order
                updateOO(er.m_clOrdId, qty);
                break;
            }

            case '3': // done for day
            case '4': // cancel
            case '5': // replaced
            case '7': // stopped
            case 'C': // Expired
            {
                deleteOO(er.m_clOrdId);
                break;
            }

            case '8': // rejected
            {
                // reject either new or replace, the state would
                // not be changed in these two cases, so just log
                logError("Reject received: %s", er.m_clOrdId);
                break;
            }
            /*
             * fall to default for loggings
             *
            case 'A': // pending new
            case 'E': // pending replace
            case '6': // pending cancel
                break;
            */

            default : // everything else
            {
                logError("unknown execution tag39(%c) received on clOrdid(%s)", er.m_tag39[0], er.m_clOrdId);
                logError("the unknown execution report detail(%s)", er.toString().c_str());
                break;
            }
        }
    }

    void IntraDayPosition::resetPositionUnsafe(int64_t qty, double vap, uint64_t last_utc) {
        // reset the position.  
        m_qty = qty;
        m_vap = vap;
        m_pnl = 0;
        m_last_micro=last_utc;
        if (m_last_micro==0) 
            m_last_micro=utils::TimeUtil::cur_micro();
    }

    void IntraDayPosition::operator+=(const IntraDayPosition& idp) {
        addFill(idp.m_qty, idp.m_vap, idp.m_last_micro);
        m_pnl += idp.m_pnl;
        for (const auto& me : idp.listOO()) {
            if (m_oo.find(me->m_clOrdId) == m_oo.end()) {
                auto iter = idp.m_oo.find(me->m_clOrdId);
                m_oo[me->m_clOrdId] = iter->second;
            }
        }
    }

    IntraDayPosition IntraDayPosition::operator+(const IntraDayPosition& idp) {
        *this+=idp;
        return *this;
    }

    void IntraDayPosition::resetPnl() {
        m_pnl = 0;
    }

    // this aggregates the two positions
    std::string IntraDayPosition::diff(const IntraDayPosition& idp, bool check_pnl) const 
    // finds difference with the given idp, 
    // return "" in case no difference is found 
    {
        // compare qty, vap and pnl
        int64_t qty1, qty2;
        double vap1, vap2, pnl1, pnl2;
        qty1=getPosition(&vap1, &pnl1);
        qty2=idp.getPosition(&vap2, &pnl2);

        // check position
        std::string ret((qty1==qty2)?"":
                "qty: "+std::to_string(qty1) + " != " + std::to_string(qty2) + "\n");

        // check vap
        if (qty1 || qty2) {
            ret += (std::string((std::fabs(vap1-vap2)<1e-10)?"":
                    "vap: "+std::to_string(vap1) + " != " + std::to_string(vap2) + " diff(" + std::to_string(std::fabs(vap1-vap2)) + ")\n"));
        }

        if (check_pnl) {
            // check pnl
            ret += (std::string((std::fabs(pnl1-pnl2)<1e-10)?"":
                    "pnl: " + std::to_string(pnl1) + " != " + std::to_string(pnl2) + "\n"));
        }

        if (ret.size()>0) {
            ret = m_algo+":"+m_symbol+" "+ret;
        }
        return ret;
    }

    int64_t IntraDayPosition::tgt_fill(int64_t tgt_qty, double tgt_vap, double* fill_px) const {
        logInfo("Calculating fill qty and price from current state %s to target_qty: %lld and target_vap: %s\n",
                toString().c_str(), (long long)tgt_qty, PriceCString(tgt_vap));
        int64_t qty = tgt_qty - m_qty;
        if (!qty) {
            logInfo("nothing to be adjusted");
            return 0;
        }
        if (fill_px) {
            if (tgt_qty * m_qty == 0) {
                *fill_px = tgt_vap;
            } else if (m_qty * qty > 0) {
                // same sidea
                *fill_px = (tgt_qty*tgt_vap - m_qty*m_vap)/qty;
            } else {
                // cover
                *fill_px = tgt_vap;
            }
        }
        logInfo("got fill_qty = %lld and fill_px = %s", (long long) qty, PriceCString(fill_px?*fill_px:0.0));
        return qty;
    }

    bool IntraDayPosition::operator==(const IntraDayPosition& idp) const {
        return diff(idp, false).size()==0;
    };

    utils::CSVUtil::LineTokens IntraDayPosition::toCSVLine(bool use_mtm_pnl, double mtm_pnl_adjust, const double* ref_px) const {
        // token sequence: algo, mts_contract, qty, vap, pnl(daily realized), last_utc, ccy
        utils::CSVUtil::LineTokens vec;
        double vap, pnl;
        int64_t qty = getPosition(&vap, &pnl);
        if (use_mtm_pnl) {
            pnl = getMtmPnl(ref_px) + mtm_pnl_adjust;
        }
        const auto* ti = utils::SymbolMapReader::get().getByTradable(m_symbol);
        vec.push_back(m_algo);
        vec.push_back(ti->_mts_contract);
        vec.push_back(std::to_string(qty));
        vec.push_back(PriceString(vap));
        vec.push_back(PnlString(pnl));
        vec.push_back(std::to_string(m_last_micro));
        vec.push_back(ti->_currency);
        return vec;
    }

    utils::CSVUtil::LineTokens IntraDayPosition::toCSVLineMtmDaily(const utils::CSVUtil::FileTokens& prev_vec, const double* ref_px) const {
        // token sequence: algo, mts_contract, qty, vap, pnl(daily realized plus DAILY outstanding mtm change), last_utc, ccy
        // search the prev_vec for the the algo and symbol
        // if found, adjust with previous mtm and pnl as the following:
        //      daily_mtm_pnl = current_mtm - (previous_mtm - prevous_pnl)
        //
        const auto* ti = utils::SymbolMapReader::get().getByTradable(m_symbol);
        for (const auto& line: prev_vec) {
            if ( (line[0] == m_algo) && (line[1] == ti->_mts_contract) ) {
                const double last_mtm = std::stod(line[4]);  // refer to toCSVLineMtm() for position
                const double last_pnl = std::stod(line[5]);
                return this->toCSVLine(true, -(last_mtm-last_pnl), ref_px);
            }
        }
        return this->toCSVLine(true,0, ref_px);
    }

    utils::CSVUtil::LineTokens IntraDayPosition::toCSVLineMtm(const double* ref_px) const {
        // token sequence: algo, mts_contract, qty, vap, pnl(daily realized plus outstanding mtm), pnl(daily realized), last_utc, ccy
        utils::CSVUtil::LineTokens vec;
        double vap, pnl, pnl_mtm = getMtmPnl(ref_px);
        int64_t qty = getPosition(&vap, &pnl);
        const auto* ti = utils::SymbolMapReader::get().getByTradable(m_symbol);
        vec.push_back(m_algo);
        vec.push_back(ti->_mts_contract);
        vec.push_back(std::to_string(qty));
        vec.push_back(PriceString(vap));
        vec.push_back(PnlString(pnl_mtm));
        vec.push_back(PnlString(pnl));
        vec.push_back(std::to_string(m_last_micro));
        vec.push_back(ti->_currency);
        return vec;
    }

    std::string IntraDayPosition::toString() const {
        double vap, pnl, pnl_mtm = getMtmPnl();
        int64_t qty = getPosition(&vap, &pnl);

        int pnlms = PnlString(pnl_mtm).size();
        int pnls = PnlString(pnl).size();
        size_t ms = _MAX_(0, (int) (8-pnlms));
        size_t ps = _MAX_(0, (int) (8-pnls));
        std::string s1 (ms, ' '), s2(ps, ' ');

        char buf[256];
        size_t bytes = snprintf(buf, sizeof(buf), "%-13s  %-16s  %-5s  %-10s  %-8s   (%-8s)  ",
                m_algo.c_str(), 
                utils::SymbolMapReader::get().getByTradable(m_symbol)->_mts_contract.c_str(), 
                std::to_string(qty).c_str(), PriceString(vap).substr(0,10).c_str(), (s1+PnlStringColor(pnl_mtm)).c_str(), (s2+PnlStringColor(pnl)).c_str());
        bytes += utils::TimeUtil::frac_UTC_to_string(m_last_micro, buf+bytes, sizeof(buf)-bytes,6);
        return std::string(buf);
    }

    std::string IntraDayPosition::dumpOpenOrder() const {
        std::string ret(m_algo+":"+utils::SymbolMapReader::get().getByTradable(m_symbol)->_mts_contract+" "+std::to_string(m_oo.size())+" open orders");
        auto oovec (listOO());
        for (const auto& oo:oovec) {
            ret += "\n\t";
            ret += oo->toString();
        }
        return ret;
    }

    bool IntraDayPosition::hasPosition() const {
        return m_qty != 0;
    }

    int IntraDayPosition::getPointValue() const {
        try {
            return int(utils::SymbolMapReader::get().getByTradable(m_symbol)->_point_value + 0.5);
        } catch (const std::exception & e) {
            logError("cannot get point value for %s, set to 1.0", m_symbol.c_str());
            return 1;
        }
    }

    int64_t IntraDayPosition::getPosition(double* ptr_vap, double* ptr_pnl) const {
        if (ptr_vap) *ptr_vap=m_vap;
        if (ptr_pnl) *ptr_pnl=m_pnl;
        return m_qty;
    }

    int64_t IntraDayPosition::getOpenQty() const {
        int64_t qty = 0;
        for(auto iter=m_oo.begin(); iter!=m_oo.end(); ++iter) {
            qty += iter->second->m_open_qty;
        }
        return qty;
    }

    std::vector<std::shared_ptr<const OpenOrder> > IntraDayPosition::listOO() const {
        std::vector<std::shared_ptr<const OpenOrder> > vec;
        for (auto iter=m_oo.begin(); iter!=m_oo.end(); ++iter) {
            if (iter->second->m_open_qty) {
                vec.push_back(iter->second);
            }
        };
        return vec;
    }

    std::shared_ptr<const OpenOrder> IntraDayPosition::findOO(const std::string& clOrdId) const {
        const auto iter = m_oo.find(clOrdId);
        if (iter != m_oo.end()) {
            return iter->second;
        }
        return std::shared_ptr<const OpenOrder>();
    }

    double IntraDayPosition::getRealizedPnl() const {
        double pnl;
        getPosition(nullptr, &pnl);
        return pnl;
    }
    
    double IntraDayPosition::getMtmPnl(const double* ref_px_ptr) const {
        double vap, pnl;
        int64_t m_qty = getPosition(&vap, &pnl);
        if (m_qty == 0) {
            return pnl;
        }
        double ref_px = 0;
        if (!ref_px_ptr) {
            try {
                double bidpx, askpx;
                int bidsz,  asksz;
                if (md::getBBO(m_symbol, bidpx, bidsz, askpx, asksz)) {
                    ref_px = m_qty > 0? bidpx : askpx;
                } else {
                    logInfo("cannot get %s snap price, MTM pnl not calculated", m_symbol.c_str());
                }
            } catch (const std::exception& e) {
                logInfo("Exception get %s snap price, MTM pnl not calculated: %s", m_symbol.c_str(), e.what());
            }
        } else {
            ref_px = *ref_px_ptr;
        }
        return pnl + m_qty*(ref_px-vap) * m_contract_size;
    }

    void IntraDayPosition::addOO(const ExecutionReport& er) {
        auto oop_iter = m_oo.find(er.m_clOrdId);
        if (oop_iter!=m_oo.end()) {
            const auto& oop = oop_iter->second;
            logInfo("Warning! Recived a new on existing clOrdId, overwriting.\nIncoming new: %s, existing open order %s", er.toString().c_str(), oop->toString().c_str());
            oop_iter->second = std::make_shared<OpenOrder>(this, er);
        } else {
            m_oo.emplace(er.m_clOrdId, std::make_shared<OpenOrder>(this, er));
        }
    }

    void IntraDayPosition::deleteOO(const char* clOrdId) {
        if (clOrdId) {
            auto iter = m_oo.find(clOrdId);
            if (iter != m_oo.end()) {
                iter->second->m_open_qty=0;
                m_oo.erase(iter);
            } else {
                logInfo("Warning! delete a nonexisting open order! clOrdId: %s", clOrdId);
            }
            return ;
        }
        for (auto iter = m_oo.begin(); iter!=m_oo.end(); ++iter) {
            iter->second->m_open_qty = 0;
        }
        m_oo.clear();
    }

    void IntraDayPosition::updateOO(const char* clOrdId, int64_t qty) {
        auto iter = m_oo.find(clOrdId);
        if (iter != m_oo.end()) {
            iter->second->m_open_qty-=qty;
            if (iter->second->m_open_qty == 0) {
                deleteOO(clOrdId);
            }
        } else {
            logInfo("Warning! update a nonexisting open order! clOrdId: %s, qty: %lld", clOrdId, (long long)qty);
        }
    }

    void IntraDayPosition::addFill(int64_t qty, double px, uint64_t utc_micro) {
        if (qty * m_qty == 0) {
            // initial qty
            if (m_qty == 0) {
                m_vap = px;
            }
        } else if (qty * m_qty > 0) {
            // add to position
            m_vap = (m_vap * m_qty + px * qty)/(qty+m_qty);
        } else {
            // covering, update pnl
            if (qty + m_qty == 0) {
                // total covering
                m_pnl -= (m_qty*m_vap + qty*px)*m_contract_size;
                m_vap = 0;
            } else if ((qty + m_qty)*m_qty > 0) {
                // partial cover
                m_pnl -= (-qty*m_vap + qty*px)*m_contract_size;
            } else {
                // cover + new position
                m_pnl -= (m_qty*m_vap - m_qty*px)*m_contract_size;
                m_vap = px;
            }
        }
        m_qty += qty;
        m_last_micro = utc_micro;
    }


    OpenOrder::OpenOrder(const IntraDayPosition* idp) 
    : m_idp(idp), m_ord_qty(0), m_open_qty(0), m_open_px(0), m_open_micro(0)
    {
        memset(m_clOrdId, 0, sizeof(m_clOrdId));
    }

    OpenOrder::OpenOrder(const IntraDayPosition* idp, const ExecutionReport& er)
    : m_idp(idp), m_ord_qty(er.m_qty), m_open_qty(er.m_qty), m_open_px(er.m_px), m_open_micro(er.m_recv_micro)
    {
        memcpy(m_clOrdId, er.m_clOrdId, sizeof(IDType));
        if (er.m_reserved != 0) {
            // take as cumQty
            m_open_qty -= er.m_reserved;
            logInfo("got new with cumQty: er:%s, oo:%s",
                    er.toString().c_str(), 
                    toString().c_str());
        }
    }

    std::string OpenOrder::toString() const {
        char buf[256];
        snprintf(buf, sizeof(buf), 
                "OpenOrder(clOrdId=%s,%s,ord_qty=%lld,open_qty=%lld,open_px=%s,open_time=%s)",
                m_clOrdId, m_open_qty>0?"Buy":"Sell",std::llabs(m_ord_qty), std::llabs(m_open_qty), 
                PriceCString(m_open_px),
                utils::TimeUtil::frac_UTC_to_string(m_open_micro,6).c_str());
        return std::string(buf);
    }
}
