#include "AlgoBase.h"

namespace algo {

    AlgoBase::AlgoBase(const std::string& name, pm::FloorBase::ChannelType& channel)
    : m_name(name), m_channel(channel), m_should_run(false) {
    }

    AlgoBase::~AlgoBase() {
        logInfo("%s Base destructed", m_name.c_str());
    }

    std::string AlgoBase::toString(bool dump_state) const {
        char buf[256];
        snprintf(buf, sizeof(buf), "name=%s, cfg=%s, should_run=%s", 
                m_name.c_str(), cfgFile().c_str(), (m_should_run?"Y":"N"));
        std::string ret (buf);

        if (!dump_state) {
            return ret;
        }

        ret += " subscriptions=[ ";
        for (const auto& sym: m_symbols) {
            ret += sym->toString();
            ret += " ";
        }
        ret += "] state=[ ";
        ret += onDump();
        ret += " ]";
        return ret;
    }

    // out-going updates
    bool AlgoBase::getPosition(int symid, int64_t& qty_done, int64_t& qty_open) {
        static pm::FloorBase::MsgType msgreq(pm::FloorBase::GetPositionReq, nullptr, 0), msgresp;

        const auto& sinfo (m_symbols[symid]);
        const pm::FloorBase::PositionRequest pr(m_name, sinfo->_bcfg.symbol, 0, 0);
        msgreq.copyData((const char*)&pr, sizeof(pm::FloorBase::PositionRequest));
        msgresp.copyString("");
        if (! m_channel->request(msgreq, msgresp)) {
            logError("%s cannot get position: %s", m_name.c_str(), msgresp.buf);
            return false;
        }
        const auto* prr((const pm::FloorBase::PositionRequest*)msgresp.buf);
        qty_done = prr->qty_done;
        qty_open = prr->qty_open;
        return true;
    }

    bool AlgoBase::setPositionMarket(int symid, int64_t tgt_qty) {
        const auto& sinfo (m_symbols[symid]);
        return setPosition(pm::FloorBase::PositionInstruction(
                    m_name, 
                    sinfo->_bcfg.symbol, 
                    tgt_qty, 
                    0, 
                    0, 
                    pm::FloorBase::PositionInstruction::MARKET
                )
            );
    }

    bool AlgoBase::setPositionLimit(int symid, int64_t tgt_qty, double tgt_px) {
        const auto& sinfo (m_symbols[symid]);
        return setPosition(pm::FloorBase::PositionInstruction(
                    m_name, 
                    sinfo->_bcfg.symbol, 
                    tgt_qty, 
                    tgt_px, 
                    0, 
                    pm::FloorBase::PositionInstruction::LIMIT
                )
            );
    }

    bool AlgoBase::setPositionPassive(int symid, int64_t tgt_qty, double tgt_px,  time_t tgt_utc) {
        const auto& sinfo (m_symbols[symid]);
        return setPosition(pm::FloorBase::PositionInstruction(
                    m_name, 
                    sinfo->_bcfg.symbol, 
                    tgt_qty, 
                    tgt_px, 
                    tgt_utc, 
                    pm::FloorBase::PositionInstruction::PASSIVE
                )
            );
    }

    bool AlgoBase::setPositionTWAP(int symid, int64_t tgt_qty, time_t tgt_utc) {
        const auto& sinfo (m_symbols[symid]);
        return setPosition(pm::FloorBase::PositionInstruction(
                    m_name, 
                    sinfo->_bcfg.symbol, 
                    tgt_qty, 
                    0, 
                    tgt_utc, 
                    pm::FloorBase::PositionInstruction::TWAP
                )
            );
    }

    bool AlgoBase::setPositionTWAP2(int symid, int64_t tgt_qty, time_t tgt_utc) {
        const auto& sinfo (m_symbols[symid]);
        return setPosition(pm::FloorBase::PositionInstruction(
                    m_name, 
                    sinfo->_bcfg.symbol, 
                    tgt_qty, 
                    0, 
                    tgt_utc, 
                    pm::FloorBase::PositionInstruction::TWAP2
                )
            );
    }

    bool AlgoBase::setPosition(const pm::FloorBase::PositionInstruction& pi) {
        static pm::FloorBase::MsgType msgreq(pm::FloorBase::SetPositionReq, nullptr, 0), msgresp;

        msgreq.copyData((const char*)&pi, sizeof(pm::FloorBase::PositionInstruction));
        msgresp.copyString("");
        if (!m_channel->requestAndCheckAck(msgreq, msgresp, 1, pm::FloorBase::SetPositionAck)) {
            logError("%s failed to set position: %s", m_name.c_str(), msgresp.buf);
            return false;
        }
        return true;
    }

    bool AlgoBase::coverPosition(int symid) {
        return setPositionMarket(symid, 0);
    }

    bool AlgoBase::getBar(int symid, md::BarPrice& bp) {
        const auto& sinfo (m_symbols[symid]);
        return sinfo->_bar_reader->read(bp);
    }

    bool AlgoBase::getSnap(int symid, md::BookDepot& sp) {
        const auto& sinfo (m_symbols[symid]);
        return sinfo->_snap_reader->getLatestUpdate(sp);
    }

    bool AlgoBase::getHistBar(int symid, int numBars, std::vector<std::shared_ptr<md::BarPrice> >& bp) {
        const auto& sinfo (m_symbols[symid]);
        return sinfo->_bar_reader->readLatest(bp, numBars);
    }

    int AlgoBase::getNextBar(int symid, time_t since, std::vector<std::shared_ptr<md::BarPrice> >& bp) {
        const auto& sinfo (m_symbols[symid]);
        md::BarPrice bar;
        sinfo->_bar_reader->read(bar);
        if (since + sinfo->_barsec > bar.bar_time) {
            return 0;
        }
        if (since + sinfo->_barsec == bar.bar_time) {
            bp.emplace_back( new md::BarPrice(bar) );
            return 1;
        }
        // a bit slower operation, needs to read the bar file and do gap fills
        return sinfo->_bar_reader->readPeriod(bp, since + sinfo->_barsec, bar.bar_time);
    }

    int AlgoBase::addSymbol(const std::string& venue, const std::string& symbol,
                            const std::string& snap_level, int barsec) {
        m_symbols.emplace_back(
                std::make_shared<SymbolInfo>(
                    md::BookConfig(venue, symbol, snap_level), 
                    barsec
                )
            );
        logInfo("%s subscribed to %s", m_name.c_str(), m_symbols[m_symbols.size()-1]->toString().c_str());
        return (int)m_symbols.size()-1;
    }

    void AlgoBase::removeSymbol(int symid) {
        m_symbols.erase(m_symbols.begin() + symid);
    }

    void AlgoBase::removeAllSymbol() {
        m_symbols.clear();
    }

    AlgoBase::SymbolInfo::SymbolInfo(md::BookConfig&& bcfg, int barsec)
    : _bcfg(std::move(bcfg)),
      _barsec(barsec), 
      _bq(std::make_shared<md::BookQType>(_bcfg, true)),
      _snap_reader(_bq->newReader()),
      _bar_reader(std::make_shared<md::BarReader>(_bcfg, barsec))
    {
    }

    std::string AlgoBase::SymbolInfo::toString() const {
        char buf[256];
        snprintf(buf, sizeof(buf), "bcfg=%s, barsec=%d",
                _bcfg.toString().c_str(), _barsec);
        return std::string(buf);
    }

    AlgoBase::SymbolInfo AlgoBase::getSymbolInfo(int symid) const {
        return *m_symbols[symid];
    }
}
