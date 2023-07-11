#include "AlgoThread.h"
#include "md_snap.h"
#include "PositionManager.h"
#include "strat/AR1.h"
#include "strat/idbo/idbo_tf.h"

namespace pm {

class PositionManagerMock : public PositionManager {
public:

    PositionManagerMock(const std::string& name, const std::string& trace_csv_file)
    : PositionManager(name),
      _trace_csv(fopen(trace_csv_file.c_str(), "at+"))
    {
        if (!_trace_csv) {
            logError("Cannot open trace file %s", trace_csv_file.c_str());
            throw std::runtime_error("Cannot open trace file " + trace_csv_file);
        }
        logInfo("PositionManangerMock %s created, writing trace to %s", name.c_str(), trace_csv_file.c_str());
    };

    ~PositionManagerMock() {
        if (_trace_csv) {
            fclose(_trace_csv);
            _trace_csv = NULL;
        }
    }

    // This by passes execution report and directly call idp's private addFill
    void update(const std::string& algo, const std::string& symbol, int32_t qty, double px) {
        // get the IntraDayPosition from algo, symbol
        auto idp = m_algo_pos[algo][symbol];
        if (!idp) {
            idp = std::make_shared<IntraDayPosition>(symbol, algo);
            m_algo_pos[algo][symbol] = idp;
            m_symbol_pos[symbol][algo] = idp;
        }

        // update idp with given qty and px
        idp->addFill(qty, px, utils::TimeUtil::cur_micro());
       
        // write the position/pnl to trace csv
        auto token_vec = idp->toCSVLine();
        token_vec.insert(token_vec.begin(), utils::TimeUtil::frac_UTC_to_string(0,0));
        utils::CSVUtil::write_file(token_vec, _trace_csv);
    };

private:
    FILE* _trace_csv;

};
}; // namespace pm

namespace algo {

template<typename Strat>
class StratMock : public Strat {
public:
    StratMock(const std::string& name, const std::string& strat_cfg, pm::FloorBase::ChannelType& channel, uint64_t cur_micro,
             std::shared_ptr<pm::PositionManagerMock>& pm)
    : Strat(name, strat_cfg, channel, cur_micro),
      _pm(pm)
    {}

    // get current position from local PM
    bool getPosition(int symid, int64_t& qty_done, int64_t& qty_open) {
        const auto& sinfo (this->getSymbolInfo(symid));
        qty_done = _pm->getPosition(this->getName(), sinfo._bcfg.symbol);
        qty_open = 0;
        return true;
    }

    // Assuming 100% fill, update local PM with desired qty.
    //
    // Note this logic is not exactly the same with FloorManager in 
    // live trading, but very similar.  We could reuse that part of
    // logic in simulation, however, that would involve a separate
    // process using floor and therefore timing difficulties. 
    // Alternatively, we can mock the floor communication to be local.
    // This has not yet been implemented, to be reviewed in end-to-end
    // system simulations.
    //
    bool setPosition(const pm::FloorBase::PositionInstruction& pi) {
        // get the qty and px
        int64_t done_qty = _pm->getPosition(pi.algo, pi.symbol), open_qty = 0;
        int64_t trade_qty = pi.qty - (done_qty + open_qty);
        double px = pi.px;
        if (__builtin_expect(trade_qty == 0, 0)) {
            return true;
        };

        try {
            // figure out the execution price
            double bidpx, askpx;
            int bidsz, asksz;
            if (!md::getBBO(pi.symbol, bidpx, bidsz, askpx, asksz)) {
                logError("Problem getting BBO for %s", pi.symbol);
                return false;
            }
            //double ticksz = utiils::SymbolMapReader::TickSize(pi.symbol);
            if (pi.type == pm::FloorBase::PositionInstruction::MARKET) {
                // aggressive 
                px = ((trade_qty > 0)? askpx: bidpx);
            } else if (pi.type == pm::FloorBase::PositionInstruction::PASSIVE) {
                // passive
                px = ((trade_qty > 0)? bidpx : askpx);
            }
        } catch (const std::exception& e) {
            logError("Exception thrown in setPosition: %s", e.what());
            return false;
        }
        _pm->update(pi.algo, pi.symbol, trade_qty, px);
        return true;
    }

private:
    std::shared_ptr<pm::PositionManagerMock> _pm;
};

/*
 * Algothread that creates strategies that mock get/set position
 * with a local position manager, which simply satisfies desired
 * position with desired price.
 */

class AlgoThreadMock : public AlgoThread {
public:
    AlgoThreadMock(const std::string& inst, const std::string& cfg)
    : AlgoThread(inst, cfg),
      _pm(std::make_shared<pm::PositionManagerMock>(std::string("mockpm_")+inst, plcc_getString("SimTraceFile")))
    {
        init();
        logInfo("PositionManagerMock %s created!", _pm->toString().c_str());
    }

    // this creates a mock version for simulation
    void addAlgo(const std::string& name,  const std::string& class_name,const std::string& cfg) {
        std::shared_ptr<AlgoBase> algp;
        if (class_name == "AR1") {
            algp = std::make_shared<StratMock<AR1> >(name, cfg, m_floor.m_channel, TimerType::cur_micro(), _pm);
        } else if (class_name == "IDBO_TF") {
            algp = std::make_shared<StratMock<IDBO_TF> >(name, cfg, m_floor.m_channel, TimerType::cur_micro(), _pm);
        } else {
            logError("Unknown object name (%s) when createing strategy %s", class_name.c_str(), name.c_str()); 
        }
        m_algo_map.emplace(name, algp);
    }

    void eodPosition() {
        logInfo("Writing EoD Position to %s", _pm->eod_csv().c_str());
        _pm->persist();
    }

private:
    std::shared_ptr<pm::PositionManagerMock> _pm;
};

}
