#pragma once

#include "ExecutionTrader.h"

/*
 * simple TWAP minute-by-minute algo, parameters
 * 
 * qty_interval_micro = 60000000 #60 seconds
 * oo_interval_micro  = 10000000 #10 seconds
 * max_slice_qty = 20
 *
 */

namespace pm {
    class TWAPTrader : public ExecutionTrader {
    public:
        TWAPTrader(const std::string& name, FloorTrader& floor, const std::string& cfg_fn);

        std::shared_ptr<ETInfo> onEntry(const PIType& pi) override;
        bool onUpdate(const PIType& pi, std::shared_ptr<ETInfo>& etinfo) override;
        bool onRun(std::shared_ptr<ETInfo>& etinfo) override;
        bool onER(const ExecutionReport& er, std::shared_ptr<ETInfo>& etinfo) override;

        ~TWAPTrader() {};

    protected:
        struct TWAPState {
            uint64_t _last_qty_micro;
            uint64_t _last_oo_micro;
            TWAPState()
            : _last_qty_micro(0),
              _last_oo_micro(0) {
            }
        };

        struct TWAPParam {
            // parameters
            uint64_t qty_interval_micro;
            uint64_t oo_interval_micro;
            int max_slice_qty;
            explicit TWAPParam (const std::string& cfg_fn);
            std::string toString() const;
        };

        TWAPParam _param;

    private:
        uint64_t scanQty(const std::shared_ptr<ETInfo>& etinfo, std::shared_ptr<TWAPState>& state);
        /*
        uint64_t scanOO(const std::shared_ptr<ETInfo>& etinfo, std::shared_ptr<TWAPState>& state);
        */
    };
}
