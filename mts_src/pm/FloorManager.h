#pragma once

#include "FloorCPR.h"

#include <Interface/PositionServer.h>

namespace pm {
    class FloorManager :
            public FloorCPR<FloorManager> {
    public:
        static FloorManager& get();
        ~FloorManager();

        explicit FloorManager(const std::string& name);
        FloorManager(const FloorManager& mgr) = delete;
        FloorManager& operator=(const FloorManager& mgr) = delete;

        void start_derived(); //called upon position loaded and run_loop_drived will be called afterwards
        void shutdown_derived();
        std::string toString_derived() const;
        void addPositionSubscriptions_derived(); // C++ position get/set
        void run_loop_derived(); // extra handling after handling all messages in loop
        void updatePositionServer(bool snapshot = false); // Update PS with real-time p&l
        void handleExecutionReport_derived(const pm::ExecutionReport& er); // extra handling after receiving an er
        bool handleUserReq_derived(const MsgType& msg, std::string& respstr);  // handles all except buy/sell, position req, display, return emtpy string if not handled

        bool handlePositionReq_derived(const MsgType& msg, MsgType& msg_out);  
        // handles positions requests from C++ and user cmd 'X' 
        // Note - for msg type GetPositionReq - it comes from C++, 
        // with msg buffer being the PositionRequest;
        // for msg type SetPositionReq, it comes from both C++ and 
        // 'X', with msg buffer being PositionInstruction.

        friend class FloorBase;
        bool isFloorManager() const { return true; };

        [[nodiscard]] bool isPositionServerEnabled() const { return enable_position_server_; }
    private:
        std::string handleUserReqMD(const char* cmd);
        enum {
            IdleSleepMicro       = 50000,    // sleep duration

            // execution related parameters
            ScanIntervalMicro    = 100000,    // open order scan interval
            MaxPegTickDiff     = 3,       // maximum spreads before go aggressive in peg
            PegAggIntervalMilli  = 2000,      // OO Peg, wait between consecutive aggressive
        };

        bool enable_position_server_{};

        bool shouldScan(const std::shared_ptr<const OpenOrder>& oo, bool& peg_passive) const;
        bool scanOpenOrders();

        void initializePositionServerInterface() const;

        [[nodiscard]] std::optional<double> getReferencePx(const std::string& symbol) const;
        void syncLocalPositionServerStates(const std::vector<com::massar::core::interface::AlgoStateMessage>& local_algo_states,
                auto now) const;
    };
};
