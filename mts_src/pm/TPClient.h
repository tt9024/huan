#pragma once

#include "ExecutionReport.h"
#include "FloorBase.h"
#include "csv_util.h"
#include <stdio.h>

namespace pm {

    template<typename OrderConnection>
    class FloorClientOrder: public FloorBase {
        // FloorClientOrder act as an interface between 
        // Floor Manager and the FIX connections to venues
        // for sending orders and receiving execution reports. 
        //
        // Templated type OrderConnection needs to implement :
        // 1. errstr = sendOrder(const char* data)
        // 2. errstr = requestReplay(const char* data, int size)
        // 3. errstr = requestOpenOrder(const char* data, int size)

    public:
        explicit FloorClientOrder(const std::string& name, OrderConnection& conn)
        : FloorBase(name, false), m_conn(conn)
        {
            setSubscriptions();
        }

        ~FloorClientOrder(){};

        // update the floor manager whenever
        void sendMsg(const FloorBase::MsgType& msg) {
            m_channel->update(msg);
        }

        // out-going updates
        bool sendExecutionReport(const pm::ExecutionReport& er) {
            m_msgout.type = FloorBase::ExecutionReport;
            m_msgout.copyData((char*)&er, sizeof(ExecutionReport));
            m_channel->update(m_msgout);
            return true;
        }

        bool sendExecutionReplayDone() {
            m_msgout.type = FloorBase::ExecutionReplayDone;
            m_msgout.copyString("Ack");
            m_channel->update(m_msgout);
            return true;
        }

    protected:
        OrderConnection& m_conn;
        FloorClientOrder(const FloorClientOrder& mgr) = delete;
        FloorClientOrder& operator=(const FloorClientOrder& mgr) = delete;

        void handleMessage(MsgType& msg_in) {
            switch (msg_in.type) {
                case FloorBase::SendOrderReq: 
                {
                    handleOrderReq(msg_in);
                    break;
                }
                case FloorBase::ExecutionOpenOrderReq: 
                {
                    handleOpenOrderReq(msg_in);
                    break;
                }
                case FloorBase::ExecutionReplayReq: 
                {
                    handleExecutionReplayReq(msg_in);
                    break;
                }
                default :
                    logError("unkown requests %s", msg_in.toString().c_str());
            }
        }

        // incoming requests
        void setSubscriptions() {
            std::set<int> type_set;
            type_set.insert((int)FloorBase::SendOrderReq);
            type_set.insert((int)FloorBase::ExecutionOpenOrderReq);
            type_set.insert((int)FloorBase::ExecutionReplayReq);
            m_channel->addSubscription(type_set);
        }

        void handleOrderReq(const MsgType& msg) {
            const std::string errstr = m_conn.sendOrder(msg.buf);
            m_channel->updateAck(msg, m_msgout, FloorBase::SendOrderAck, errstr);
        }

        void handleOpenOrderReq(const MsgType& msg) {
            const std::string errstr = m_conn.requestOpenOrder(msg.buf, msg.data_size);
            m_channel->updateAck(msg, m_msgout, FloorBase::ExecutionOpenOrderAck, errstr);
        }

        void handleExecutionReplayReq(const MsgType& msg) {
            std::string ready_file;
            const std::string errstr = m_conn.requestReplay(msg.buf, msg.data_size, ready_file);
            m_channel->updateAck(msg, m_msgout, FloorBase::ExecutionReplayAck, errstr);

            // if replay has been ready, indicate to the floor manager
            // this is the case for CEngine in live trading, but could be
            // different in exch_mock, where the "Done" event is scheduled 
            // 2 seconds later.
            if ( ready_file.size() > 0) {
                m_msgout.type = FloorBase::ExecutionReplayDone;
                m_msgout.copyString(ready_file);
                m_channel->update(m_msgout);
            }
        }

        friend class FloorBase;
    };
};
