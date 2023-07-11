#include "FloorBase.h"

namespace pm {
    class FloorClientUser: public FloorBase {
        // FloorClientUser encapsulates user input, sends to 
        // FloorManager and output the response.
        // 

    public:
        explicit FloorClientUser(const std::string& name)
        : FloorBase(name, false), _prev_ref(utils::Floor::Message::NOREF)
        {}

        ~FloorClientUser() = default;

        // update the floor manager whenever
        std::string sendReq(const std::string& cmd) {
            _prev_reader = nullptr;
            _prev_ref = utils::Floor::Message::NOREF;
            switch (cmd[0]) {
            case '@':
            {
                // FloorBase::AlgoUserCommand
                m_msgin.type = FloorBase::AlgoUserCommand;
                m_msgin.copyString(cmd.c_str()+1);
                if ((m_channel->requestWithReader(m_msgin, m_msgout, _prev_reader, 5) && 
                    (m_msgout.type == FloorBase::AlgoUserCommandResp))) {
                    _prev_ref = m_msgout.ref;
                    return std::string(m_msgout.buf);
                } else {
                    _prev_reader = nullptr;
                    return "Algo Reqeust Timedout!";
                }
            }
            case 'X':
            {
                // create a PI from cmd+1, serialize to msg_buf and send it
                auto pi = std::make_shared<PositionInstruction>(cmd.c_str()+1);
                m_msgin.type = FloorBase::SetPositionReq;
                m_msgin.copyData((const char*)pi.get(), sizeof(PositionInstruction));
                if ((m_channel->requestWithReader(m_msgin, m_msgout, _prev_reader, 5) &&
                    (m_msgout.type == FloorBase::SetPositionAck))) {
                    _prev_ref = m_msgout.ref;
                    return std::string(m_msgout.buf);
                } else {
                    _prev_reader = nullptr;
                    return "SetPosition Reqeust Timedout!";
                }
            }
            case 'Z':
            {
                const auto& tk (utils::CSVUtil::read_line(cmd.c_str()+1));
                if (__builtin_expect(tk.size() == 3,1)) {
                    // set pause
                    // note there is no reference for this local msg
                    // no response is expected.
                    // don't use m_msgin since its reference could be set
                    pm::FloorBase::MsgType msg (
                            pm::FloorBase::TradingStatusNotice,
                            cmd.c_str(),
                            cmd.size()+1);
                    m_channel->update(msg);
                    return "Sent";
                }
                // fall through to UserReq for get
            }
            default:
            {
                // FloorBase::UserReq
                m_msgin.type = FloorBase::UserReq;
                m_msgin.copyString(cmd);
             if ((m_channel->requestWithReader(m_msgin, m_msgout, _prev_reader, 5) &&
                    (m_msgout.type == FloorBase::UserResp))) {
                    _prev_ref = m_msgout.ref;
                    return std::string(m_msgout.buf);
                } else {
                    _prev_reader = nullptr;
                    return "User Reqeust Timedout!";
                }
            }
            }
        }

        bool checkPrevResp(std::string& resp) {
            if ( !_prev_reader || (_prev_ref == utils::Floor::Message::NOREF) ) {
                return false;
            }
            if (m_channel->nextMessageRef(&m_msgout, _prev_reader, _prev_ref)) {
                resp = std::string(m_msgout.buf);
                return true;
            }
            return false;
        }

    protected:
        FloorClientUser(const FloorClientUser& mgr) = delete;
        FloorClientUser& operator=(const FloorClientUser& mgr) = delete;
        std::shared_ptr<utils::Floor::QType::Reader> _prev_reader;
        uint64_t _prev_ref;
    };
};

