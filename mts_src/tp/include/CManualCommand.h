#pragma once

#include <cstring>
#include "CEvent.h"
#include "CDateTime.h"

namespace Mts
{
    namespace OrderBook
    {
        class CManualCommand : public Mts::Event::CEvent
        {
        public:
            CManualCommand();
            CManualCommand(const Mts::Core::CDateTime & dtTimestamp,  const std::string & command);
            
            const Mts::Core::CDateTime & getTimestamp() const;
            std::string toString() const;
            std::string getCommand() const;
        private:
            Mts::Core::CDateTime m_dtTimestamp;
            char m_cmd_buffer[256];
        };
    }
}
