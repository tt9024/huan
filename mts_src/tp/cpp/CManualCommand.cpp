#include "CManualCommand.h"

using namespace Mts::OrderBook;

CManualCommand::CManualCommand() {
    memset((void *) m_cmd_buffer, 0, sizeof(m_cmd_buffer));
}


CManualCommand::CManualCommand(const Mts::Core::CDateTime &dtTimestamp, const std::string &strCommand)
        : CEvent(MANUAL_COMMAND),
          m_dtTimestamp(dtTimestamp) {
    snprintf(m_cmd_buffer, sizeof(m_cmd_buffer), "%s", strCommand.c_str());
}


const Mts::Core::CDateTime &CManualCommand::getTimestamp() const {
    return m_dtTimestamp;
}


std::string CManualCommand::getCommand() const {
    return std::string(m_cmd_buffer);
}

std::string CManualCommand::toString() const {
    char szBuffer[1024];
    sprintf(szBuffer, "%llu %s", m_dtTimestamp.getCMTime(), m_cmd_buffer);
    return szBuffer;
}



