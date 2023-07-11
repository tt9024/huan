#ifndef CDATETIMEEVENT_HEADER

#define CDATETIMEEVENT_HEADER

#include "CDateTime.h"
#include "CEvent.h"

namespace Mts {
    namespace Core {
        class CDateTimeEvent : public Mts::Event::CEvent
        {
        public:
            CDateTimeEvent()
                    : CEvent(Mts::Event::CEvent::DATETIME) {

            }

            CDateTimeEvent(const Mts::Core::CDateTime &dtTimestamp)
                    : CEvent(Mts::Event::CEvent::DATETIME),
                      m_dtTimestamp(dtTimestamp) {

            }

            const Mts::Core::CDateTime &getTimestamp() const;

            std::string toString() const;

        private:
            Mts::Core::CDateTime m_dtTimestamp;
        };
    }
}

#include "CDateTimeEvent.hxx"

#endif

