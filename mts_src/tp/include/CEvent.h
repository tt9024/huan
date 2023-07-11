#ifndef CEVENT_HEADER

#define CEVENT_HEADER

namespace Mts {
    namespace Event {
        class CEvent
        {
        public:
            enum EventID
            {
                UNDEFINED,
                CONSOLIDATED_ORDER_BOOK,
                TRADE,
                ORDER_STATUS,
                FILL,
                QUOTE,
                BID_ASK,
                DATETIME,
                EXEC_REPORT,
                KEY_VALUE,
                MANUAL_COMMAND
            };

        public:
            CEvent();

            CEvent(EventID iEventID);

            CEvent::EventID getEventID() const;

            void setEventID(EventID iEventID);

        private:
            EventID m_iEventID;
        };
    }
}

#endif


