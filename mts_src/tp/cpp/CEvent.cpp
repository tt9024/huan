#include "CEvent.h"


using namespace Mts::Event;


CEvent::CEvent()
        : m_iEventID(UNDEFINED) {

}


CEvent::CEvent(EventID iEventID)
        : m_iEventID(iEventID) {

}


CEvent::EventID CEvent::getEventID() const {

    return m_iEventID;
}


void CEvent::setEventID(EventID iEventID) {

    m_iEventID = iEventID;
}

