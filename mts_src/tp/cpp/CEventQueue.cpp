#include "CEventQueue.h"


using namespace Mts::Event;


CEventQueue::CEventQueue(unsigned int iBufferSizeBytes)
        : m_EventBuffer(iBufferSizeBytes),
          m_iPushCount(0),
          m_iPopCount(0) {

}


void CEventQueue::lock() {

    m_Mutex.lock();
}


void CEventQueue::unlock() {

    m_Mutex.unlock();
}


void CEventQueue::wait() {

    m_CondVar.wait(m_Mutex);
}


boost::condition_variable_any &CEventQueue::getCondVar() {

    return m_CondVar;
}


bool CEventQueue::isReadReady() {

    return !m_EventBuffer.isEmpty();
}


CEvent::EventID CEventQueue::getNextEventID() {

    Mts::Event::CEvent::EventID iEventID;
    m_EventBuffer.peek(iEventID);
    return iEventID;
}


void CEventQueue::reset() {

    boost::mutex::scoped_lock lock(m_Mutex);

    m_EventBuffer.initialize();
}
