using namespace Mts::Event;

template<typename T>
bool CEventQueue::push(const T &objEvent) {

    boost::mutex::scoped_lock lock(m_Mutex);

    bool bRet = m_EventBuffer.push(objEvent);
    ++m_iPushCount;

    m_CondVar.notify_one();

    return bRet;
}


template<typename T>
void CEventQueue::pop(T &objEvent) {

    m_EventBuffer.pop(objEvent);
    ++m_iPopCount;
}



