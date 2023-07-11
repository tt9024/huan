#ifndef CEVENTQUEUE_HEADER

#define CEVENTQUEUE_HEADER

#include <boost/thread/mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include "CEvent.h"
#include "CCircCharBuffer.h"

namespace Mts {
    namespace Event {
        class CEventQueue
        {
        public:
            CEventQueue(unsigned int iBufferSizeBytes);

            void lock();

            void unlock();

            void wait();

            boost::condition_variable_any &getCondVar();

            bool isReadReady();

            template<typename T>
            bool push(const T &objEvent);

            template<typename T>
            void pop(T &objEvent);

            CEvent::EventID getNextEventID();

            void reset();

        private:
            // circular buffer containing different events
            Mts::Core::CCircCharBuffer m_EventBuffer;
            unsigned int m_iPushCount;
            unsigned int m_iPopCount;

            boost::mutex m_Mutex;
            boost::condition_variable_any m_CondVar;
        };
    }
}

#include "CEventQueue.hxx"

#endif

