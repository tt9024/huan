#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef __USE_GNU
#define __USE_GNU
#endif

#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <atomic>

namespace utils {
    /// a template for a Runnable
    class Runnable {
    public:
        void run(void* para);
        void stop();
    };

    template<typename RunnableType>
    class ThreadWrapper {
    public:
        // note runnable has to live in its own scope. 
        ThreadWrapper(RunnableType& runnable) : m_runnable(runnable), m_param(NULL), m_isRunning(false), m_thread(0) {};
        void run(void* para) {
            if (!m_isRunning) {
                m_param = para;
                int ret = pthread_create( &m_thread, NULL, ThreadWrapper<RunnableType>::threadFunc, (void*) this);
                if (ret != 0) {
                    throw std::runtime_error(std::string("pthread creation error! errno=") + std::to_string(errno));
                }
                m_isRunning = true;
            }
        }
        void join() {
            if (m_isRunning)
                pthread_join(m_thread, NULL);
        }
        void stop() {
            if (m_isRunning) {
                m_runnable.stop();
                m_isRunning = false;
            };
        }

        void setAffinity(int cpuID) {
              cpu_set_t mask;
              CPU_ZERO(&mask);
              CPU_SET(cpuID, &mask);
              if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
                  throw std::runtime_error(std::string("set thread affinity error! errno=") + std::to_string(errno));
              }
        }

        ~ThreadWrapper() {
            if (m_isRunning) {
                m_runnable.stop();
                //sleep 100 millisecond and wait any pending loops
                struct timespec tspec;
                tspec.tv_sec = 0;
                tspec.tv_nsec = 100000000 ;
                nanosleep(&tspec, &tspec);
            }

            if (m_thread)
                pthread_cancel(m_thread);
        }

        // getters
        RunnableType& getRunnable() const { return m_runnable; };
        void* getRunParam() const { return m_param; };

    private:
        RunnableType& m_runnable;
        void* m_param;
        bool m_isRunning;
        pthread_t m_thread;
        static void* threadFunc(void* para) {
            ThreadWrapper<RunnableType> *wrapper = (ThreadWrapper<RunnableType>*)para;
            wrapper->getRunnable().run(wrapper->getRunParam());
            return (void*)NULL;
        }
    };

    /*
     * Context scope spin lock, exception safe. 
     * Lock is obtained upon construction and is released upon destruction. 
     * The input parameter "LockType& lock" is assumed to be a member variable
     * or a temparary whose life span is longer than the SpinLock object. 
     * Examples refer to the unittests.
     */
    class SpinLock {
    public:
        using LockType = std::atomic<bool>;
        explicit SpinLock(LockType& lock) noexcept 
        :_lock(&lock)
        {
            acquire();
        }

        ~SpinLock() {
            release();
        }

    private:
        void acquire() noexcept {
            while (1) {
                if (!_lock->exchange(true, std::memory_order_acquire)) {
                    break;
                }
                while (_lock->load(std::memory_order_relaxed)) {
                    __builtin_ia32_pause();
                }
            }
        }

        void release() noexcept {
            _lock->store(false, std::memory_order_release);
        }


        LockType* _lock;

        // no copy or assign
        SpinLock(const SpinLock&) = delete;
        void operator=(const SpinLock&) = delete;
    };
}
