#ifndef IRUNNABLE_HEADER

#define IRUNNABLE_HEADER

namespace Mts {
    namespace Thread {
        class IRunnable
        {
        public:
            virtual void run() = 0;

            virtual void operator()() = 0;
        };
    }
}

#endif

