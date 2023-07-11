#ifndef CTCPENGINECONTROLLER_HEADER

#define CTCPENGINECONTROLLER_HEADER

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/thread/thread.hpp>
#include <vector>
#include "CTCPConnection.h"

using boost::asio::ip::tcp;

namespace Mts {
    namespace Engine {
        class CEngine;
    }
}

namespace Mts {
    namespace Networking {
        class CTCPEngineController
        {
        public:
            CTCPEngineController(unsigned int iEngineControlPort,
                    boost::shared_ptr<Mts::Engine::CEngine> ptrEngine);

            void startServer();

            void stopServer();

            void runIOService();

            void start_accept();

            void handle_accept(boost::shared_ptr<CTCPConnection> ptrConnection,
                    const boost::system::error_code &objError);

        private:
            void executeCommand(const std::string &strCommand);

        private:
            boost::shared_ptr<Mts::Engine::CEngine> m_ptrEngine;
            boost::asio::io_service m_IOService;
            tcp::acceptor m_Acceptor;
            boost::mutex m_MutexEngine;
        };
    }
}

#endif

