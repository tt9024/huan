#ifndef CTCPCONNECTION_HEADER

#define CTCPCONNECTION_HEADER

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

using boost::asio::ip::tcp;

namespace Mts {
    namespace Networking {
        class CTCPConnection : public boost::enable_shared_from_this<CTCPConnection>
        {
        public:
            static boost::shared_ptr<CTCPConnection> createConnection(boost::asio::execution_context &objIOService);

            tcp::socket &getSocket();

            std::string read();

            void write(int iData);

            void write(const std::string &strMessage);

            bool isAlive() const;

            void connectionTestAsyncRead();

            void handlerTestAsyncRead(const boost::system::error_code &iError,
                    std::size_t iBytesTransferred);

        private:
            CTCPConnection(boost::asio::execution_context &objIOService);

            void handle_read(const boost::system::error_code &iErr,
                    size_t iBytesTransferred);

            void handle_write(const boost::system::error_code &iErr,
                    size_t iBytesTransferred);

        private:
            enum
            {
                READ_BUFFER_LEN = 512
            };

            char m_szBuffer[READ_BUFFER_LEN];

            tcp::socket m_Socket;

            bool m_bLiveConnection;
        };
    }
}

#endif


