//
// Created by Joren Gaucher on 1/25/23.
//

#ifndef BPIPE_ASYNCUDPSENDER_H
#define BPIPE_ASYNCUDPSENDER_H

#include <iostream>
#include <queue>

#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>

//#include "FixedBuffer.h"
#include "DynamicBuffer.h"

namespace utils {

    struct AbstractAsyncUDPSender
    {
        virtual void write_complete(std::size_t bytes) = 0;
    };

    template <typename Callback>
    class AsyncUDPSender {
    public:

        using size_type = std::size_t;

        AsyncUDPSender() = delete;

        explicit
        AsyncUDPSender(std::string host, std::int32_t port, Callback& callback) :
                host_(std::move(host)),
                port_(port),
                callback_(callback),
                io_context_(),
                socket_(io_context_)
        {
            using namespace boost::asio::ip;

            assert(port_ > 0 && port < 65535 && "AsyncUDPSender check port range.");

            udp::resolver resolver(io_context_);
            udp::resolver::iterator iter = resolver.resolve(host_, std::to_string(port_));
            endpoint_ = iter->endpoint();

            try {

                socket_.open(endpoint_.protocol());

                // Simple snd buffer size and reuse for quick-restarts
                socket_.set_option(udp::socket::reuse_address(true));
                socket_.set_option(udp::socket::send_buffer_size(1'048'576));

            } catch (boost::system::system_error &error) {
                std::cout << "Error: " << error.what() << std::endl;
                return;
            }
        }

        ~AsyncUDPSender()
        {
            io_context_.stop();
            socket_.close();
        }

        [[nodiscard]] bool poll() noexcept
        {
            bool flush_ret{};

            if (!empty()) [[likely]] {
                flush_ret = flush(batch_size_);
            }

            io_context_.poll_one();

            return flush_ret;
        }

        [[nodiscard]] bool send(const char *buffer, const size_type len)
        {
            assert(buffer && len > 0 && "Invalid write: check buffer and len.");
            write_queue_.emplace(buffer, len);
            return poll();
        }

        [[nodiscard]] auto size() const noexcept { return write_queue_.size(); }
        [[nodiscard]] auto empty() const noexcept { return write_queue_.empty(); }

        void async_send_to(const boost::system::error_code& error,
            std::size_t bytes_sent) noexcept
        {
            callback_.write_complete(bytes_sent);

            if (error) [[unlikely]] {
                std::cout << "Unable to async_send_to: " << error.message() << "\n";
                return;
            }

            //std::cout << "Write complete called: " << bytes_sent << '\n';
        }

        AsyncUDPSender(const AsyncUDPSender &rhs) = delete;
        AsyncUDPSender(AsyncUDPSender &&rhs) noexcept = delete;

        AsyncUDPSender &operator=(const AsyncUDPSender &rhs) = delete;
        AsyncUDPSender &operator=(AsyncUDPSender &&rhs) noexcept = delete;

    private:
        std::string host_{};
        std::int32_t port_{};
        Callback& callback_;

        constinit static inline std::int32_t batch_size_{1};

        // Use boost ASIO for simplicity and robustness
        boost::asio::io_context io_context_;
        boost::asio::ip::udp::socket socket_;
        boost::asio::ip::udp::endpoint endpoint_;

        //std::queue<SmallFixedBuffer> write_queue_{};
        std::queue<Buffer> write_queue_{};

        void poll_io()
        {
            io_context_.poll_one();
        }

        [[nodiscard]] bool flush(std::int32_t batch_count)
        {
            auto count = 0;
            while (!write_queue_.empty() && count++ <= batch_count) {

                auto& fixed_buffer = write_queue_.front();
                socket_.async_send_to(
                        boost::asio::buffer(fixed_buffer.data(), fixed_buffer.size()),
                        endpoint_,
                        boost::bind(&AsyncUDPSender::async_send_to, this,
                                    boost::asio::placeholders::error,
                                    boost::asio::placeholders::bytes_transferred));

                 write_queue_.pop();
            }

            // We need to ensure the service is restarted for poll()
            if (io_context_.stopped())
                io_context_.restart();

            return count == batch_count;
        }

    };

}   // namespace utils

#endif //BPIPE_ASYNCUDPSENDER_H
