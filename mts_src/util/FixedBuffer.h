//
// Created by joren on 1/25/23.
//

#ifndef UTIL_FIXEDBUFFER_H
#define UTIL_FIXEDBUFFER_H

#include <iostream>
#include <memory>
#include <cassert>
#include <cstring>
#include <type_traits>

template <typename T, std::size_t Capacity = 512>
class FixedBuffer
{
public:

    using size_type = std::size_t;
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    explicit
    FixedBuffer(const char *buffer, size_type length) :
        capacity_(Capacity),
        size_(length)
    {
        assert(size_ < capacity_ && "Increase FixedBuffer size: Too small");
        std::memcpy(data_, buffer, size_);
    }

    FixedBuffer() noexcept :
        size_{0},
        capacity_{Capacity}
    {}

    // Only enabling these copy+move for debug (compiler can generate them)
    FixedBuffer(const FixedBuffer& rhs)
    {
        assert(capacity_ == rhs.capacity_ &&
            "No support to copy FixedBuffers of different sizes");

        std::memcpy(data_, rhs.data_, capacity_);
        size_ = rhs.size_;
    }

    FixedBuffer& operator=(const FixedBuffer& rhs)
    {
        if (this == &rhs)
            return *this;

        assert(capacity_ == rhs.capacity_ &&
            "No support to copy FixedBuffers of different sizes");

        std::memcpy(data_, rhs.data_, capacity_);
        size_ = rhs.size_;
        return *this;
    }

    FixedBuffer(FixedBuffer&& rhs) noexcept
    {
        assert(capacity_ == rhs.capacity_ &&
            "No support to move FixedBuffers of different sizes");

        std::swap(data_, rhs.data_);
        size_ = rhs.size_;
    }

    FixedBuffer& operator=(FixedBuffer&& rhs) noexcept
    {
        if (this == &rhs)
            return *this;

        assert(capacity_ == rhs.capacity_ &&
            "No support to move FixedBuffers of different sizes");

        std::swap(data_, rhs.data_);
        size_ = rhs.size_;
        return *this;
    }

    pointer data() noexcept { return &data_[0]; }
    const_pointer data() const noexcept { return &data_[0]; }

    [[nodiscard]] auto size() const noexcept { return size_; }
    [[nodiscard]] auto empty() const noexcept { return size_ == capacity_; }

private:
    size_type size_{};
    size_type capacity_{Capacity};
    T data_[Capacity]{};
};

typedef FixedBuffer<char, 256> SmallFixedBuffer;
typedef FixedBuffer<char, 1024> MediumFixedBuffer;
typedef FixedBuffer<char, 4096> LargeFixedBuffer;

#endif //UTIL_FIXEDBUFFER_H
