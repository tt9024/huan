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

template <typename T, std::size_t Capacity = 256>
class DynamicBuffer
{
public:

    using size_type = std::size_t;
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    explicit
    DynamicBuffer(const char *buffer, size_type length) :
            size_(length),
            capacity_(length)
    {
        assert(size_ > 0  && "Invalid DynamicBuffer length supplied. (<= 0).");
        data_ = new T[capacity_]{};
        std::copy(buffer, buffer + capacity_, data_);
    }

    DynamicBuffer() = delete;

    ~DynamicBuffer()
    {
        delete[] data_;
    }

    DynamicBuffer(const DynamicBuffer& rhs)
    {
        ensure_capacity(rhs);

        std::copy(rhs.data_, rhs.data_ + capacity_, data_);
        size_ = rhs.size_;
    }

    DynamicBuffer& operator=(const DynamicBuffer& rhs)
    {
        if (this == &rhs)
            return *this;

        ensure_capacity(rhs);

        std::copy(rhs.data_, rhs.data_ + capacity_, data_);
        size_ = rhs.size_;

        return *this;
    }

    DynamicBuffer(DynamicBuffer&& rhs) noexcept
    {
        assert(capacity_ == rhs.capacity_ &&
               "No support to move DynamicBuffers of different sizes");

        std::swap(data_, rhs.data_);
        size_ = rhs.size_;

        delete[] rhs.data_;
        rhs.data_ = nullptr;
        rhs.size_ = 0;
        rhs.capacity_ = 0;
    }

    DynamicBuffer& operator=(DynamicBuffer&& rhs) noexcept
    {
        if (this == &rhs)
            return *this;

        assert(capacity_ == rhs.capacity_ &&
               "No support to move DynamicBuffers of different sizes");

        std::swap(data_, rhs.data_);
        size_ = rhs.size_;

        delete[] rhs.data_;
        rhs.data_ = nullptr;
        rhs.size_ = 0;
        rhs.capacity_ = 0;

        return *this;
    }

    pointer data() noexcept { return &data_[0]; }
    const_pointer data() const noexcept { return &data_[0]; }

    [[nodiscard]] auto size() const noexcept { return size_; }
    [[nodiscard]] auto empty() const noexcept { return size_ == capacity_; }

private:
    size_type size_{};
    size_type capacity_{};
    T *data_{};

    // Simple operation to ensure that during copy/move our
    // local capacity can withstand the operation
    void ensure_capacity(const DynamicBuffer& rhs)
    {
        assert(rhs.size_ > 0 && rhs.capacity_ >= size_ && "Invalid DynamicBuffer setup/copy.");

        if (rhs.capacity_ > capacity_) {
            delete[] data_;
            data_ = new T[rhs.capacity_]{};
            capacity_ = rhs.capacity_;
        }

        assert(size_ <= capacity_ && "Invalid size_ and capacity_. Check Buffers.");
    }

};

typedef DynamicBuffer<uint8_t> Buffer;

#endif //UTIL_FIXEDBUFFER_H
