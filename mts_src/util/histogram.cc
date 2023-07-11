//
// Created by Joren Gaucher 01/23/2023
//

#include <utility>
#include <cmath>
#include <bit>

#include "histogram.h"

Histogram::Histogram() noexcept:
        max_(INT64_MIN),
        min_(INT64_MAX),
        name_("Histogram")
{
    // nothing
}

Histogram::Histogram(std::string name) noexcept :
        count_(0),
        max_(INT64_MIN),
        mean_(0),
        min_(INT64_MAX),
        stddev_(0),
        cv_(0),
        name_(std::move(name))
{
    // nothing
}

Histogram::Histogram(Histogram&& rhs) noexcept :
        count_(rhs.count_),
        max_(rhs.max_),
        mean_(rhs.mean_),
        min_(rhs.min_),
        stddev_(rhs.stddev_),
        cv_(rhs.cv_),
        name_(std::move(rhs.name_))
{
    // nothing
}

Histogram& Histogram::operator=(Histogram&& rhs) noexcept
{
    if (this == &rhs)
        return *this;

    count_ = rhs.count_;
    max_ = rhs.max_;
    mean_ = rhs.mean_;
    min_ = rhs.min_;
    name_ = std::move(rhs.name_);
    stddev_ = rhs.stddev_;
    cv_ = rhs.cv_;

    for (auto i = 0; i < HISTO_BUCKET_SIZE; i++) {
        buckets_[i][1] = rhs.buckets_[i][1];
    }

    return *this;
}

Histogram& Histogram::operator=(const Histogram& rhs) noexcept
{
    if (this == &rhs)
        return *this;

    count_ = rhs.count_;
    max_ = rhs.max_;
    mean_ = rhs.mean_;
    min_ = rhs.min_;
    name_ = rhs.name_;
    stddev_ = rhs.stddev_;
    cv_ = rhs.cv_;

    for (auto i = 0; i < HISTO_BUCKET_SIZE; i++) {
        buckets_[i][1] = rhs.buckets_[i][1];
    }

    return *this;
}

inline double Histogram::fast_pow(double a, double b) noexcept
{
    union {
        double d;
        int x[2];
    } u = { a };

    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;

    return u.d;
}


//https://en.wikipedia.org/wiki/Fast_inverse_square_root
inline float Histogram::fast_sqrt(const float& n) noexcept
{
    static union{int32_t i; float f;} u;
    u.i = 0x5F375A86 - (*(int32_t*)&n >> 1);
    return (int(3) - n * u.f * u.f) * n * u.f * 0.5f;
}

inline constexpr float Histogram::fast_sqrt1(float number) noexcept
{
    static_assert(std::numeric_limits<float>::is_iec559); // (enable only on IEEE 754)
    auto y = std::bit_cast<float>(
            0x5f3759df - (std::bit_cast<std::uint32_t>(number) >> 1));
    return y * (1.5f - (number * 0.5f * y * y));
}

// nb: This function is bad news for front-end stalls on perf (too many divides)
void Histogram::add_value(const int64_t value) noexcept
{
    if (value > max_)
        max_ = value;

    if (value < min_)
        min_ = value;

    mean_ *= count_;
    mean_ += value;
    ++count_;
    mean_ /= count_;

    // Record dispersion around the updated mean
    var_ *= (float) (count_ - 1);
    var_ += (value-mean_)*(value-mean_);
    var_ /= (float) (count_ > 1 ? (count_ - 1) : 1);
    stddev_ = fast_sqrt(var_); //std::sqrt(var);
    cv_ = stddev_ / (float) mean_;

    add_bucket_val(value);
}

void Histogram::accumulate(int64_t mi, int64_t ma, int64_t m, int64_t c, int64_t b[][2]) noexcept
{
    count_ += c;

    if (ma > max_) {
        max_ = ma;
    }

    mean_ = (mean_ + m) / 2;
    if (mi < min_) {
        min_ = mi;
    }

    for (int32_t i = 0; i < HISTO_BUCKET_SIZE; i++)
        buckets_[i][1] += b[i][1];
}

inline void Histogram::add_bucket_val(const int64_t value) noexcept
{
    auto b = get_prev_pow2(value);
    buckets_[bucket_get_bucket(b)][1]++;
}

void Histogram::clear() noexcept
{
    count_ = 0;
    mean_ = 0;
    cv_ = 0.0;
    stddev_ = 0.0;

    max_ = INT64_MIN;
    min_ = INT64_MAX;

    for (auto& bucket : buckets_)
        bucket[1] = 0;
}


