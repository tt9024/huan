//
// Joren Gaucher - Histogram class for recording stats
//

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <cstdio>
#include <cassert>
#include <cstdint>
#include <string>

#define HISTO_BUCKET_SIZE 12
#define HISTO_BUCKET_COL_SIZE 2
#define SCALED_HISTO_BUCKET_SIZE 100

class Histogram {
public:
    explicit
        Histogram(std::string name) noexcept;

    Histogram() noexcept;
    ~Histogram() = default;

    Histogram(Histogram&& rhs) noexcept;
    Histogram& operator=(Histogram&& rhs) noexcept;

    Histogram(const Histogram&& rhs) = delete;
    Histogram& operator=(const Histogram& rhs) noexcept;

    void add_value(int64_t value) noexcept;
    inline void add_bucket_val(int64_t value) noexcept;
    void accumulate(int64_t mi, int64_t ma, int64_t m, int64_t c, int64_t b[][2]) noexcept;
    void set_name(std::string _name) noexcept { name_ = std::move(_name); }
    void clear() noexcept;
    void accumulate_count(int64_t c) noexcept { count_ += c; }

    [[nodiscard]] static constexpr int64_t get_next_pow2(const int64_t value) noexcept
    {
        auto n = value;

        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        n++;

        return n;
    }

    [[nodiscard]] static constexpr int64_t get_prev_pow2(const int64_t value) noexcept
    {
        auto n = value;

        n |= n >> 1;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;

        return n - (n >> 1);
    }

    [[nodiscard]] auto get_count() const noexcept { return count_; }
    [[nodiscard]] auto get_max() const noexcept { return max_; }
    [[nodiscard]] auto get_mean() const noexcept { return mean_; }
    [[nodiscard]] auto get_min() const noexcept { return min_; }
    [[nodiscard]] auto get_std() const noexcept { return stddev_; }
    [[nodiscard]] auto get_cv() const noexcept { return cv_; }
    [[nodiscard]] const std::string& get_name() const noexcept { return name_; }

    [[nodiscard]] static constexpr int32_t bucket_get_bucket(const int64_t value) noexcept
    {
        if (value <= 0)
            return 10;

        if (value <= 256)
            return 0;
        else if (value == 512)
            return 1;
        else if (value == 1024)
            return 2;
        else if (value == 2048)
            return 3;
        else if (value == 4096)
            return 4;
        else if (value == 8192)
            return 5;
        else if (value == 16384)
            return 6;
        else if (value == 32768)
            return 7;
        else if (value == 65536)
            return 8;
        else if (value == 131072)
            return 9;
        else if (value >= 262144)
            return 10;

        return 10;
    };

    void logline(char *const buf, const size_t len) noexcept
    {
        assert(buf != nullptr && len >= 512);

        snprintf(buf, len, "[%s][<=%ld]=%ld, [%ld]=%ld, [%ld]=%ld, [%ld]=%ld, [%ld]=%ld, [%ld]=%ld, [%ld]=%ld, [%ld]=%ld, "
            "[%ld]=%ld, [%ld]=%ld, [>=%ld]=%ld, mean=%ld, std=%.2f, cv=%.2f",
            name_.c_str(),
            buckets_[0][0], buckets_[0][1],
            buckets_[1][0], buckets_[1][1],
            buckets_[2][0], buckets_[2][1],
            buckets_[3][0], buckets_[3][1],
            buckets_[4][0], buckets_[4][1],
            buckets_[5][0], buckets_[5][1],
            buckets_[6][0], buckets_[6][1],
            buckets_[7][0], buckets_[7][1],
            buckets_[8][0], buckets_[8][1],
            buckets_[9][0], buckets_[9][1],
            buckets_[10][0], buckets_[10][1],
            mean_, stddev_, cv_);
    }

    std::string logline() noexcept
    {
        std::string log_line{};

        log_line += "[" + name_ + "]";
        log_line += "[<=" + std::to_string(buckets_[0][0]) + "]=" + std::to_string(buckets_[0][1]);
        log_line += ", [" + std::to_string(buckets_[1][0]) + "]=" + std::to_string(buckets_[1][1]);
        log_line += ", [" + std::to_string(buckets_[2][0]) + "]=" + std::to_string(buckets_[2][1]);
        log_line += ", [" + std::to_string(buckets_[3][0]) + "]=" + std::to_string(buckets_[3][1]);
        log_line += ", [" + std::to_string(buckets_[4][0]) + "]=" + std::to_string(buckets_[4][1]);
        log_line += ", [" + std::to_string(buckets_[5][0]) + "]=" + std::to_string(buckets_[5][1]);
        log_line += ", [" + std::to_string(buckets_[6][0]) + "]=" + std::to_string(buckets_[6][1]);
        log_line += ", [" + std::to_string(buckets_[7][0]) + "]=" + std::to_string(buckets_[7][1]);
        log_line += ", [" + std::to_string(buckets_[8][0]) + "]=" + std::to_string(buckets_[8][1]);
        log_line += ", [" + std::to_string(buckets_[9][0]) + "]=" + std::to_string(buckets_[9][1]);
        log_line += ", [" + std::to_string(buckets_[10][0]) + "]=" + std::to_string(buckets_[10][1]);
        log_line += ", mean=" + std::to_string(mean_);
        log_line += ", std=" + std::to_string(stddev_);
        log_line += ", cv=" + std::to_string(cv_);

        return log_line;
    }

private:
    std::int64_t count_{};
    std::int64_t max_{};
    std::int64_t mean_{};
    std::int64_t min_{};

    float var_{};
    float stddev_{};
    float cv_{};

    std::string name_{};

    int64_t buckets_[HISTO_BUCKET_SIZE][HISTO_BUCKET_COL_SIZE] = {
            {256, 0},
            {512, 0},
            {1024, 0},
            {2048, 0},
            {4096, 0},
            {8192, 0},
            {16384, 0},
            {32768, 0},
            {65536, 0},
            {131072, 0},
            {262144, 0},
            {524288, 0}
    };

    static inline double fast_pow(double a, double b) noexcept;
    static inline float fast_sqrt(const float& n) noexcept;
    static inline float constexpr fast_sqrt1(float number) noexcept;
};


#endif //HISTOGRAM_H
