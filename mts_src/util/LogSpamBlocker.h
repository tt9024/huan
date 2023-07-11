//
// Created by joren on 3/9/23.
//

#ifndef KISCO_LOGSPAMBLOCKER_H
#define KISCO_LOGSPAMBLOCKER_H

#include <string>
#include <chrono>
#include <utility>
#include <unordered_map>

namespace com::massar::core {

    class LogSpamBlocker final
    {
    public:

        // Validate the message and last time seen (if >kSecondsToPrintSpam then print it)
        [[nodiscard]] bool validateLogSpam(const std::string &message) const
        {
            using namespace std::chrono;
            const auto now = high_resolution_clock::now();

            auto it = spam_blocker_map_.find(message);
            if (it != spam_blocker_map_.end()) {

                auto& [count, last_seen] = it->second;

                if (duration_cast<seconds>(now - last_seen) >= seconds(kSecondsToPrintSpam)) {

                    // Update count/last seen (printed) time
                    count++;
                    last_seen = now;

                    // Allow message
                    return true;
                }

                // Block message
                return false;
            }

            // First time message has been seen
            const auto [iter, inserted] = spam_blocker_map_.emplace(std::piecewise_construct,
                                                                    std::forward_as_tuple(message),
                                                                    std::forward_as_tuple(1, now));
            // Allow message
            return inserted;
        }

        // Cleanup the map (if last seen > kMinutesToKillSpam)
        void invalidateLogSpam()
        {
            using namespace std::chrono;

            const auto now = high_resolution_clock::now();

            auto it = spam_blocker_map_.begin();
            while (it != spam_blocker_map_.end()) {

                const auto& [count, last_seen] = it->second;
                if (duration_cast<minutes>(now - last_seen) >= minutes(kMinutesToKillSpam)) {
                    it = spam_blocker_map_.erase(it);
                } else {
                    ++it;
                }
            }
        }

    private:
        static inline const std::int32_t kSecondsToPrintSpam = 30;
        static inline const std::int32_t kMinutesToKillSpam = 30;

        // Log spam for event-loop dumps
        using spam_blocker_t = std::pair<std::int32_t, std::chrono::high_resolution_clock::time_point>;
        mutable std::unordered_map<std::string, spam_blocker_t> spam_blocker_map_{};
    };

}
#endif //KISCO_LOGSPAMBLOCKER_H
