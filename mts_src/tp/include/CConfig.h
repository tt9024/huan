#ifndef CCONFIG_HEADER

#define CCONFIG_HEADER

#include <string>

#define MTSConfigInstance Mts::Core::CConfig::getInstance()
#define MTSConfigFile(f) MTSConfigInstance.getConfigFile(f)

namespace Mts {
    namespace Core {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(__WIN32) && !defined(__CYGWIN__)
        static const char PATH_SLASH = '\\';
#else
        static const char PATH_SLASH = '/';
#endif

        static inline std::string RemoveEndSlashPath(const std::string &path) noexcept;

        static inline std::string RemoveAllSlashFile(const std::string &file) noexcept;

        static inline std::string PathJoin(const std::string &path, const std::string &file) noexcept;

        static const std::string ENGINE_CONFIG_FILE("engine.xml");
        static const std::string CONFIG_PATH("config/legacy");
        static const std::string ALGO_PATH("algo");
        static const std::string CALENDAR_PATH("calendars");
        static const std::string DROPCOPY_FIX_CONFIG("fix_dropcopy_config.txt");
        static const std::string DROPCOPY_SESSION_CONFIG("dropcopy_TT.xml");

        class CConfig
        {
        public:
            static CConfig &getInstance();

            CConfig() = default;

            CConfig(const CConfig &objRhs) = delete;

            const std::string &getLogFileDirectory() const;

            unsigned int getLogBufferSize() const;

            unsigned int getLogIntervalSec() const;

            const std::string &getSQLServer() const;

            const std::string &getSQLDatabase() const;

            const std::string &getSMTPServer() const;

            const std::string &getFromEmailAddr() const;

            const std::string &getToEmailAddr() const;

            bool isEmailEnabled() const;

            unsigned int getEngineID() const;

            const std::string &getRecoveryFileDirectory() const;

            const std::string getConfigFile(const std::string &filename) const;

            const std::string getEngineConfigFile() const;

            const std::string getAlgoConfigFile(const std::string &algoname) const;

            const std::string getCalendarFile(const std::string &filename) const;

            const std::string getDropCopyFIXConfigFile() const;

            const std::string getDropCopySessionConfigFile() const;

            void setEnginePath(const std::string &path);

            void setLogFileDirectory(const std::string &strLogFileDirectory);

            void setLogBufferSize(unsigned int iBufferSize);

            void setLogIntervalSec(unsigned int iIntervalSecs);

            void setSQLServer(const std::string &strSQLServer);

            void setSQLDatabase(const std::string &strSQLDatabase);

            void setSMTPServer(const std::string &strSMTPServer);

            void setFromEmailAddr(const std::string &strFromEmailAddr);

            void setToEmailAddr(const std::string &strToEmailAddr);

            void setEnableEmailer(bool bEnableEmailer);

            void setEngineID(unsigned int iEngineID);

            void setRecoveryFileDirectory(const std::string &strRecoveryFileDirectory);

        public:
            enum
            {
                MAX_NUM_ALGOS = 60, MAX_NUM_SYMBOLS = 250, MAX_NUM_PROVIDERS = 25, MAX_NUM_CCYS = 50
            };

        private:
            // logging
            std::string m_strLogFileDirectory;
            unsigned int m_iLogFileBufferSize;
            unsigned int m_iLogFileLogIntervalSecs;
            std::string m_strRecoveryFileDirectory;

            // SQL database
            std::string m_strSQLServer;
            std::string m_strSQLDatabase;

            // email
            std::string m_strSMTPServer;
            std::string m_strFromEmailAddr;
            std::string m_strToEmailAddr;
            bool m_bEnableEmailer;

            // engine ID
            unsigned int m_iEngineID;

            // engine paths
            std::string m_engine_path;
            std::string m_config_path;
        };
    }
}


namespace Mts {
    namespace Core {
        // inline function definition
        static inline std::string RemoveEndSlashPath(const std::string &path) noexcept {
            std::string path_(path);
            size_t sz = path_.size();
            while ((path_.back() == Mts::Core::PATH_SLASH) && (sz >= 2)) {
                if (path_[sz - 2] != ':') {
                    // last slash of C:\\ should be retained
                    path_.pop_back();
                    sz -= 1;
                } else {
                    break;
                };
            }
            return path_;
        }

        static inline std::string RemoveAllSlashFile(const std::string &file) noexcept {
            std::string filename = RemoveEndSlashPath(file);
            size_t sz = filename.size();
            while ((sz > 0) && (filename.front() == Mts::Core::PATH_SLASH)) {
                filename = filename.substr(1, sz - 1);
                sz -= 1;
            }
            return filename;
        }

        static inline std::string PathJoin(const std::string &path, const std::string &file) noexcept {
            const std::string pathname = RemoveEndSlashPath(path);
            if (pathname.size() == 0) {
                return RemoveEndSlashPath(file);
            }
            const std::string filename = RemoveAllSlashFile(file);
            std::string slash = std::string(1, Mts::Core::PATH_SLASH);
            if ((pathname.back() == Mts::Core::PATH_SLASH) ||
                (filename.size() == 0)) {
                slash = "";
            }
            return pathname + slash + filename;
        }
    }
}
#endif

