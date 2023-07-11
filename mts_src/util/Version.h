//
// Created by Joren Gaucher 01/20/23
// First time:  $ git tag "1.0.0"
//              $ git fetch --tags
// Do not modify: file is auto-generated

#ifndef MTS_ENGINE_VERSION_H
#define MTS_ENGINE_VERSION_H

extern "C"
{

constexpr const char *BUILD_TYPE = "";
constexpr const char *BUILD_DATE = __DATE__;
constexpr const char *BUILD_TIME = __TIME__;

constexpr const char *GIT_BRANCH = "repo";
constexpr const char *GIT_COMMIT_HASH = "13c0ce7";
constexpr const char *GIT_DATE = "2023-06-01 21:19:44 -0400";
constexpr const char *GIT_TAG = "1.0.0-148-g13c0ce7";

constexpr const char *MAJOR_VERSION = "1";
constexpr const char *MINOR_VERSION = "0";
constexpr const char *PATCH_VERSION = "0";

// nb: This could be just VERSION for any project
constexpr const char *MTS_VERSION = "1.0.0";

}

#endif // MTS_ENGINE_VERSION_H
