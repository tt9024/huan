
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Protobuf
#set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")

# Protobuf-compiler
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# gRPC
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_GRPCPP gRPC::grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# basic FloorCPR functions
add_library(floorlib STATIC
    ExecutionReport.cpp
    PositionData.cpp
    PositionManager.cpp
    RiskMonitor.cpp
    )

target_link_libraries(floorlib PUBLIC
        plcc
        protobuf::libprotobuf
        protocols
        grpc
        grpc++
        rt
        pthread
        )

# Include gRPC includes and .proto build .cc/.h
target_include_directories(floorlib PUBLIC ${gRPC_INCLUDE_DIR})
target_include_directories(floorlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../md)
target_include_directories(floorlib PUBLIC ${PROTOCOLS_SOURCE_DIR}/src/main/cpp/Proto)

# Floor manager
add_executable(floor
        FloorManagerMain.cpp
        FloorManager.cpp
)

target_link_libraries(floor PRIVATE
        floorlib
        grpc
        grpc++
        )

# Ensure floor builds with proto_headers (all generated gRPC services and .proto .cc/.h sources)
add_dependencies(floor protocol_headers)

# (DEBUG) Print the include directories to the console
#get_property(include_dirs TARGET floor PROPERTY INCLUDE_DIRECTORIES)
#message(STATUS "Include directories: ${include_dirs}")

# floor trader
# add akk execution traders here
add_executable(ftrader FloorTraderMain.cpp 
                       FloorTrader.cpp
                       TWAPTrader.cpp
    )
target_link_libraries(ftrader PRIVATE floorlib)

# flr
add_executable(flr flr.cpp)
target_link_libraries(flr PRIVATE
        floorlib
      )

# floortap
add_executable(ftap floortap.cpp)
target_link_libraries(ftap PRIVATE floorlib)

# floorpoke
add_executable(fpoke floorpoke.cpp)
target_link_libraries(fpoke PRIVATE floorlib)

# exch_mock
add_executable(exch_mock exch_mock.cpp)
target_link_libraries(exch_mock PRIVATE floorlib)

## TESTS
add_subdirectory(test)

