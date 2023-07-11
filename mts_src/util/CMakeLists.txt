add_library(plcc STATIC
        time_util.cpp
        symbol_map.cpp
        rate_limiter.cpp
        rate_estimator.cpp
        plcc/ConfigureReader.cpp
        plcc/PLCC.cpp
)

# Expose includes
target_include_directories(plcc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)
target_include_directories(plcc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/util)
target_include_directories(plcc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/util/plcc)
target_include_directories(plcc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/plcc)

add_executable (cfgupd
        plcc/ConfigUtil.cpp)

target_link_libraries(cfgupd
        plcc
        rt
)

add_executable (json_upd
        plcc/JsonUtil.cpp
)

target_link_libraries(json_upd
        plcc
        rt
)

set_property(TARGET plcc PROPERTY COMPILE_OPTIONS -Wno-error)
#set_property(TARGET plcc APPEND PROPERTY COMPILE_OPTIONS -Wno-unused-parameter)
#set_property(TARGET plcc APPEND PROPERTY COMPILE_OPTIONS -Wno-sign-compare)
#set_property(TARGET plcc APPEND PROPERTY COMPILE_OPTIONS -Wno-depreciated-declarations)

add_subdirectory(test)
