
add_executable(time_test time_test.cpp)
target_link_libraries(time_test PRIVATE gtest gtest_main rt pthread plcc)
set_target_properties(time_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME time_test COMMAND 
    $<TARGET_FILE:time_test>
)

add_executable(csv_test  csv_test.cpp)
target_link_libraries(csv_test PRIVATE gtest gtest_main rt pthread plcc)
set_target_properties(csv_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME csv_test COMMAND 
    $<TARGET_FILE:csv_test>
)

add_executable(rate_test  rate_test.cpp)
target_link_libraries(rate_test PRIVATE gtest gtest_main rt pthread plcc)
set_target_properties(rate_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME rate_test COMMAND 
    $<TARGET_FILE:rate_test>
)

add_executable(config_test  config_test.cpp)
target_link_libraries(config_test PRIVATE plcc gtest gtest_main rt pthread)
set_target_properties(config_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME config_test COMMAND 
    $<TARGET_FILE:config_test>
)

add_executable(symmap_test  symbol_map_test.cpp)
target_link_libraries(symmap_test PRIVATE plcc gtest gtest_main rt pthread)
set_target_properties(symmap_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME symmap_test COMMAND 
    $<TARGET_FILE:symmap_test>
)

### this needs more than 2 CPU
#add_executable(thread_test  thread_test.cpp)
#target_link_libraries(thread_test PRIVATE plcc gtest gtest_main rt pthread)
#set_target_properties(thread_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
#add_test(NAME thread_test COMMAND 
#    $<TARGET_FILE:thread_test>
#)

# qtest is a manual test
add_executable(qtest qtest_shm.cpp)
target_include_directories(qtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qtest PRIVATE rt pthread plcc)

# cfg_reader is a manual test
add_executable(cfg_reader cfg_reader.cpp)
target_include_directories(cfg_reader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cfg_reader PRIVATE rt pthread plcc)

# plcc_log is a manual test
add_executable(plcc_log plcc_test.cpp)
target_include_directories(plcc_log PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(plcc_log PRIVATE rt pthread plcc)

# OnlyMe is a manual test
add_executable(onlyme onlyme.cpp)
target_include_directories(onlyme PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(onlyme PRIVATE rt pthread plcc)

