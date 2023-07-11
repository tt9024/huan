# AlgoBase
add_library(algolib STATIC
    AlgoBase.cpp
    AlgoThread.cpp
    )

target_link_libraries(algolib PUBLIC floorlib rt pthread)
target_include_directories(algolib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    #${CMAKE_CURRENT_SOURCE_DIR}/../util
    ${CMAKE_CURRENT_SOURCE_DIR}/../md
    ${CMAKE_CURRENT_SOURCE_DIR}/../pm
    )

# All strategies as static libraries
# TODO consider making it  Shared
add_library(stratlib STATIC
    strat/AR1.cpp
    strat/idbo/idbo_tf.cpp
    )

target_link_libraries(stratlib PUBLIC algolib)
target_include_directories(stratlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/strat
    ${CMAKE_CURRENT_SOURCE_DIR}/strat/idbo
    )

add_executable(strat_run 
    ThreadMain.cpp
    )
target_link_libraries(strat_run PUBLIC rt pthread algolib stratlib)

add_executable(idbo_test test/test_idbo.cpp)
target_link_libraries(idbo_test PRIVATE gtest gtest_main rt pthread algolib stratlib)
set_target_properties(idbo_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

add_test(NAME idbo_test COMMAND 
    $<TARGET_FILE:idbo_test>
)

add_executable(floor_pos_test  test/test_base.cpp)
target_link_libraries(floor_pos_test PRIVATE gtest gtest_main rt pthread algolib)
set_target_properties(floor_pos_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

add_test(NAME floor_pos_test COMMAND 
    $<TARGET_FILE:floor_pos_test>
)

add_subdirectory(sim)
