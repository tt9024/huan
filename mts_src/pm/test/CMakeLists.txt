
# er_test
add_executable(er_test er_test.cpp)
target_link_libraries(er_test PRIVATE gtest gtest_main floorlib)
set_target_properties(er_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME er_test COMMAND $<TARGET_FILE:er_test>)

# idp_test
add_executable(idp_test idp_test.cpp)
target_link_libraries(idp_test PRIVATE gtest gtest_main floorlib)
set_target_properties(idp_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME idp_test COMMAND $<TARGET_FILE:idp_test>)

# pm_test
add_executable(pm_test pm_test.cpp)
target_link_libraries(pm_test PRIVATE gtest gtest_main floorlib)
set_target_properties(pm_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME pm_test COMMAND $<TARGET_FILE:pm_test>)

# risk_test
add_executable(risk_test risk_test.cpp)
target_link_libraries(risk_test PRIVATE gtest gtest_main floorlib)
set_target_properties(risk_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME risk_test COMMAND $<TARGET_FILE:risk_test> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

configure_file(risk/vs_5_15_60m.csv "${CMAKE_BINARY_DIR}/tests/vs_5_15_60m.csv")
configure_file(risk/risk.json "${CMAKE_BINARY_DIR}/tests/risk.json")
configure_file(risk/trading_status.csv "${CMAKE_BINARY_DIR}/tests/trading_status.csv")
configure_file(risk/STRATEGY_WEIGHTS.yaml "${CMAKE_BINARY_DIR}/tests/STRATEGY_WEIGHTS.yaml")

