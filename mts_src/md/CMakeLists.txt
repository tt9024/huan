include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../util)

# booktap 
add_executable(booktap snap_reader.cpp)
target_link_libraries(booktap PRIVATE plcc rt pthread)

#target_include_directories(booktap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# bar_reader
add_executable(bar_reader bar_reader.cpp)
target_link_libraries(bar_reader PRIVATE plcc rt pthread)

#target_include_directories(bar_reader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# td_parser
add_executable(td_parser td_parser.cpp)
target_link_libraries(td_parser PRIVATE plcc rt pthread)

#target_include_directories(td_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# tick_parser
add_executable(tick_parser tick_parser.cpp)
target_link_libraries(tick_parser PRIVATE plcc rt pthread)

#target_include_directories(td_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

## test
add_subdirectory(test)

