find_library(blpapi_location  NAMES libblpapi3_64.so)
message(STATUS ${blpapi_location})
add_library(blpapi SHARED IMPORTED)
set_target_properties(blpapi PROPERTIES IMPORTED_LOCATION ${blpapi_location})

# The blpapi shared object currently does not define an "soname", and we
# need to inform cmake of that. Otherwise, the linker will use the file
# path to the shared object as the name in the dependency list, which will
# later on cause the loader to exclusively use that file path and disregard
# 'LD_LIBRARY_PATH' (this is the behavior of the loader when it sees a '/'
# in the name of a dependency).
#set_property(TARGET blpapi PROPERTY IMPORTED_NO_SONAME TRUE)

add_executable(bpmain bbtp.cpp bbmain.cpp)
target_link_libraries(bpmain PRIVATE blpapi plcc rt pthread)
target_include_directories(bpmain PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../util
    ${CMAKE_CURRENT_SOURCE_DIR}/../../md
    )

add_executable(bptest bbtp_test.cpp bbmain.cpp)
target_link_libraries(bptest PRIVATE blpapi plcc rt pthread)
target_include_directories(bptest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../util
    ${CMAKE_CURRENT_SOURCE_DIR}/../../md
    )


