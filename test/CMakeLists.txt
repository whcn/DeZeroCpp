project (unittest)

file(GLOB_RECURSE CPP_FILES ${CMAKE_SOURCE_DIR}/test/*.cpp)

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/thirdparty
        ${CMAKE_SOURCE_DIR}/thirdparty/googletest/include/)

link_directories(${CMAKE_SOURCE_DIR}/thirdparty/googletest/lib)

add_executable(${PROJECT_NAME} ${CPP_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE gtest)