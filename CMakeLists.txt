cmake_minimum_required(VERSION 3.31)
project(DeZeroCpp)

set(CMAKE_CXX_STANDARD 20)

include_directories(thirdparty)

#add_executable(${PROJECT_NAME} src/main.cpp)

enable_testing()
add_subdirectory(test)