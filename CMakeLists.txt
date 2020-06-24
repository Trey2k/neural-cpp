cmake_minimum_required(VERSION 3.7)

# set the project name and version
project(nerualcpp)

# specify the C++ standard
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")

link_directories(${CMAKE_SOURCE_DIR}/dep/opennn/opennn)

include_directories(${CMAKE_SOURCE_DIR}/dep/opennn/opennn)

# main app
add_executable(
    neualcpp
    src/main.cpp
)

target_link_libraries(neualcpp opennn)
