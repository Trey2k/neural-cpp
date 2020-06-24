cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(nerual)

# specify the C++ standard
set(CMAKE_BUILD_TYPE Debug)

# main app
add_executable(
    neual
    src/main.cpp
)
