cmake_minimum_required(VERSION 3.26)
project(Lab1)

set(CMAKE_CXX_STANDARD 20)

# Add the -O2 optimization flag
add_compile_options(-O2)

add_executable(Lab1 main.cpp)
