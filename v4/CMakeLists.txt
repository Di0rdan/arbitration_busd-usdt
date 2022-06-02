cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(.)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main main.cpp)
target_link_libraries (main Eigen3::Eigen)

add_definitions(-D Model=ML_Simple)
