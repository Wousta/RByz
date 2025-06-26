cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

project(profiler LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../shared)
find_package(Threads REQUIRED)
add_executable(profiler src/profiler.cpp)
target_link_libraries(profiler PUBLIC 
  Threads::Threads
)

# Set output directory to build folder
set_target_properties(profiler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)