cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

file(GLOB SRC "src/*.cpp")
file(GLOB HDR "include/*.hpp" "../shared/net.hpp"
"../shared/util.hpp")

add_library(norm SHARED ${SRC} ${HDR})

target_compile_options(norm PRIVATE -g)

target_include_directories(norm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(norm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_include_directories(norm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../normal/include)
target_include_directories(norm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../conn/include)

find_library(IBVERBS_LIBRARY ibverbs HINTS /usr/lib/x86_64-linux-gnu/)
find_library(RDMACM_LIBRARY rdmacm HINTS /usr/lib/x86_64-linux-gnu/)
target_link_libraries(norm PUBLIC ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY})
target_link_libraries(norm PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so)