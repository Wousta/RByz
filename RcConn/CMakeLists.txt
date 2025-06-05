cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

file(GLOB SRC "src/*.cpp")
file(GLOB HDR "include/*.hpp" "../../shared/redis.hpp" "../../shared/net.hpp" "../../shared/util.hpp" "../../shared/macros.hpp")

add_library(rc_conn SHARED ${SRC} ${HDR})

target_compile_options(rc_conn PRIVATE -g) # comment for release mode

target_include_directories(rc_conn PUBLIC ${nlohmann_json_INCLUDE_DIR})
target_include_directories(rc_conn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(rc_conn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../shared)

find_library(IBVERBS_LIBRARY ibverbs HINTS /usr/lib/x86_64-linux-gnu/)
find_library(RDMACM_LIBRARY rdmacm HINTS /usr/lib/x86_64-linux-gnu/)
target_link_libraries(rc_conn PUBLIC ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY})
target_link_libraries(rc_conn PUBLIC /usr/lib/x86_64-linux-gnu/libhiredis.so)