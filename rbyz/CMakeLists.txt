project(RByz)
cmake_minimum_required(VERSION 3.22.1)
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cache.cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")

# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#   message(STATUS "Enabling AddressSanitizer flags...")
#   add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
#   add_link_options(-fsanitize=address -fno-omit-frame-pointer -g)
# endif()

find_package(lyra REQUIRED)

add_library(libnorm SHARED IMPORTED)
set_target_properties(libnorm PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  IMPORTED_NO_SONAME TRUE)
add_library(subsetSampler INTERFACE)
add_library(registeredMNIST INTERFACE)
add_library(mnistTrain STATIC src/mnistTrain.cpp)
add_library(rdmaOps STATIC src/rdmaOps.cpp)
add_library(tensorOps STATIC src/tensorOps.cpp)
add_library(attacks STATIC src/attacks.cpp)
add_library(rbyzAux STATIC src/rbyzAux.cpp)
add_executable(srvr src/srvr.cpp)
add_executable(clnt src/clnt.cpp)

target_compile_options(srvr PRIVATE -g)
target_compile_options(clnt PRIVATE -g)
target_compile_options(attacks PRIVATE -g)

target_include_directories(mnistTrain PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(subsetSampler INTERFACE 
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(registeredMNIST INTERFACE 
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(rdmaOps PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(tensorOps PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(attacks PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(rbyzAux PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(srvr PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(clnt PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(srvr PUBLIC ${TORCH_INCLUDE_DIRS})

target_precompile_headers(mnistTrain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hpp)
target_precompile_headers(tensorOps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hpp)
target_precompile_headers(attacks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hpp)
target_precompile_headers(rbyzAux PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hpp)
target_precompile_headers(srvr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hpp)
target_precompile_headers(clnt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hpp)

target_link_libraries(rdmaOps PUBLIC libnorm)

target_link_libraries(mnistTrain PUBLIC 
  ${TORCH_LIBRARIES}
  registeredMNIST
  subsetSampler
)

target_link_libraries(attacks PUBLIC 
  ${TORCH_LIBRARIES}
  mnistTrain
)

target_link_libraries(rbyzAux PUBLIC 
  ${TORCH_LIBRARIES}
  rdmaOps
)

target_link_libraries(tensorOps PUBLIC ${TORCH_LIBRARIES})

target_link_libraries(srvr PUBLIC  
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  ${TORCH_LIBRARIES}
  mnistTrain  
  rdmaOps
  tensorOps
  attacks
  rbyzAux
)

target_link_libraries(clnt PUBLIC 
  rdmaOps
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  ${TORCH_LIBRARIES}
  mnistTrain
  tensorOps
)