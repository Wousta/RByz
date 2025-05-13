cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-9" CACHE STRING "C++ compiler path")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE STRING "CUDA compiler path")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

project(RByz LANGUAGES CXX CUDA)
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cache.cmake)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(Torch_DIR "/home/bustaman/libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#   message(STATUS "Enabling AddressSanitizer flags...")
#   add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
#   add_link_options(-fsanitize=address -fno-omit-frame-pointer -g)
# endif()

find_package(Threads REQUIRED)
find_package(lyra REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../shared
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(libnorm SHARED IMPORTED)
set_target_properties(libnorm PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  IMPORTED_NO_SONAME TRUE)

add_library(torch_interface INTERFACE)
target_include_directories(torch_interface INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(torch_interface INTERFACE ${TORCH_LIBRARIES})

add_library(subsetSampler INTERFACE)
add_library(registeredMNIST INTERFACE)
add_library(baseMnistTrain STATIC src/datasetLogic/baseMnistTrain.cpp)
add_library(regularMnistTrain STATIC src/datasetLogic/regularMnistTrain.cpp)
add_library(registeredMnistTrain STATIC src/datasetLogic/registeredMnistTrain.cpp)
add_library(rdmaOps STATIC src/rdmaOps.cpp)
add_library(tensorOps STATIC src/tensorOps.cpp)
add_library(attacks STATIC src/attacks.cpp)
add_library(rbyzAux STATIC src/rbyzAux.cpp)
add_executable(srvr src/entities/srvr.cpp)
add_executable(clnt src/entities/clnt.cpp)

target_compile_options(srvr PRIVATE -g)
target_compile_options(clnt PRIVATE -g)
target_compile_options(attacks PRIVATE -g)

target_include_directories(baseMnistTrain PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(regularMnistTrain PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(registeredMnistTrain PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(subsetSampler INTERFACE)

target_include_directories(registeredMNIST INTERFACE)

target_include_directories(rdmaOps PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(tensorOps PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(attacks PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(rbyzAux PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_include_directories(srvr PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(clnt PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(srvr PUBLIC)

target_precompile_headers(baseMnistTrain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)
target_precompile_headers(regularMnistTrain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)
target_precompile_headers(registeredMnistTrain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)
target_precompile_headers(tensorOps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)
target_precompile_headers(attacks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)
target_precompile_headers(rbyzAux PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)
target_precompile_headers(srvr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)
target_precompile_headers(clnt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/global/pch.hpp)

target_link_libraries(rdmaOps PUBLIC libnorm)

target_link_libraries(baseMnistTrain PUBLIC 
  torch_interface
  registeredMNIST
  subsetSampler
)

target_link_libraries(regularMnistTrain PUBLIC 
  torch_interface
  registeredMNIST
  subsetSampler
)

target_link_libraries(registeredMnistTrain PUBLIC 
  torch_interface
  registeredMNIST
  subsetSampler
)

target_link_libraries(attacks PUBLIC 
  torch_interface
  baseMnistTrain
)

target_link_libraries(rbyzAux PUBLIC 
  torch_interface
  rdmaOps
)

target_link_libraries(tensorOps PUBLIC torch_interface)

target_link_libraries(srvr PUBLIC  
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  torch_interface
  baseMnistTrain
  regularMnistTrain
  registeredMnistTrain  
  rdmaOps
  tensorOps
  attacks
  rbyzAux
  Threads::Threads
)

target_link_libraries(clnt PUBLIC 
  rdmaOps
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  torch_interface
  baseMnistTrain
  regularMnistTrain
  registeredMnistTrain
  tensorOps
  rbyzAux
  Threads::Threads
)