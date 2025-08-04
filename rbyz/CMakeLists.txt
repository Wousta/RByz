cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-12" CACHE STRING "C++ compiler path")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE STRING "CUDA compiler path")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

project(RByz LANGUAGES CXX CUDA)
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cache.cmake)

include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
  set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  message(STATUS "nvToolsExt not found - creating dummy target")
endif()

set(Torch_DIR "/home/bustaman/libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

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
  ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/datasetLogic/include
)

add_library(libnorm SHARED IMPORTED)
set_target_properties(libnorm PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  IMPORTED_NO_SONAME TRUE)

add_library(torch_interface INTERFACE)
target_include_directories(torch_interface INTERFACE 
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)
target_link_libraries(torch_interface INTERFACE 
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
)
target_compile_definitions(torch_interface INTERFACE 
  TORCH_API_INCLUDE_EXTENSION_H
  USE_PYTHON=0
)
target_precompile_headers(torch_interface INTERFACE 
  <torch/torch.h>
)

# Create a unified source file (Unity build)
set(UNITY_SRC ${CMAKE_BINARY_DIR}/unity_torch_build.cpp)
file(WRITE ${UNITY_SRC} "// Auto-generated unity build file\n")
# avoid headers that require Python
file(APPEND ${UNITY_SRC} "#define USE_PYTHON 0\n")
file(APPEND ${UNITY_SRC} "#include <torch/csrc/api/include/torch/all.h>\n")
file(APPEND ${UNITY_SRC} "#include \"../src/tensorOps.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../src/attacks.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../datasetLogic/include/subsetSampler.hpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../datasetLogic/include/dataset/registeredMNIST.hpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../datasetLogic/src/transform.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../datasetLogic/src/dataset/registeredCIFAR10.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../datasetLogic/src/manager/baseRegDatasetMngr.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../datasetLogic/src/manager/regMnistMngr.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../datasetLogic/src/manager/regCIFAR10Mngr.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../algorithm/src/fltrustSrvr.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../algorithm/src/datasetSplitter.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../algorithm/src/rbyzSrvr.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../algorithm/src/rbyzClnt.cpp\"\n")
file(APPEND ${UNITY_SRC} "#include \"../src/rdmaOps.cpp\"\n")

add_library(torch_implementations STATIC ${UNITY_SRC})
target_link_libraries(torch_implementations PUBLIC 
  torch_interface
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
)

add_executable(srvr src/entities/srvr.cpp)
add_executable(clnt src/entities/clnt.cpp)

target_compile_options(srvr PRIVATE -g)
target_compile_options(clnt PRIVATE -g)


target_include_directories(srvr PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(clnt PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(srvr PUBLIC)

target_link_libraries(srvr PUBLIC  
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  torch_implementations
  Threads::Threads
)

target_link_libraries(clnt PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  torch_implementations
  Threads::Threads
)