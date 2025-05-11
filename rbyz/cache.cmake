# Preload and cache Torch configuration to optimize build time.

# Set required internal CMake variables that are missing
set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".so;.a;.dylib")

# Make sure CMake can find the Torch package
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/bustaman/libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128/libtorch")

# Cache the Torch directory. Adjust the path if your Torch installation is located elsewhere.
if(NOT DEFINED Torch_DIR)
  set(Torch_DIR "/home/bustaman/libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128/libtorch/share/cmake/Torch" CACHE PATH "Directory for Torch cmake configuration")
endif()

# Print information about the libtorch installation
message(STATUS "Checking libtorch installation...")
message(STATUS "Torch_DIR: ${Torch_DIR}")
if(EXISTS "${Torch_DIR}/TorchConfig.cmake")
  message(STATUS "TorchConfig.cmake found")
else()
  message(FATAL_ERROR "TorchConfig.cmake not found at ${Torch_DIR}. Please check your libtorch installation.")
endif()

# Cache Torch variables
if(NOT Torch_FOUND)
  message(STATUS "Finding Torch ...")
  find_package(Torch REQUIRED)
  set(Torch_FOUND TRUE CACHE BOOL "Torch has been found")
else()
  message(STATUS "Using cached Torch configuration")
endif()

if(NOT Torch_INCLUDE_DIRS)
  set(Torch_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}" CACHE STRING "Torch include directories")
endif()

if(NOT Torch_LIBRARIES)
  set(Torch_LIBRARIES "${TORCH_LIBRARIES}" CACHE STRING "Torch libraries")
endif()

message(STATUS "Found Torch version: ${Torch_VERSION}")
message(STATUS "Torch include directories: ${Torch_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${Torch_LIBRARIES}")