# Preload and cache Torch configuration to optimize build time.

# Cache the Torch directory. Adjust the path if your Torch installation is located elsewhere.
if(NOT DEFINED Torch_DIR)
  set(Torch_DIR "/home/bustaman/libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128/libtorch/share/cmake/Torch" CACHE PATH "Directory for Torch cmake configuration")
endif()

if(NOT Torch_FOUND)
  message(STATUS "Finding Torch ...")
  find_package(Torch REQUIRED)
else()
  message(STATUS "Using cached Torch configuration")
endif()

message(STATUS "Using Torch_DIR: ${Torch_DIR}")

# Find the Torch package (this result will be cached).
#find_package(Torch REQUIRED)

# Cache key Torch variables to prevent re-configuration.
if(NOT Torch_INCLUDE_DIRS)
  set(Torch_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}" CACHE STRING "Torch include directories")
endif()

if(NOT Torch_LIBRARIES)
  set(Torch_LIBRARIES "${TORCH_LIBRARIES}" CACHE STRING "Torch libraries")
endif()

message(STATUS "Found Torch version: ${Torch_VERSION}")