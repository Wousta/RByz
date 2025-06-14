#!/bin/bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#rm -rf build
mkdir -p build
cd build
export PATH="/home/bustaman/.local/bin:$PATH"
conan install .. --output-folder=. --build=missing
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DTorch_DIR="/home/bustaman/libtorch/share/cmake/Torch" ..
cmake -C ../cache.cmake -S .. 
cmake --build . -j28
