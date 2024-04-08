cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

project(tensorcuda CUDA CXX)

file(GLOB_RECURSE MYSRC CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.cu"
        "src/*.c"
)

add_library(tensorcuda SHARED
    ${MYSRC}
)

include_directories(./include)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -arch=sm_60 -rdc=true -lcudart -lcudadevrt -lpthread --std c++20")

set_target_properties( tensorcuda
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)