cmake_minimum_required(VERSION 3.27)
include(CMakePrintHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CXX_COMPILER g++)
# execute_process(COMMAND which nvcc OUTPUT_VARIABLE CMAKE_CUDA_COMPILER)

project(tensorcuda CUDA CXX)

file(GLOB_RECURSE MYSRC CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.cu"
        "src/*.c"
)

add_library(tensorcuda SHARED
        ${MYSRC}
)

add_custom_command(TARGET tensorcuda
        PRE_BUILD COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/ewise_arith_gen.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

include_directories(./include)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -arch=sm_60 -rdc=true -lcudart -lcudadevrt -lpthread --std c++20")

set_target_properties(tensorcuda
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)