cmake_minimum_required(VERSION 3.27)
project(tensorcuda CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

file(GLOB MYSRC CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.cu"
        "src/*.c"
)

add_library(tensorcuda SHARED
    ${MYSRC}
)