cmake_minimum_required(VERSION 3.27)
include(CMakePrintHelpers)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_VERBOSE_MAKEFILE ON)

execute_process(COMMAND bash -c "uname | tr -d '\n'" OUTPUT_VARIABLE OS)
execute_process(COMMAND bash -c "uname -p | tr -d '\n'" OUTPUT_VARIABLE ARCH)
cmake_print_variables(OS ARCH)

set(CMAKE_CXX_COMPILER g++-13)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-missed -fopt-info-loop-optimized -ftree-vectorizer-verbose=2")

if (OS STREQUAL "Darwin")
    execute_process(COMMAND bash -c "g++-13 --version | grep -oE '\\d+\\.\\d+\.\\d+' | head -n 1 | tr -d '\n'" OUTPUT_VARIABLE CPPVER)
    execute_process(COMMAND bash -c "echo \"${CPPVER}\" | cut -d '.' -f 1 | tr -d '\n'" OUTPUT_VARIABLE CPPMAJVER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include/c++/${CPPMAJVER}")
    if (ARCH STREQUAL "arm")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include/c++/${CPPMAJVER}/aarch64-apple-darwin23 -mtune=native")
    endif ()
endif ()

if (ARCH MATCHES "^x86")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
endif ()
if (ARCH MATCHES "^arm")
endif ()

cmake_print_variables(CMAKE_CXX_FLAGS_DEBUG)

project(tensorcpu C CXX)

file(GLOB_RECURSE MYSRC CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.c"
)

include_directories(./include)

###### tensorcpu
add_library(tensorcpu SHARED
        ${MYSRC}
)
target_link_libraries(tensorcpu pthread m)

###### binary_arith_test
add_executable(binary_arith_test test/binary_arith_test.cpp)

###### gemm_test
add_executable(gemm_test test/gemm_test.cpp)

###### dummy
add_executable(dummy test/dummy.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)
pkg_check_modules(TBB REQUIRED tbb)

foreach (target IN ITEMS tensorcpu binary_arith_test gemm_test dummy)
    target_link_directories(${target} PUBLIC ./build)
    if (NOT ${target} STREQUAL "tensorcpu")
        target_link_libraries(${target} tensorcpu)
    endif ()
    target_link_libraries(${target} pthread m)

    target_link_libraries(${target} ${OPENBLAS_LIBRARIES})
    target_link_directories(${target} PRIVATE ${OPENBLAS_LIBRARY_DIRS})
    target_include_directories(${target} PRIVATE ${OPENBLAS_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE ${OPENBLAS_CFLAGS_OTHER})

    target_link_libraries(${target} ${TBB_LIBRARIES})
    target_link_directories(${target} PRIVATE ${TBB_LIBRARY_DIRS})
    target_include_directories(${target} PRIVATE ${TBB_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE ${TBB_CFLAGS_OTHER})

    set_target_properties(${target}
            PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
    )
endforeach ()
