cmake_minimum_required(VERSION 3.27)
include(CMakePrintHelpers)
set(CMAKE_CXX_STANDARD 23)

execute_process(COMMAND bash -c "uname | tr -d '\n'" OUTPUT_VARIABLE OS)
execute_process(COMMAND bash -c "uname -p | tr -d '\n'" OUTPUT_VARIABLE ARCH)
# cmake_print_variables(OS ARCH)

set(CMAKE_CXX_COMPILER g++-13)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -ffast-math")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-missed -fopt-info-loop-optimized -ftree-vectorizer-verbose=2")

if (OS STREQUAL "Darwin")
	execute_process(COMMAND brew list --versions | fgrep tbb | cut -d ' ' -f 2 OUTPUT_VARIABLE tbbVersions)
	execute_process(COMMAND g++-13 --version | grep -oP '\\d+\\.\\d+\.\\d+' | head -n 1 OUTPUT_VARIABLE CPPVER)
	execute_process(COMMAND echo "${CPPVER}" | cut -d '.' -f 1 OUTPUT_VARIABLE CPPMAJVER)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include/c++/${CPPMAJVER} -I/opt/homebrew/Cellar/tbb/${tbbVersions}/include -L/opt/homebrew/Cellar/tbb/${tbbVersions}/lib")
	if (ARCH STREQUAL "arm")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include/c++/${CPPMAJVER}/aarch64-apple-darwin23")
	endif ()
endif ()

project(tensorcpu CXX)

file(GLOB_RECURSE MYSRC CONFIGURE_DEPENDS
		"src/*.cpp"
		"src/*.c"
)

add_library(tensorcpu SHARED
		${MYSRC}
)

target_link_libraries(tensorcpu tbb pthread m)

add_executable(binary_arith_test test/binary_arith_test.cpp)

target_link_directories(binary_arith_test PUBLIC ./build)
target_link_libraries(binary_arith_test tensorcpu tbb pthread m)

include_directories(./include)

set_target_properties(tensorcpu
		PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

set(CMAKE_VERBOSE_MAKEFILE ON)