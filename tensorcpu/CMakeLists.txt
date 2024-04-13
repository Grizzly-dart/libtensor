cmake_minimum_required(VERSION 3.27)
include(CMakePrintHelpers)
set(CMAKE_CXX_STANDARD 23)

execute_process(COMMAND uname OUTPUT_VARIABLE OS)
execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)

set(CMAKE_CXX_COMPILER gcc-13)

if(OS STREQUAL "Darwin")
	execute_process(COMMAND brew list --versions | fgrep tbb | cut -d ' ' -f 2 OUTPUT_VARIABLE tbbVersions)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/Cellar/tbb/${tbbVersions}/include -L/opt/homebrew/Cellar/tbb/${tbbVersions}/lib")
	if(ARCH STREQUAL "arm")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include/c++/${CPPMAJVER}/aarch64-apple-darwin23")
	endif ()
elseif (OS STREQUAL "Linux")
    message(tejag)
endif ()

project(tensorcuda CXX)

file(GLOB_RECURSE MYSRC CONFIGURE_DEPENDS
		"src/*.cpp"
		"src/*.c"
)

add_library(tensorcpu SHARED
		${MYSRC}
)

link_libraries(tbb pthread)

add_executable(for_each test/for_each.cpp)

target_link_directories(for_each PRIVATE ./build)
target_link_libraries(for_each tensorcpu)

include_directories(./include)

set_target_properties(tensorcpu
		PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)