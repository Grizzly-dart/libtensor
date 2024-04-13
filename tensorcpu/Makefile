OS := $(shell uname | tr A-Z a-z)
ifeq ($(OS),linux)
	CPP := g++
	SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}
	HEADERS = ${shell find include/ -type f -regextype egrep -regex ".*\.(hpp|h)$$"}
endif
ifeq ($(OS),darwin)
	CPPMAJVER = $(shell brew list --versions | fgrep gcc | cut -d ' ' -f 2 | cut -d '.' -f1)
	CPP = g++-${CPPMAJVER}
	CXXFLAGS += -I/opt/homebrew/include/c++/${CPPMAJVER}
	ifeq ($(shell uname -p),arm)
		CXXFLAGS += -I/opt/homebrew/include/c++/${CPPMAJVER}/aarch64-apple-darwin23
	endif
	tbbVersions = $(shell brew list --versions | fgrep tbb | cut -d ' ' -f 2)
	CXXFLAGS +=  -I/opt/homebrew/Cellar/tbb/${tbbVersions}/include -L/opt/homebrew/Cellar/tbb/${tbbVersions}/lib
	SOURCES = ${shell find -E src -type f -regex ".*\.(cpp|cu|c)$$"}
	HEADERS = ${shell find -E include -type f -regex ".*\.(hpp|h)$$"}
endif
ifeq ($(OS),cygwin)

endif

CXXFLAGS += -I ./include -L ./build -std=c++23 -g3 -O3 -lpthread -mavx512f -ftree-vectorize -ftree-vectorizer-verbose=1

all: build build/for_each

%.o: %.cpp
	${CPP} ${CXXFLAGS} -fPIC -c $< -o $@

build/libtensorcpu.so: $(SOURCES) $(HEADERS)
	# python3 scripts/ewise_binary_arith.py
	${CPP} ${CXXFLAGS} --shared -fPIC -o build/libtensorcpu.so ${SOURCES}
	ls -lh build/libtensorcpu.so

build/for_each: test/for_each.cpp build/libtensorcpu.so
	${CPP} ${CXXFLAGS} -o $@ test/for_each.cpp -ltensorcpu -ltbb

build: build/libtensorcpu.so

copy_dart: build/libtensorcpu.so only_copy

only_copy:
	cp build/libtensorcpu.so ../../gpuc_dart/lib/asset/$(OS)/

cmake_gen:
	cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=ninja -G Ninja -S . -B ./cmake-build-debug

clean:
	rm -f build/libtensorcpu.so
	rm -f build/for_each

.PHONY: all build clean copy_dart only_copy