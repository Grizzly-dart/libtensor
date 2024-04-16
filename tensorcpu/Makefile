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
		CXXFLAGS += -I/opt/homebrew/include/c++/${CPPMAJVER}/aarch64-apple-darwin23 -arch arm64
	endif
	#tbbVersions = $(shell brew list --versions | fgrep tbb | cut -d ' ' -f 2)
	#CXXFLAGS +=  -I/opt/homebrew/Cellar/tbb/${tbbVersions}/include -L/opt/homebrew/Cellar/tbb/${tbbVersions}/lib
	SOURCES = ${shell find -E src -type f -regex ".*\.(cpp|cu|c)$$"}
	HEADERS = ${shell find -E include -type f -regex ".*\.(hpp|h)$$"}
endif
ifeq ($(OS),cygwin)

endif

CXXFLAGS += -I ./include -L ./build -std=c++23 -g3 -OO -lpthread -ftree-vectorize -ftree-vectorizer-verbose=1
# CXXFLAGS += -save-temps

CXXFLAGS += $(shell pkg-config --cflags tbb)
LDFLAGS += $(shell pkg-config --libs tbb)
CXXFLAGS += $(shell pkg-config --cflags openblas)
LDFLAGS += $(shell pkg-config --libs openblas)

all: build build/for_each

%.o: %.cpp
	${CPP} ${CXXFLAGS} -fPIC -c $< -o $@

build/libtensorcpu.so: $(SOURCES) $(HEADERS)
	# python3 scripts/ewise_binary_arith.py
	${CPP} ${CXXFLAGS} --shared -fPIC -o build/libtensorcpu.so ${SOURCES} ${LDFLAGS}
	ls -lh build/libtensorcpu.so

build/binary_arith_test: test/binary_arith_test.cpp build/libtensorcpu.so
	${CPP} ${CXXFLAGS} -o $@ test/binary_arith_test.cpp -ltensorcpu ${LDFLAGS}

build/gemm_test: test/gemm_test.cpp build/libtensorcpu.so
	${CPP} ${CXXFLAGS} -save-temps -o $@ test/gemm_test.cpp -ltensorcpu ${LDFLAGS}

build/dummy: test/dummy.cpp build/libtensorcpu.so
	${CPP} ${CXXFLAGS} -o $@ test/dummy.cpp -ltensorcpu ${LDFLAGS}

build: build/libtensorcpu.so

copy_dart: build/libtensorcpu.so only_copy

only_copy:
	cp build/libtensorcpu.so ../../gpuc_dart/lib/asset/$(OS)/

cmake_gen:
	cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=ninja -G Ninja -S . -B ./cmake-build-debug

setup_ubuntu:
	sudo apt-get install -y libtbb-dev libopenblas-openmp-dev

setup_darwin:
	brew install gcc@13
	brew install tbb
	brew install openblas
	find /opt/homebrew/Cellar -name 'pkgconfig' -type d | grep -E 'tbb|openblas' | tr '\n' ':'
	@echo "\nAdd the above directories to PKG_CONFIG_PATH in your .bashrc or .zshrc"

clean:
	rm -f build/libtensorcpu.so
	rm -f build/for_each

.PHONY: all build clean copy_dart only_copy