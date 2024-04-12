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

CXXFLAGS += -I ./include -L ./build -std=c++23 -g3 -O3 -lpthread -mavx512f

all: build build/for_each

build/libtensorc.so: $(SOURCES) $(HEADERS)
	# python3 scripts/ewise_binary_arith.py
	${CPP} ${CXXFLAGS} --shared -fPIC -o build/libtensorc.so ${SOURCES}

build/for_each: scratch/for_each.cpp build/libtensorc.so
	${CPP} ${CXXFLAGS} -o $@ scratch/for_each.cpp -ltensorc -ltbb

build: build/libtensorc.so

copy_dart: build/libtensorc.so only_copy

only_copy:
	cp build/libtensorc.so ../../gpuc_dart/lib/asset/$(OS)/

clean:
	rm -f build/libtensorc.so
	rm -f build/for_each

.PHONY: all build clean copy_dart only_copy