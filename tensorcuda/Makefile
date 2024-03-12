OS := $(shell uname | tr A-Z a-z)
HOST_COMPILER?=g++
NVCC:=nvcc -ccbin $(HOST_COMPILER) -arch=sm_60 -rdc=true -lcudart -lcudadevrt --std c++20 
SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}
HEADERS = ${shell find include/ -type f -regextype egrep -regex ".*\.(hpp|h|cuh)$$"}
TESTS = ${shell find test/ -type f -regextype egrep -regex ".*_test\.cpp$$"}
BUILD_TESTS = ${shell find test/ -type f -regextype egrep -regex ".*_test\.cpp$$" | xargs basename -a -s .cpp  | awk '{print "build_"$$0}'}
RUN_TESTS = ${shell find test/ -type f -regextype egrep -regex ".*_test\.cpp$$" | xargs basename -a -s .cpp  | awk '{print "run_"$$0}'}

build/libtensorcuda.so: $(SOURCES) $(HEADERS)
	$(NVCC) -I ./include --compiler-options '-fPIC' --shared -o $@ $(SOURCES)

build: build/libtensorcuda.so

build_all: $(BUILD_TESTS)

test_all: $(RUN_TESTS) 

build_%: build $(TESTS) $(HEADERS)
	$(NVCC) -I ./include -L ./build -ltensorc -o build/$* ${shell find test/ -type f -regextype egrep -regex ".*$*\.cpp$$"}

run_%: build_% $(TESTS)
	LD_LIBRARY_PATH=./build ./build/$*

clean:
	rm -rf build/*

copy_dart: build/libtensorcuda.so
	cp build/libtensorcuda.so ../../gpuc_dart/lib/asset/$(OS)/

.PHONY: build/libtensorcuda.so build clean test_vector_add test_matmul test_sum2d test_all test_mean2d