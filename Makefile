CUDA_PATH?=/usr/local/cuda
HOST_COMPILER?=g++
NVCC:=$(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) -arch=sm_60 -rdc=true -lcudadevrt 
SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}
HEADERS = ${shell find include/ -type f -regextype egrep -regex ".*\.(hpp|h|cuh)$$"}
TESTS = ${shell find test/ -type f -regextype egrep -regex ".*_test\.cpp$$"}
BUILD_TESTS = ${shell find test/ -type f -regextype egrep -regex ".*_test\.cpp$$" | xargs basename -a -s .cpp  | awk '{print "build_"$$0}'}
RUN_TESTS = ${shell find test/ -type f -regextype egrep -regex ".*_test\.cpp$$" | xargs basename -a -s .cpp  | awk '{print "run_"$$0}'}

build/libgpuc_cuda.so: $(SOURCES) $(HEADERS)
	$(NVCC) -I ./include --compiler-options '-fPIC' --shared -o $@ $^

build: build/libgpuc_cuda.so

build_%: build $(TESTS) $(HEADERS)
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/$* ${shell find test/ -type f -regextype egrep -regex ".*$*\.cpp$$"}

run_%: build_% $(TESTS)
	LD_LIBRARY_PATH=./build ./build/$*

build_all: $(BUILD_TESTS)

test_all: $(RUN_TESTS) 

clean:
	rm -rf build/*

.PHONY: build clean test_vector_add test_matmul test_sum2d test_all test_mean2d