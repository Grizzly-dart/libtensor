CUDA_PATH?=/usr/local/cuda
HOST_COMPILER?=g++
NVCC:=$(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) -arch=sm_60
SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}
TESTS = ${shell find test/ -type f -regextype egrep -regex ".*_test\.cpp$$"}

build/libgpuc_cuda.so: $(SOURCES)
	$(NVCC) -I ./include --compiler-options '-fPIC' --shared -o $@ $^

build: build/libgpuc_cuda.so

build_test_vector_add:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/vector_add_test test/vector/vector_add_test.cpp 

test_vector_add: build build_test_vector_add
	LD_LIBRARY_PATH=./build ./build/vector_add_test

build_test_matmul:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/matmul_test test/matmul_test.cpp

test_matmul: build build_test_matmul
	LD_LIBRARY_PATH=./build ./build/matmul_test

build_test_sum2d:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/sum2d_test test/stat2d/sum2d_test.cpp

test_sum2d: build build_test_sum2d
	LD_LIBRARY_PATH=./build ./build/sum2d_test

build_test_mean2d:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/mean2d_test test/stat2d/mean2d_test.cpp

test_mean2d: build build_test_mean2d
	LD_LIBRARY_PATH=./build ./build/mean2d_test

test_all: test_vector_add test_matmul test_sum2d

clean:
	rm -rf build/*

.PHONY: build clean test_vector_add test_matmul test_sum2d test_all test_mean2d