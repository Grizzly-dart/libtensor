CUDA_PATH?=/usr/local/cuda
HOST_COMPILER?=g++
NVCC:=$(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) -arch=sm_60
SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}

build/libgpuc_cuda.so: $(SOURCES)
	echo $(REGEXP) $(SOURCES)
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

build_test_rowwise_sum:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/rowwise_sum_test test/rowwise/rowwise_sum_test.cpp

test_rowwise_sum: build build_test_rowwise_sum
	LD_LIBRARY_PATH=./build ./build/rowwise_sum_test

test_all: test_vector_add test_matmul test_rowwise_sum

clean:
	rm -rf build/*

.PHONY: build clean vector_add