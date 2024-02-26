CUDA_PATH?=/usr/local/cuda
HOST_COMPILER?=g++
NVCC:=$(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

build/libgpuc_cuda.o:
	$(NVCC) -I ./include --compiler-options '-fPIC' --shared -o build/libgpuc_cuda.so src/tensor.cpp src/vector.cu src/matmul.cu

build: build/libgpuc_cuda.o

build_vector_add:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/vector_add test/vector_add/vector_add_test.cpp 

test_vector_add: build build_vector_add
	LD_LIBRARY_PATH=./build ./build/vector_add

build_matmul:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/matmul test/vector_add/matmul_test.cpp

test_matmul: build build_matmul
	LD_LIBRARY_PATH=./build ./build/matmul

clean:
	rm -rf build/*

.PHONY: build clean vector_add