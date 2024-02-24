CUDA_PATH?=/usr/local/cuda
HOST_COMPILER?=g++
NVCC:=$(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

build/libgpuc_cuda.o:
	$(NVCC) -I ./include --compiler-options '-fPIC' --shared -o build/libgpuc_cuda.so src/vector_add.cu

build: build/libgpuc_cuda.o

vector_add:
	$(NVCC) -I ./include -L ./build -lgpuc_cuda -o build/vector_add test/vector_add/vector_add.cpp 

clean:
	rm -rf build/*

.PHONY: build clean vector_add