#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include <libgpuc_cuda.hpp>

/// Adds two vectors
__global__ void vectorAdd2Kernel(const double* in1, const double* in2, double* out, uint32_t n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        out[i] = in1[i] + in2[i];
    }
}

void elementwiseAdd2(double *out, double *in1, double *in2, uint32_t n) {
    uint32_t threads = n;
    uint32_t blocks = 1;
    if (n > MAX_THREADS_PER_BLOCK) {
        threads = 1024;
        blocks = (n + threads - 1) / threads;
    }
    cudaLaunchConfig_t config = {};
    config.blockDim.x = threads;
    config.gridDim.x = blocks;
    auto err = cudaLaunchKernelEx(&config, vectorAdd2Kernel, in1, in2, out, n);
    if (err != cudaSuccess) {
        throw std::string(cudaGetErrorString(err));
    }
}

void Tensor::read(double* out, uint64_t size) {
	if(size != this->x) {
		throw std::string("Size mismatch");
	}
  auto err = cudaMemcpy(out, this->mem, size * sizeof(double), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}

void Tensor::write(double* inp, uint64_t size) {
	if(size != this->x) {
		throw std::string("Size mismatch");
	}
  auto err = cudaMemcpy(this->mem, inp, size * sizeof(double), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}

Tensor Tensor::make1D(uint64_t n) {
  auto tensor = Tensor{x: n};
  cudaMalloc(&tensor.mem, n * sizeof(double));
  return tensor;
}

void Tensor::release() {
    if (released)
        return;

    cudaFree(this->mem);
    released = true;
}