#include <stdio.h>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

#define BLOCK_SIZE 32

template<typename T>
__global__ void matmulKernel(T* matOut, T* matIn1, T* matIn2, uint32_t m, uint32_t n, uint32_t k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= k) return;

  T sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += matIn1[row * n + i] * matIn2[i * k + col];
  }
  matOut[row * k + col] = sum;
}

void matmulF64(Tensor out, Tensor in1, Tensor in2) {
  if(in1.ndim != in2.ndim || in1.ndim != out.ndim) {
    throw std::string("All input tensors must have the same number of dimensions");
  }

  if(in1.ndim < 2) {
    throw std::string("All input tensors must have at least 2 dimensions");
  }

  uint64_t m = getTensorM(in1);
  uint64_t n = getTensorN(in1);
  uint64_t k = getTensorN(in2);
  if(k != getTensorM(in2) || m != getTensorM(out) || k != getTensorN(out)) {
    throw std::string("Invalid tensor dimensions");
  }

  // TODO do not consume more than needed
  cudaLaunchConfig_t config = {};
  config.blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
  if(m < BLOCK_SIZE) {
    config.blockDim.x = m;
  }
  if(k < BLOCK_SIZE) {
    config.blockDim.y = k;
  }
  config.gridDim = dim3((m + config.blockDim.x - 1) / config.blockDim.x, (k + config.blockDim.y - 1) / config.blockDim.y);
  for(int i = 0; i < getTensorCountMat(in1); i++) {
    auto err = cudaLaunchKernelEx(&config, matmulKernel<double>, out.mem + i * m * k, in1.mem + i * m * n, in2.mem + i * n * k, m, n, k);
    if(err != cudaSuccess) {
      throw std::string("Failed to launch kernel: ") + cudaGetErrorString(err);
    }
  }
}