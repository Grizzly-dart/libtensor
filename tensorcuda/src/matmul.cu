#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

#define BLOCK_SIZE 16

template <typename T>
__global__ void matmulTiledKernel(T* matOut, T* matIn1, T* matIn2, uint32_t m, uint32_t n, uint32_t k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ T tile1[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T tile2[BLOCK_SIZE][BLOCK_SIZE];

  T sum = 0.0;
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    if (row < m && i + threadIdx.x < n) {
      tile1[threadIdx.y][threadIdx.x] = matIn1[row * n + i + threadIdx.x];
    } else {
      tile1[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (col < k && i + threadIdx.y < n) {
      tile2[threadIdx.y][threadIdx.x] = matIn2[(i + threadIdx.y) * k + col];
    } else {
      tile2[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j) {
      if (row < m && col < k) {
        sum += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
      }
    }
    __syncthreads();
  }

  if (row < m && col < k) {
    matOut[row * k + col] = sum;
  }
}

template <typename T>
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

void matmul(Tensor out, Tensor in1, Tensor in2) {
  if (in1.ndim != in2.ndim || in1.ndim != out.ndim) {
    throw std::string("All input tensors must have the same number of dimensions");
  }

  if (in1.ndim < 2) {
    throw std::string("All input tensors must have at least 2 dimensions");
  }

  uint64_t m = getTensorM(in1);
  uint64_t n = getTensorN(in1);
  uint64_t k = getTensorN(in2);
  if (k != getTensorM(in2) || m != getTensorM(out) || k != getTensorN(out)) {
    throw std::string("Invalid tensor dimensions");
  }

  cudaLaunchConfig_t config = {};
  config.blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
  if (k < BLOCK_SIZE) {
    config.blockDim.x = k;
  }
  if (m < BLOCK_SIZE) {
    config.blockDim.y = m;
  }
  config.gridDim = dim3((m + config.blockDim.x - 1) / config.blockDim.x, (k + config.blockDim.y - 1) / config.blockDim.y);
  for (int i = 0; i < getTensorCountMat(in1); i++) {
    auto err = cudaLaunchKernelEx(&config, matmulTiledKernel<double>, out.mem + i * m * k, in1.mem + i * m * n, in2.mem + i * n * k, m, n, k);
    if (err != cudaSuccess) {
      throw std::string("Failed to launch kernel: ") + cudaGetErrorString(err);
    }
  }
}