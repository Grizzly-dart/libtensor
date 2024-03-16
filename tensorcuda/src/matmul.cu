#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

#define BLOCK_SIZE 16

// https://siboehm.com/articles/22/CUDA-MMM
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

char const* libtcCudaMatMul(libtcCudaStream& stream, double* out, double* inp1, double* inp2, uint32_t m, uint32_t n, uint32_t k) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {};
  config.blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
  if (k < BLOCK_SIZE) {
    config.blockDim.x = k;
  }
  if (m < BLOCK_SIZE) {
    config.blockDim.y = m;
  }
  config.gridDim = dim3((k + config.blockDim.x - 1) / config.blockDim.x, (m + config.blockDim.y - 1) / config.blockDim.y);
  err = cudaLaunchKernelEx(&config, matmulTiledKernel<double>, out, inp1, inp2, m, n, k);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}