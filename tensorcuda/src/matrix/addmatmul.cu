#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

#define TILE_SIZE 32

template <typename T>
__global__ void addMatmulTiledKernel(T* matOut, T* matIn1, T* matIn2, uint32_t m, uint32_t n, uint32_t k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ T tile1[TILE_SIZE][TILE_SIZE];
  __shared__ T tile2[TILE_SIZE][TILE_SIZE + 1];

  T sum = 0.0;
  for (int i = 0; i < n; i += TILE_SIZE) {
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

    for (int j = 0; j < TILE_SIZE; ++j) {
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