#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <string>

#define TILE_SIZE 32

// https://siboehm.com/articles/22/CUDA-MMM
template <typename T>
__global__ void caddmm(
    T *out, T *inp1, T *inp2, T *add, uint32_t m, uint32_t n, uint32_t k
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = blockIdx.z;

  inp1 += batch * m * n;
  inp2 += batch * n * k;
  out += batch * m * k;

  // TILE_SIZE+1 to avoid shared memory bank conflicts
  __shared__ T tile1[TILE_SIZE][TILE_SIZE + 1];
  __shared__ T tile2[TILE_SIZE][TILE_SIZE];

  T sum = 0.0;
  for (int i = 0; i < n; i += TILE_SIZE) {
    if (row < m && i + threadIdx.x < n) {
      T val = inp1[row * n + i + threadIdx.x];
      tile1[threadIdx.y][threadIdx.x] = val;
    }
    if (i + threadIdx.y < n && col < k) {
      T val = inp2[(i + threadIdx.y) * k + col];
      tile2[threadIdx.y][threadIdx.x] = val;
    }
    __syncthreads();

    for (int j = 0; j < TILE_SIZE; ++j) {
      if (row < m && col < k && i + j < n) {
        sum += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
      }
    }
    __syncthreads();
  }

  if (row < m && col < k) {
    out[row * k + col] = sum + add[col];
  }
}

char const *tcuMatMulCadd(
    tcuStream &stream, double *out, double *inp1, double *inp2,
    double *add, uint32_t m, uint32_t n, uint32_t k, uint32_t batches
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  uint32_t max = m > k ? m : k;
  max = max > n ? max : n;
  if (max < TILE_SIZE) {
    config.blockDim = dim3(max, max);
  } else {
    config.blockDim = dim3(TILE_SIZE, TILE_SIZE);
  }
  config.gridDim.x = (k + config.blockDim.x - 1) / config.blockDim.x;
  config.gridDim.y = (m + config.blockDim.y - 1) / config.blockDim.y;
  config.gridDim.z = batches;
  err = cudaLaunchKernelEx(
      &config, caddmm<double>, out, inp1, inp2, add, m, n, k
  );
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}