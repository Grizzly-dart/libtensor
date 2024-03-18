#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

// TODO should this be warpSize?
#define TILE_SIZE 32

template <typename T>
__global__ void matmulT(
    T *out, T *inp1, T *inp2T, uint32_t m, uint32_t n, uint32_t k
) {
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int batches = blockIdx.z;

  inp1 += batches * m * n;
  inp2T += batches * n * k;
  out += batches * m * k;

  // TILE_SIZE+1 to avoid shared memory bank conflicts
  __shared__ T tile1[TILE_SIZE][TILE_SIZE];
  __shared__ T tile2[TILE_SIZE][TILE_SIZE + 1];

  T sum = 0.0;
  for (int i = 0; i < n; i += TILE_SIZE) {
    if (outRow < m && i + threadIdx.x < n) {
      T val = inp1[outRow * n + i + threadIdx.x];
      tile1[threadIdx.y][threadIdx.x] = val;
    }
    if (outCol < k && i + threadIdx.y < n) {
      T val = inp2T[outCol * k + i + threadIdx.y];
      tile2[threadIdx.y][threadIdx.x] = val;
    }
    __syncthreads();

    for (int j = 0; j < TILE_SIZE; ++j) {
      if (outRow < m && outCol < k && i + j < n) {
        sum += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
      }
    }
    __syncthreads();
  }

  if (outRow < m && outCol < k) {
    out[outRow * k + outCol] = sum;
  }
}

char const *libtcCudaMatMulT(
    libtcCudaStream &stream, double *out, double *inp1, double *inp2T,
    uint32_t m, uint32_t n, uint32_t k, uint32_t batches
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  uint32_t max = m > k ? m : k;
  max = max > n? max : n;
  if(max < TILE_SIZE) {
    config.blockDim = dim3(max, max);
  } else {
    config.blockDim = dim3(TILE_SIZE, TILE_SIZE);
  }
  config.gridDim.x = (k + config.blockDim.x - 1) / config.blockDim.x;
  config.gridDim.y = (m + config.blockDim.y - 1) / config.blockDim.y;
  config.gridDim.z = batches;
  err = cudaLaunchKernelEx(&config, matmulT<double>, out, inp1, inp2T, m, n, k);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}