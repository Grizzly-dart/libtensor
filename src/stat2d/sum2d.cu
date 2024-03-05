#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <libgpuc_cuda.hpp>
#include <string>

template <typename T>
__global__ void sum2DKern(T* out, T* in, uint32_t numCols) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  // uint32_t numRows = gridDim.y;
  uint32_t row = blockIdx.y;
  T sum = 0;
  for (uint32_t col = threadIdx.x + blockIdx.x * blockDim.x; col < numCols; col += numThreads) {
    if (col < numCols) {
      uint32_t idx = row * numCols + col;
      sum += in[idx];
    }
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  __syncthreads();

  uint8_t lane = threadIdx.x % warpSize;
  uint8_t warp = threadIdx.x / warpSize;

  __shared__ T sdata[32];
  if (warp == 0) {
    sdata[threadIdx.x] = 0;
  }
  __syncthreads();

  if (lane == 0) {
    sdata[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    sum = sdata[lane];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  } else {
    sum = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(out + row, sum);
  }
}

void sum2DCudaCkernDouble(double* out, double* in, CSize2D inSize) {
  cudaLaunchConfig_t config = {};
  auto err = cudaLaunchKernelEx(&config, sum2DKern<double>, out, in, in.dim[1]);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}

void sum2DTensor(Tensor out, Tensor in) {
  // TODO use streams
  if (in.ndim != 2) {
    throw std::string("Input tensor must be 2D");
  } else if (out.ndim != 1) {
    throw std::string("Output tensor must be 1D");
  } else if (out.dim[0] != in.dim[0]) {
    throw std::string("Size mismatch between input and output tensors");
  }

  cudaLaunchConfig_t config = {};
  if (in.dim[1] < MAX_THREADS_PER_BLOCK) {
    config.blockDim.x = in.dim[1];
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = MAX_THREADS_PER_BLOCK;
    config.gridDim.x = (in.dim[1] + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
  }
  config.gridDim.y = in.dim[0];
  // std::cout << "Block dim: " << config.blockDim.x << " Grid dim: " << config.gridDim.x << " " << config.gridDim.y << std::endl;

  auto err = cudaLaunchKernelEx(&config, sum2DKern<double>, out.mem, in.mem, in.dim[1]);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
  // TODO remove
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}