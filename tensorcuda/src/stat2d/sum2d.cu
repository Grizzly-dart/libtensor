#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

template <typename T>
__global__ void sum2d(T* out, T* in, uint64_t numCols) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t row = blockIdx.y;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;
  T sum = 0;
  for (uint64_t col = thId; col < numCols; col += numThreads) {
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

const char* libtcCudaSum2d(libtcCudaStream& stream, double* out, double* in, Dim2 inSize, dtype outType, dtype inpType) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  uint32_t numThreads = props.multiProcessorCount * 128;
  if(numThreads > inSize.c) {
    numThreads = inSize.c;
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if (numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x = (numThreads + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }
  config.gridDim.y = inSize.r;

  if(outType == dtype::f64) {
    // TODO
  } else if(outType == dtype::f32) {
    // TODO
  } else if(outType == dtype::u64) {
    // TODO
  } else if(outType == dtype::i64) {
    // TODO
  } else if(outType == dtype::u32) {
    // TODO
  } else if(outType == dtype::i32) {
    // TODO
  } else {
    return "Unsupported output type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}