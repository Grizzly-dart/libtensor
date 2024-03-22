#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <reducers.hpp>
#include <string>

template <typename O, typename I>
__global__ void mean2d(O* out, I* inp, uint64_t numCols) {
  uint32_t numThreads = blockDim.x;
  uint32_t numRows = gridDim.x;
  uint32_t row = blockIdx.x;
  
  inp += row * numCols;
  
  Mean<double> record{};
  for (uint64_t col = threadIdx.x; col < numCols; col += numThreads) {
    if (col < numCols) {
      record.consume(inp[col]);
    }
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    record.merge(record.shfl_down(offset));
  }
  __syncthreads();

  uint8_t lane = threadIdx.x % warpSize;
  uint8_t warp = threadIdx.x / warpSize;

  __shared__ Mean<T> sdata[32];
  if (warp == 0) {
    sdata[threadIdx.x] = {0};
  }
  __syncthreads();

  if (lane == 0) {
    sdata[warp] = record;
  }
  __syncthreads();

  if (warp == 0) {
    record = sdata[lane];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      record.merge(record.shfl_down(offset));
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    out[row] = record.mean;
  }
}

const char *libtcMean2d_f64_f64(
    libtcCudaStream &stream, void *out, void *inp, uint32_t rows, uint64_t cols
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if (cols < 1024) {
    config.blockDim.x = cols;
  } else {
    config.blockDim.x = 1024;
  }
  config.gridDim.x = rows;

  err = cudaLaunchKernelEx(&config, mean2d<double, double>, (double*)out, (double*)inp, cols);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}