#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <reducers.hpp>
#include <string>

template <typename O, typename I>
__global__ void variance(
    O *out, I *inp, uint64_t nel, uint64_t correction
) {
  uint32_t numThreads = blockDim.x;

  inp += row * numCols;

  Variance<double> record{};
  for (uint64_t col = threadIdx.x; col < numCols; col += numThreads) {
    record.consume(inp[col]);
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    record.merge(record.shfl_down(offset));
  }
  __syncthreads();

  uint8_t lane = threadIdx.x % warpSize;
  uint8_t warp = threadIdx.x / warpSize;

  __shared__ Variance<double> sdata[32];
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
    O val = record.m2 / (numCols - correction);
    *out = val;
  }
}