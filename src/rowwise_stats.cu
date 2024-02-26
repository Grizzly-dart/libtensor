#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

// TODO two dimensional blocks
template <typename T>
__global__ void sum2DKernel(T* out, T* in, uint32_t n) {
  T sum = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    uint32_t idx = blockIdx.x * blockDim.x + i;
    sum += in[idx];
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  __syncthreads();

  uint8_t lane = threadId.x % warpSize;
  uint8_t wrap = threadId.x / warpSize;

  extern __shared__ T sdata[32];

  if (lane == 0) {
    sdata[wrap] = sum;
  }
  __syncthreads();

  if (wrap == 0) {
    sum = (lane < blockDim.x / warpSize) ? sdata[lane] : 0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }

  // TODO atomic add across blocks
  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

void sum2D(Tensor out, Tensor in) {
  if (in.ndim != 2) {
    throw std::string("Input tensor must be 2D");
  } else if (out.ndim != 1) {
    throw std::string("Output tensor must be 1D");
  } else if (out.dim[0] != in.dim[0]) {
    throw std::string("Size mismatch between input and output tensors");
  }

  cudaLaunchConfig_t config = {};
  // TODO decide num blocks and threads

  auto err = cudaLaunchKernelEx(&config, sum2DKernel<double>, out.mem, in.mem, n);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}