#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

template <typename O, typename I>
__global__ void sqr(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp[i] * inp[i];
  }
}

template <typename O, typename I>
__global__ void sqrt(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::sqrt(inp[i]);
  }
}

template <typename O, typename I>
__global__ void exp(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::exp(inp[i]);
  }
}