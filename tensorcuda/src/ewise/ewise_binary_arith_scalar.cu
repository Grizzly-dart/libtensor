#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

// TODO implement stride and split
/// Adds two tensors
template <typename O, typename I1, typename I2>
__global__ void addScalar(O *out, const I1 *in1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] + inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void subScalar(O *out, const I1 *in1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] - inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void subLhsScalar(O *out, const I1 *in1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp2 - in1[i];
  }
}

template <typename O, typename I1, typename I2>
__global__ void mulScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] * in2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void divScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] / in2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void divLhsScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in2 / in1[i];
  }
}

template <typename O, typename I1, typename I2>
__global__ void powScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::pow(in1[i], in2);
  }
}

template <typename O, typename I1, typename I2>
__global__ void powLhsScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::pow(in2, in1[i]);
  }
}