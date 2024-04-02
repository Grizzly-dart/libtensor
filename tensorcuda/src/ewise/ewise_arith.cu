#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

/// Adds two tensors
template <typename O, typename I1, typename I2>
__global__ void plus(
    O *out, I1 *inp1, I2 * inp2, I2 scalar, uint64_t n, uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    if (inp2 == nullptr) {
      out[i] = inp1[i] + scalar;
    } else {
      out[i] = inp1[i] + inp2[i];
    }
  }
}

template <typename O, typename I1, typename I2>
__global__ void minus(
    O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n, uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    if (inp2 == nullptr) {
      if (flipScalar) {
        out[i] = scalar - inp1[i];
      } else {
        out[i] = inp1[i] - scalar;
      }
    } else {
      out[i] = inp1[i] - inp2[i];
    }
  }
}

template <typename O, typename I1, typename I2>
__global__ void mul(
    O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n, uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    if (inp2 == nullptr) {
      out[i] = inp1[i] * scalar;
    } else {
      out[i] = inp1[i] * inp2[i];
    }
  }
}

template <typename O, typename I1, typename I2>
__global__ void div(
    O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n, uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    if (inp2 == nullptr) {
      if (flipScalar) {
        out[i] = scalar / inp1[i];
      } else {
        out[i] = inp1[i] / scalar;
      }
    } else {
      out[i] = inp1[i] / inp2[i];
    }
  }
}

template <typename O, typename I1, typename I2>
__global__ void pow(
    O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n, uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    if (inp2 == nullptr) {
      if (flipScalar) {
        out[i] = std::pow(scalar, inp1[i]);
      } else {
        out[i] = std::pow(inp1[i], scalar);
      }
    } else {
      out[i] = std::pow(inp1[i], inp2[i]);
    }
  }
}

#include "ewise_arith_gen.inc"