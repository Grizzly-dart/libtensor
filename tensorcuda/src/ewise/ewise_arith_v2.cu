#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <tensorcuda.hpp>

/*
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
}*/