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

template <typename O, typename I>
__global__ void cast(O *out, const I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp[i];
  }
}

const char *setupElementwiseKernelStrided(
    libtcCudaStream &stream, uint64_t n, cudaLaunchConfig_t &config
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
  uint32_t numThreads = props.multiProcessorCount * 128;
  if (numThreads > n) {
    numThreads = n;
  }

  config.stream = stream.stream;
  if (numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x =
        (numThreads + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }

  return nullptr;
}

/*
// TODO implement stride and split
/// Adds two tensors
template <typename O, typename I1, typename I2>
__global__ void addScalar(O *out, const I1 *inp1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp1[i] + inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void subScalar(O *out, const I1 *inp1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp1[i] - inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void subLhsScalar(O *out, const I1 *inp1, const I2 inp2, uint64_t n)
{ uint32_t numThreads = blockDim.x * gridDim.x; uint32_t thId = threadIdx.x +
blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp2 - inp1[i];
  }
}

template <typename O, typename I1, typename I2>
__global__ void mulScalar(O *out, const I1 *inp1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp1[i] * inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void divScalar(O *out, const I1 *inp1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp1[i] / inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void divLhsScalar(O *out, const I1 *inp1, const I2 inp2, uint64_t n)
{ uint32_t numThreads = blockDim.x * gridDim.x; uint32_t thId = threadIdx.x +
blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp2 / inp1[i];
  }
}

template <typename O, typename I1, typename I2>
__global__ void powLhsScalar(
    O *out, const I1 *inp1, const I2 inp2, uint64_t n
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::pow(inp2, inp1[i]);
  }
}

*/

const char *setupElementwiseKernel(
    libtcCudaStream &stream, uint64_t n, cudaLaunchConfig_t &config
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

  config.stream = stream.stream;
  config.blockDim = {(uint)props.maxThreadsPerBlock, 1, 1};
  if (n > props.maxThreadsPerBlock) {
    config.gridDim.x =
        (n + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  } else {
    config.blockDim.x = n;
  }

  return nullptr;
}

#include "ewise_arith_gen.inc"