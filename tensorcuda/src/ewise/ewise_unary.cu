#include <cmath>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

template <typename O, typename I>
__global__ void neg(O *out, I *inp, uint64_t nel, I minVal, I maxVal) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < nel; i += numThreads) {
    if (inp[i] == minVal) {
      out[i] = maxVal;
    } else {
      out[i] = -inp[i];
    }
  }
}

template <typename T>
__global__ void abs(T *out, T *inp, uint64_t nel, T minVal, T maxVal) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < nel; i += numThreads) {
    if (inp[i] == minVal) {
      out[i] = maxVal;
    } else {
      out[i] = inp[i] >= 0 ? inp[i] : -inp[i];
    }
  }
}

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

extern const char *libtcCudaNeg(
    libtcCudaStream &stream, void *out, void *inp, uint64_t n,
    dtype outType, dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (inType == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, neg<double, double>, (double *)out, (double *)inp, n,
        __DBL_MIN__, __DBL_MAX__
    );
  } else if(inType == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, neg<float, float>, (float *)out, (float *)inp, n,
        __FLT_MIN__, __FLT_MAX__
    );
  } else if(inType == dtype::i64) {
    err = cudaLaunchKernelEx(
        &config, neg<int64_t, int64_t>, (int64_t *)out, (int64_t *)inp, n,
        INT64_MIN, INT64_MAX
    );
  } else if(inType == dtype::u64) {
        err = cudaLaunchKernelEx(
        &config, neg<int64_t, uint64_t>, (int64_t *)out, (uint64_t *)inp, n,
        INT64_MIN, INT64_MAX
    );
  } else if(inType == dtype::i32) {
    err = cudaLaunchKernelEx(
        &config, neg<int32_t, int32_t>, (int32_t *)out, (int32_t *)inp, n,
        INT32_MIN, INT32_MAX
    );
  } else if(inType == dtype::u32) {
    err = cudaLaunchKernelEx(
        &config, neg<int32_t, uint32_t>, (int32_t *)out, (uint32_t *)inp, n,
        INT32_MIN, INT32_MAX
    );
  } else if(inType == dtype::i16) {
    err = cudaLaunchKernelEx(
        &config, neg<int16_t, int16_t>, (int16_t *)out, (int16_t *)inp, n,
        INT16_MIN, INT16_MAX
    );
  } else if(inType == dtype::u16) {
    err = cudaLaunchKernelEx(
        &config, neg<int16_t, uint16_t>, (int16_t *)out, (uint16_t *)inp, n,
        INT16_MIN, INT16_MAX
    );
  } else if(inType == dtype::i8) {
    err = cudaLaunchKernelEx(
        &config, neg<int8_t, int8_t>, (int8_t *)out, (int8_t *)inp, n,
        INT8_MIN, INT8_MAX
    );
  } else if(inType == dtype::u8) {
    err = cudaLaunchKernelEx(
        &config, neg<int8_t, uint8_t>, (int8_t *)out, (uint8_t *)inp, n,
        INT8_MIN, INT8_MAX
    );
  } else {
    return "Unsupported input type";
  }

  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}