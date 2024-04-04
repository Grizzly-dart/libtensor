#include <cuda_runtime.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <reducers.hpp>
#include <string>

template <typename T>
__global__ void softplus(
    T *out, T *inp, uint64_t size, int32_t beta, int32_t threshold
) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T x = inp[idx];
    T xBeta = x * beta;
    if (xBeta > threshold) {
      out[idx] = x;
    } else {
      out[idx] = std::log1p(std::exp(xBeta)) / beta;
    }
  }
}

const char *tcuSoftplus(
    tcuStream &stream, const void *out, void *inp, uint64_t n,
    int32_t beta, int32_t threshold, dtype dtype
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr; 
  }

  cudaError_t err;
  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, softplus<double>, (double *)out, (double *)inp, n,
        beta, threshold
    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, softplus<float>, (float *)out, (float *)inp, n, beta,
        threshold
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template <typename T>
__global__ void softsign(T *out, T *inp, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T x = inp[idx];
    out[idx] = x / (1 + std::abs(x));
  }
}

const char *tcuSoftsign(
    tcuStream &stream, const void *out, void *inp, uint64_t n, dtype dtype
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr; 
  }

  cudaError_t err;
  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, softsign<double>, (double *)out, (double *)inp, n

    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, softsign<float>, (float *)out, (float *)inp, n
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template <typename T>
__global__ void mish(T *out, T *inp, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T x = inp[idx];
    out[idx] = x * std::tanh(std::log1p(std::exp(x)));
  }
}

const char *tcuMish(
    tcuStream &stream, const void *out, void *inp, uint64_t n, dtype dtype
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr; 
  }

  cudaError_t err;
  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, mish<double>, (double *)out, (double *)inp, n

    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, mish<float>, (float *)out, (float *)inp, n
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}