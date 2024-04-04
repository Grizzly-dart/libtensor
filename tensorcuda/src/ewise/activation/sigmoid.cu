#include <cuda_runtime.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <reducers.hpp>
#include <string>

template <typename T> __global__ void sigmoid(T *out, T *inp, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = 1.0 / (1.0 + exp(-inp[idx]));
  }
}

extern const char *tcuSigmoid(
    tcuStream &stream, void *out, void *inp, uint64_t n,
    dtype dtype
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr; 
  }

  cudaError_t err;
  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, sigmoid<double>, (double *)out, (double *)inp, n
    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, sigmoid<float>, (float *)out, (float *)inp, n
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template <typename T> __global__ void silu(T *out, T *inp, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T x = inp[idx];
    out[idx] =  x / (1.0 + exp(-x));
  }
}

const char *tcuSiLU(
    tcuStream &stream, const void *out, void *inp, uint64_t n,
    dtype dtype
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr; 
  }

  cudaError_t err;
  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, silu<double>, (double *)out, (double *)inp, n
    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, silu<float>, (float *)out, (float *)inp, n
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}