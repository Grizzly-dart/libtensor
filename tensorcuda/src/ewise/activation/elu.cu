#include <cuda_runtime.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <reducers.hpp>
#include <string>

template <typename T>
__global__ void elu(T *out, const T *inp, uint64_t n, const double alpha) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T inpVal = inp[idx];
    out[idx] = inpVal > 0 ? inpVal : alpha * std::expm1(inpVal);
  }
}

#define BLOCK_SIZE 1024

const char *tcuELU(
    tcuStream &stream, const void *out, const void *inp, uint64_t n,
    double alpha, dtype dtype
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr; 
  }

  cudaError_t err;
  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, elu<double>, (double *)out, (const double *)inp, n, alpha
    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, elu<float>, (float *)out, (const float *)inp, n, alpha
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template <typename T>
__global__ void minThreshold(T *out, const T *inp, T threshold, T value, uint64_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T inpVal = inp[idx];
    out[idx] = inpVal > threshold ? inpVal : value;
  }
}

const char *tcuMinThreshold(
    tcuStream &stream, const void *out, const void *inp, void *threshold,
    void *value, uint64_t n, dtype dtype
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr; 
  }

  cudaError_t err;
  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, minThreshold<double>, (double *)out, (const double *)inp,
        *(double *)threshold, *(double *)value, n
    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, minThreshold<float>, (float *)out, (const float *)inp,
        *(float *)threshold, *(float *)value, n
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}
