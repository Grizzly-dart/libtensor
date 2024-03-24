#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
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

const char *libtcCudaELU(
    libtcCudaStream &stream, const void *out, const void *inp, uint64_t n,
    double alpha, dtype dtype
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  config.blockDim = {BLOCK_SIZE, 1, 1};
  if (n > BLOCK_SIZE) {
    config.gridDim.x = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  } else {
    config.blockDim.x = n;
  }

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

template <typename T> __global__ void relu(T *out, const T *inp, uint64_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T inpVal = inp[idx];
    out[idx] = inpVal > 0 ? inpVal : inpVal;
  }
}

const char *libtcCudaRELU(
    libtcCudaStream &stream, const void *out, const void *inp, uint64_t n,
    dtype dtype
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  config.blockDim = {BLOCK_SIZE, 1, 1};
  if (n > BLOCK_SIZE) {
    config.gridDim.x = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  } else {
    config.blockDim.x = n;
  }

  if (dtype == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, relu<double>, (double *)out, (const double *)inp, n
    );
  } else if (dtype == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, relu<float>, (float *)out, (const float *)inp, n
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}
