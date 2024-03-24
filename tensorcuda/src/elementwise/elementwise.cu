#include <string>
#include <cstdint>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

// TODO implement stride and split
/// Adds two tensors
template<typename O, typename I1, typename I2>
__global__ void add2(O* out, const I1* in1, const I2* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] + in2[i];
  }
}

template<typename O, typename I1, typename I2>
__global__ void sub2(O* out, const I1* in1, const I2* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] - in2[i];
  }
}

template<typename O, typename I1, typename I2>
__global__ void mul2(O* out, const I1* in1, const I2* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] * in2[i];
  }
}

template<typename O, typename I1, typename I2>
__global__ void div2(O* out, const I1* in1, const I2* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] / in2[i];
  }
}

template<typename O, typename I>
__global__ void cast(O* out, const I* inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp[i];
  }
}

const char* setupElementwiseKernel(libtcCudaStream& stream, uint64_t n, cudaLaunchConfig_t& config) {
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
  if(numThreads > n) {
    numThreads = n;
  }
  
  config.stream = stream.stream;
  if(numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x = (numThreads + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }

  return nullptr;
}

#include "elementwise_gen.inc"

extern const char* libtcCudaAdd2_f64_f64_f64(libtcCudaStream& stream, void* out, const void* in1, const void* in2, uint64_t n) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  auto err = cudaLaunchKernelEx(&config, add2<double, double, double>, (double*)out, (double*)in1, (double*)in2, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

extern const char* libtcCudaCast_f64_f32(libtcCudaStream& stream, void* out, const void* inp, uint64_t n) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  auto err = cudaLaunchKernelEx(&config, cast<double, float>, (double*)out, (float*)inp, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}