#include <string>
#include <cstdint>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

// TODO implement stride and split
/// Adds two tensors
template<typename T>
__global__ void add2DKernel(T* out, const T* in1, const T* in2, uint32_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= n) return;
  out[i] = in1[i] + in2[i];
}

const char* libtcCudaAddCkern(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint32_t n) {
  uint32_t threads = n;
  uint32_t blocks = 1;
  if (n > MAX_THREADS_PER_BLOCK) {
    threads = MAX_THREADS_PER_BLOCK;
    blocks = (n + threads - 1) / threads;
  }
  cudaLaunchConfig_t config = {
    .stream = stream.stream,
  };
  config.blockDim.x = threads;
  config.gridDim.x = blocks;
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  err = cudaLaunchKernelEx(&config, add2DKernel<double>, out, in1, in2, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  // TODO remove
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

void add2D(Tensor out, Tensor in1, Tensor in2) {
  uint32_t n = getTensorNel(in1);
  if (n != getTensorNel(in2) || n != getTensorNel(out))
    throw std::string("Size mismatch");

  uint32_t threads = n;
  uint32_t blocks = 1;
  if (n > MAX_THREADS_PER_BLOCK) {
    threads = MAX_THREADS_PER_BLOCK;
    blocks = (n + threads - 1) / threads;
  }
  cudaLaunchConfig_t config = {};
  config.blockDim.x = threads;
  config.gridDim.x = blocks;
  auto err = cudaLaunchKernelEx(&config, add2DKernel<double>, out.mem, in1.mem, in2.mem, n);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}