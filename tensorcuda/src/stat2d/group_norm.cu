#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

template <typename T>
__global__ void rowNorm(
    T *input, T *output, uint64_t elements, 
    double epsilon
) {
  // TODO
}

const char *libtcRowNorm(
    libtcCudaStream &stream, void *out, void *inp, uint64_t cols, uint32_t rows,
    double epsilon
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

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  // TODO
}
