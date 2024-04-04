#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <string>

template <typename T, typename I>
__global__ void pickRows(
    T *out, const T *in, const I *indices, size_t rowSize
) {
  uint32_t row = blockIdx.x;
  uint64_t index = indices[row];
  for (uint32_t i = threadIdx.x; i < rowSize; i += blockDim.x) {
    out[row * rowSize + i] = in[index * rowSize + i];
  }
}

const char *tcuPickRows(
    tcuStream &stream, void *out, const void *inp, const void *indices,
    Dim2 size, dtype type, dtype itype
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  config.blockDim = {32, 1, 1};
  config.gridDim = {size.r, 1, 1};

  if (type == f32) {
    if (itype == u32) {
      err = cudaLaunchKernelEx(
          &config, pickRows<float, uint32_t>, (float *)out, (float *)inp,
          (uint32_t *)indices, size.c
      );
    } else if (itype == u64) {
      err = cudaLaunchKernelEx(
          &config, pickRows<float, uint64_t>, (float *)out, (float *)inp,
          (uint64_t *)indices, size.c
      );
    } else if (itype == u16) {
      err = cudaLaunchKernelEx(
          &config, pickRows<float, uint16_t>, (float *)out, (float *)inp,
          (uint16_t *)indices, size.c
      );
    } else if (itype == u8) {
      err = cudaLaunchKernelEx(
          &config, pickRows<float, uint8_t>, (float *)out, (float *)inp,
          (uint8_t *)indices, size.c
      );
    } else {
      return "Invalid index type";
    }
  } else if (type == f64) {
    if (itype == u32) {
      err = cudaLaunchKernelEx(
          &config, pickRows<double, uint32_t>, (double *)out, (double *)inp,
          (uint32_t *)indices, size.c
      );
    } else if (itype == u64) {
      err = cudaLaunchKernelEx(
          &config, pickRows<double, uint64_t>, (double *)out, (double *)inp,
          (uint64_t *)indices, size.c
      );
    } else if (itype == u16) {
      err = cudaLaunchKernelEx(
          &config, pickRows<double, uint16_t>, (double *)out, (double *)inp,
          (uint16_t *)indices, size.c
      );
    } else if (itype == u8) {
      err = cudaLaunchKernelEx(
          &config, pickRows<double, uint8_t>, (double *)out, (double *)inp,
          (uint8_t *)indices, size.c
      );
    } else {
      return "Invalid index type";
    }
  } else {
    return "Invalid data type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}