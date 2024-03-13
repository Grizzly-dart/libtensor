#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <cassert>

#include "libgpuc_cuda.hpp"
#include "padding.hpp"

#define BLOCK_SIZE 32

// TODO if kernel size < 16, use shared memory
 
/// https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
template <typename T>
__global__ void conv2dKernel(T* output, T* input, T* kernel, uint32_t groups, Dim3 outS,
    Dim3 inpS, Dim2 kernS, Dim2 padding, PaddingMode padMode, T pad, Dim2 stride,
    Dim2 dilation) {
  uint32_t kernNel = kernS.r * kernS.c;
  uint32_t outR = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t outC = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t outId = blockIdx.z;
  uint32_t outChannel = outId % outS.ch;
  uint32_t groupLen = inpS.ch / groups;
  uint32_t firstInpChannelId = (outChannel / groups) / groupLen;

  if (outR < outS.r && outC < outS.c) {
    T value = 0;
    for (uint32_t kRow = 0; kRow < kernS.r; kRow++) {
      uint32_t inR = outR * stride.r + kRow * dilation.r;
      for (uint32_t kCol = 0; kCol < kernS.c; kCol++) {
        uint32_t inC = outC * stride.c + kCol * dilation.c;
        if (inR < inpS.r + 2 * padding.r && inC < inpS.c + 2 * padding.c) {
          for (uint32_t g = 0; g < groupLen; g++) {
            T* inputStart = input + (firstInpChannelId + g) * inpS.c * inpS.r;
            uint32_t kIdx = outId * groupLen + g;
            T inputValue = padder<T>(inputStart, inpS.toDim2(), padding, padMode, pad, inC, inR);
            value += inputValue * kernel[kIdx * kernNel + kRow * kernS.c + kCol];
          }
        } else {
          assert(inR < inpS.r + 2 * padding.r && inC < inpS.c + 2 * padding.c);
        }
      }
    }
    output[outId * outS.r * outS.c + outR * outS.c + outC] = value;
  }
}

const char* libtcCudaConv2d(libtcCudaStream stream, double* out, double* inp, double* kernel, 
    uint32_t batches, Dim3 outS, Dim3 inpS, Dim2 kernS, uint32_t groups, Dim2 padding, 
    PaddingMode padMode, double pad, Dim2 stride, Dim2 dilation) {
  // TODO validate outS
  if (groups == 0) {
    groups = 1;
  }

  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if (outS.c < BLOCK_SIZE) {
    config.blockDim.x = outS.c;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = BLOCK_SIZE;
    config.gridDim.x = (outS.c + BLOCK_SIZE - 1) / BLOCK_SIZE;
  }
  if (outS.r < BLOCK_SIZE) {
    config.blockDim.y = outS.r;
    config.gridDim.y = 1;
  } else {
    config.blockDim.y = BLOCK_SIZE;
    config.gridDim.y = (outS.r + BLOCK_SIZE - 1) / BLOCK_SIZE;
  }
  config.gridDim.z = batches * outS.ch;

  err = cudaLaunchKernelEx(&config, conv2dKernel<double>, out, inp, kernel, groups,
    outS, inpS, kernS, padding, padMode, pad, stride, dilation);
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