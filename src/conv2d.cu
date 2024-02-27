#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "libgpuc_cuda.hpp"
#include "padding.hpp"

/// https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
template <typename T>
__global__ void conv2dKernel(T* output, const T* input, const T* kernel, Dim2 outS, Dim2 inS, Dim2 kernS, Dim2 padding, PaddingMode paddingMode, T pad, Dim2 stride, Dim2 dilation) {
  int output_row = blockIdx.y * blockDim.y + threadIdx.y;
  int output_col = blockIdx.x * blockDim.x + threadIdx.x;

  auto padder = constant2DPadding<T>;
  switch (paddingMode) {
    case CONSTANT:
      padder = constant2DPadding<T>;
      break;
    case CIRCULAR:
      padder = circular2DPadding<T>;
      break;
    case REFLECT:
      padder = reflect2DPadding<T>;
      break;
    case REPLICATION:
      padder = replicate2DPadding<T>;
      break;
  }

  if (output_row < outS.y && output_col < outS.x) {
    T value = 0;
    for (int i = 0; i < kernS.y; i++) {
      for (int j = 0; j < kernS.x; j++) {
        int input_row = output_row * stride.y + i * dilation.y;
        int input_col = output_col * stride.x + j * dilation.x;
        if (input_row >= 0 && input_row < inS.y && input_col >= 0 && input_col < inS.x) {
          value += padder(input, inS, padding, pad, input_col, input_row) * kernel[i * kernS.x + j];
        } else {
          // TODO error
        }
      }
    }
    output[output_row * outS.x + output_col] = value;
  }
}

/*
void conv2d(Tensor out, Tensor in, Tensor kernel, Dim2 padding, PaddingMode paddingMode, double pad, Dim2 stride, Dim2 dilation) {
  
  // TODO
  
  cudaLaunchConfig_t config = {};
  if(in.dim[1] < MAX_THREADS_PER_BLOCK) {
    config.blockDim.x = in.dim[1];
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = MAX_THREADS_PER_BLOCK;
    config.gridDim.x = (in.dim[1] + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
  }
  config.gridDim.y = in.dim[0];

  auto err = cudaLaunchKernelEx(&config, conv2dKernel<double>, out.mem, in.mem, in.dim[1]);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}
*/