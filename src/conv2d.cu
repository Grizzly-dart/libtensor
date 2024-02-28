#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "libgpuc_cuda.hpp"
#include "padding.hpp"

// TODO implement batches
// TODO use block.z to handle multiple channels
/// https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
template <typename T>
__global__ void conv2dKernel(T* output, const T* input, const T* kernel, 
    Dim2 outS, Dim2 inS, Dim2 kernS, Dim2 padding, PaddingMode paddingMode, 
    T pad, Dim2 stride, Dim2 dilation) {

  uint32_t output_row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t output_col = blockIdx.x * blockDim.x + threadIdx.x;

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
    for (uint32_t i = 0; i < kernS.y; i++) {
      for (uint32_t j = 0; j < kernS.x; j++) {
        uint32_t input_row = output_row * stride.y + i * dilation.y;
        uint32_t input_col = output_col * stride.x + j * dilation.x;
        if (input_row < inS.y && input_col < inS.x) {
          value += padder(input, inS, padding, pad, input_col, input_row) * kernel[i * kernS.x + j];
        } else {
          // TODO error
        }
      }
    }
    output[output_row * outS.x + output_col] = value;
  }
}

// TODO implement batches
#ifdef FALSE
void conv2d(Tensor out, Tensor in, Tensor kernel, Dim2 padding, PaddingMode paddingMode, double pad, Dim2 stride, Dim2 dilation) {
  
  // TODO
  
  cudaLaunchConfig_t config = {};
  // TODO

  // TODO
  auto err = cudaLaunchKernelEx(&config, conv2dKernel<double>, out.mem, in.mem, in.dim[1]);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}
#endif