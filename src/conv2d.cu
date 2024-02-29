#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "libgpuc_cuda.hpp"
#include "padding.hpp"

// TODO if kernel size < 16, use shared memory

/// https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
template <typename T>
__global__ void conv2dKernel(T* output, T* input, T* kernel, uint32_t inChannels, uint32_t groups,
                             Dim2 inS, Dim2 kernS,
                             Dim2 padding, PaddingMode paddingMode, T pad, Dim2 stride, Dim2 dilation) {
  uint32_t kernNel = kernS.x * kernS.y;
  uint32_t output_row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t output_col = blockIdx.x * blockDim.x + threadIdx.x;
  Dim2 outS = {x : gridDim.x * blockDim.x, y : gridDim.y * blockDim.y};
  uint32_t outId = blockIdx.z;
  uint32_t outChannels = gridDim.z;
  uint32_t groupLen = inChannels / groups;

  auto padder = constant2DPadding<T>;
  if (paddingMode == CONSTANT) {
    padder = constant2DPadding<T>;
  } else if (paddingMode == CIRCULAR) {
    padder = circular2DPadding<T>;
  } else if (paddingMode == REFLECT) {
    padder = reflect2DPadding<T>;
  } else if (paddingMode == REPLICATION) {
    padder = replicate2DPadding<T>;
  }

  if (output_row < outS.y && output_col < outS.x) {
    T value = 0;
    for (uint32_t kRow = 0; kRow < kernS.y; kRow++) {
      for (uint32_t kCol = 0; kCol < kernS.x; kCol++) {
        uint32_t input_row = output_row * stride.y + kRow * dilation.y;
        uint32_t input_col = output_col * stride.x + kCol * dilation.x;
        if (input_row < inS.y && input_col < inS.x) {
          uint32_t firstInpChannelId = (outChannels / groups) / groupLen;
          for (uint32_t g = 0; g < groupLen; g++) {
            T* inputStart = input + (firstInpChannelId + g) * inS.x * inS.y;
            uint32_t kIdx = outId * groupLen + g;
            T inputValue = padder(inputStart, inS, padding, paddingMode, pad, input_col, input_row);
            value += inputValue * kernel[kIdx * kernNel + kRow * kernS.y + kCol];
          }
        } else {
          // TODO error
        }
      }
    }
    output[outId * outS.x * outS.y + output_row * outS.x + output_col] = value;
  }
}

// TODO implement batches
void conv2d(Tensor out, Tensor in, Tensor kernel, uint32_t groups,
            Dim2 padding, PaddingMode paddingMode, double pad, Dim2 stride, Dim2 dilation) {
  if (groups == 0) {
    groups = 1;
  }
  if (out.ndim != in.ndim) {
    throw std::string("out and in should have the same number of dimensions");
  } else if (getTensorB(out) != getTensorB(in)) {
    throw std::string("out and in should have the same batch size");
  }
  const uint32_t outChannels = getTensorC(out);
  const uint32_t inChannels = getTensorC(in);
  if (groups > 1) {
    if (outChannels % groups != 0) {
      throw std::string("out channels should be divisible by groups");
    }
    if (inChannels % groups != 0) {
      throw std::string("in channels should be divisible by groups");
    }
  }
  if (kernel.ndim != 4) {
    throw std::string("kernel should have 4 dimensions");
  }
  if (kernel.dim[0] != outChannels) {
    throw std::string("kernel should have the same number of channels as out");
  }
  if (kernel.dim[1] != in.dim[1] / groups) {
    throw std::string("kernel should have the same number of channels as in");
  }

  // TODO handle batches

  cudaLaunchConfig_t config = {};
  // TODO

  // TODO
  auto err = cudaLaunchKernelEx(&config, conv2dKernel<double>, out.mem, in.mem,
                                padding, paddingMode, pad, stride, dilation);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}