#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "libgpuc_cuda.hpp"
#include "padding.hpp"

#define BLOCK_SIZE 16

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
            T inputValue = padder<T>(inputStart, inS, padding, paddingMode, pad, input_col, input_row);
            value += inputValue * kernel[kIdx * kernNel + kRow * kernS.y + kCol];
          }
        } else {
          printf("error\n");  // TODO replace with assert
        }
      }
    }
    output[outId * outS.x * outS.y + output_row * outS.x + output_col] = value;
  }
}

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
  } else if (kernel.dim[1] != in.dim[1] / groups) {
    throw std::string("kernel should have the same number of channels as in");
  }
  // Compute output size based on padding, stride, dilation
  uint32_t outM = (getTensorM(in) + 2 * padding.y - dilation.y * (kernel.dim[2] - 1) - 1) / stride.y + 1;
  uint32_t outN = (getTensorN(in) + 2 * padding.x - dilation.x * (kernel.dim[3] - 1) - 1) / stride.x + 1;
  if (outM != getTensorM(out) || outN != getTensorN(out)) {
    throw std::string("output size is not correct");
  }

  for (uint32_t batch = 0; batch < getTensorB(out); batch++) {
    double* outPtr = out.mem + batch * getTensorM(out) * getTensorN(out) * getTensorC(out);
    double* inPtr = in.mem + batch * getTensorM(in) * getTensorN(in) * getTensorC(in);
    cudaLaunchConfig_t config = {};
    if (outN < BLOCK_SIZE) {
      config.blockDim.x = outN;
      config.gridDim.x = 1;
    } else {
      config.blockDim.x = BLOCK_SIZE;
      config.gridDim.x = (outN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    if (outM < BLOCK_SIZE) {
      config.blockDim.y = outM;
      config.gridDim.y = 1;
    } else {
      config.blockDim.y = BLOCK_SIZE;
      config.gridDim.y = (outM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    config.blockDim.z = outChannels;
    auto err = cudaLaunchKernelEx(&config, conv2dKernel<double>, outPtr, inPtr, kernel.mem, inChannels, groups,
                                  Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))}, Dim2{x : uint32_t(kernel.dim[2]), y : uint32_t(kernel.dim[3])},
                                  padding, paddingMode, pad, stride, dilation);
    if (err != cudaSuccess) {
      throw std::string(cudaGetErrorString(err));
    }
  }
}