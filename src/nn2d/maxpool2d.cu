#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

#include <string>

#include "libgpuc_cuda.hpp"
#include "padding.hpp"

template <typename T>
__global__ void maxPool2DKern(T* output, T* input, Dim2 inS, Dim2 kernS,
    Dim2 padding, PaddingMode PaddingMode, T pad, Dim2 stride, Dim2 dilation) {
  Dim2 outS = {x : gridDim.x * blockDim.x, y : gridDim.y * blockDim.y};
  uint32_t outR = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t outC = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t outId = gridDim.z;
  T* inStart = input + outId * inS.x * inS.y;

  if (outR < outS.y && outC < outS.x) {
    T maxVal = -__DBL_MIN__;
    for (int kRow = 0; kRow < kernS.y; ++kRow) {
      uint32_t inR = outR * stride.y + kRow * dilation.y;
      for (int kCol = 0; kCol < kernS.x; ++kCol) {
        uint32_t inC = outC * stride.x + kCol * dilation.x;
        if (inR < inS.y && inC < inS.x) {
          T inVal = padder<T>(inStart, inS, padding, PaddingMode, pad, inC, inR);
          T val = input[outId * inS.x * inS.y + inR * inS.x + inC];
          maxVal = max(maxVal, val);
        }
      }
    }
    output[outId * outS.x * outS.y + outR * outS.x + outC] = maxVal;
  }
}

template <typename T>
__global__ void maxPool2DKernWarp(T* output, T* input, Dim2 inS, Dim2 kernS,
                                  Dim2 padding, PaddingMode PaddingMode, T pad) {
  Dim2 outS = {x : gridDim.x * blockDim.x, y : gridDim.y * blockDim.y};
  uint32_t outR = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t outC = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t outId = gridDim.z;
  T* inStart = input + outId * inS.x * inS.y;

  if (outR < outS.y && outC < outS.x) {
    T maxVal = -__DBL_MIN__;
    for (int kRow = 0; kRow < kernS.y; ++kRow) {
      uint32_t inR = outR + kRow;
      T inpVal;
      for (int kCol = 0; kCol < kernS.x; ++kCol) {
        uint32_t inC = outC + kCol;
        if (kCol == 0) {
          inpVal = padder<T>(inStart, inS, padding, PaddingMode, pad, inC, inR);
        } else {
          inpVal = __shfl_down_sync(0xFFFFFFFF, inpVal, 1);
        }
        __syncthreads();
        if (threadIdx.x % warpSize == warpSize - 1) {
          inpVal = padder<T>(inStart, inS, padding, PaddingMode, pad, inC, inR);
        }

        if (inR < inS.y + 2 * padding.y && inC < inS.x + 2 * padding.x) {
          maxVal = max(maxVal, inpVal);
        }
      }
    }
    output[outId * outS.x * outS.y + outR * outS.x + outC] = maxVal;
  }
}

// TODO launch batches based on the size of the tensor and GPU VRAM
const char* maxPool2DCKernF64(libtcCudaStream& stream, double* out, double* inp, Dim2 kernS, 
    Size2 outS, Dim2 inS, uint32_t matrices, Dim2 padding, PaddingMode PaddingMode, double pad, 
    Dim2 stride, Dim2 dilation) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if (outS.c < wrapSize) {
    config.blockDim.x = outS.c;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = wrapSize;
    config.gridDim.x = (outS.r + wrapSize - 1) / wrapSize;
  }
  if (outS.r < wrapSize) {
    config.blockDim.y = outS.r;
    config.gridDim.y = 1;
  } else {
    config.blockDim.y = wrapSize;
    config.gridDim.y = (outS.r + wrapSize - 1) / wrapSize;
  }
  config.blockDim.z = matrices;
  cudaError_t err;
  if (stride.x == 1 && stride.y == 1 && dilation.x == 1 && dilation.y == 1) {
    err = cudaLaunchKernelEx(&config, maxPool2DKernWarp<double>, out, inp, Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))},
      kernS, padding, PaddingMode, pad);
  } else {
    err = cudaLaunchKernelEx(&config, maxPool2DKern<double>, out, inp, Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))},
                             kernS, padding, PaddingMode, pad, stride, dilation);
  }
  if (err != cudaSuccess) {
    throw std::string("failed to launch kernel");
  }
}

void maxPool2D(Tensor out, Tensor in, Dim2 kernS, Dim2 padding, PaddingMode PaddingMode, double pad, Dim2 stride, Dim2 dilation) {
  uint32_t wrapSize = 32;  // TODO find this
  if (out.ndim != in.ndim) {
    throw std::string("out and in should have the same number of dimensions");
  } else if (getTensorC(out) != getTensorC(in)) {
    throw std::string("out and in should have the same number of channels");
  } else if (getTensorB(out) != getTensorB(in)) {
    throw std::string("out and in should have the same batch size");
  }
  uint32_t channels = getTensorC(in);
  // Compute output size based on padding, stride, dilation
  uint32_t outM = (getTensorM(in) + 2 * padding.y - dilation.y * (kernS.y - 1) - 1) / stride.y + 1;
  uint32_t outN = (getTensorN(in) + 2 * padding.x - dilation.x * (kernS.x - 1) - 1) / stride.x + 1;
  if (outM != getTensorM(out) || outN != getTensorN(out)) {
    throw std::string("output size is not correct");
  }

  // TODO for smaller tensors, try to launch multiple batches at once
  for (uint32_t b = 0; b < getTensorB(in); ++b) {
    double* outPtr = out.mem + b * getTensorM(out) * getTensorN(out) * channels;
    double* inPtr = in.mem + b * getTensorM(in) * getTensorN(in) * channels;
    cudaLaunchConfig_t config = {};
    if (outN < wrapSize) {
      config.blockDim.x = outN;
      config.gridDim.x = 1;
    } else {
      config.blockDim.x = wrapSize;
      config.gridDim.x = (outN + wrapSize - 1) / wrapSize;
    }
    if (outM < wrapSize) {
      config.blockDim.y = outM;
      config.gridDim.y = 1;
    } else {
      config.blockDim.y = wrapSize;
      config.gridDim.y = (outM + wrapSize - 1) / wrapSize;
    }
    config.blockDim.z = channels;
    cudaError_t err;
    if (stride.x == 1 && stride.y == 1 && dilation.x == 1 && dilation.y == 1) {
      err = cudaLaunchKernelEx(&config, maxPool2DKernWarp<double>, outPtr, inPtr, Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))},
                               kernS, padding, PaddingMode, pad);
    } else {
      err = cudaLaunchKernelEx(&config, maxPool2DKern<double>, outPtr, inPtr, Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))},
                               kernS, padding, PaddingMode, pad, stride, dilation);
    }
    if (err != cudaSuccess) {
      throw std::string("failed to launch kernel");
    }
  }
}
