#include <cuda_runtime.h>
#include <limits.h>

#include "libgpuc_cuda.hpp"

template <typename T>
__global__ void maxPool2DKern(T* output, const T* input, Dim2 inS, Dim2 kernS,
                                Dim2 padding, PaddingMode PaddingMode, T pad, Dim2 stride, Dim2 dilation) {
  Dim2 outS = {x : gridDim.x * blockDim.x, y : gridDim.y * blockDim.y};
  uint32_t outR = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t outC = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t outId = gridDim.z;
  T* inStart = input + outId * inS.x * inS.y;

  if (outR < outS.y && outC < outS.x) {
    T maxVal = -DBL_MIN;
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
__global__ void maxPool2DKernWarp(T* output, const T* input, Dim2 inS, Dim2 kernS,
                                    Dim2 padding, PaddingMode PaddingMode, T pad) {
  Dim2 outS = {x : gridDim.x * blockDim.x, y : gridDim.y * blockDim.y};
  uint32_t outR = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t outC = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t outId = gridDim.z;
  T* inStart = input + outId * inS.x * inS.y;

  if (outR < outS.y && outC < outS.x) {
    T maxVal = -DBL_MIN;
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

void maxPool2DCKernDouble(double* output, double* input, Dim2 inS, Dim2 kernS, Dim2 padding, PaddingMode PaddingMode, double pad, Dim2 stride, Dim2 dilation) {
  err = cudaLaunchKernelEx(&config, maxPool2DKern<double>, output, input, inS, kernS, padding, PaddingMode, pad, stride, dilation);
  if (err != cudaSuccess) {
    throw std::string("failed to launch kernel");
  }
}

void maxPool2DCKernWarpDouble(T* output, const T* input, Dim2 inS, Dim2 kernS,
                        Dim2 padding, PaddingMode PaddingMode, T pad) {
  err = cudaLaunchKernelEx(&config, maxPool2DKernWarp<double>, output, input, inS, kernS, padding, PaddingMode, pad);
  if (err != cudaSuccess) {
    throw std::string("failed to launch kernel");
  }
}

void maxPool2D(Tensor output, Tensor input, Dim2 kernS, Dim2 padding, PaddingMode PaddingMode, double pad, Dim2 stride, Dim2 dilation) {
  if (out.ndim != in.ndim) {
    throw std::string("out and in should have the same number of dimensions");
  } else if (getTensorC(out) != getTensorC(in)) {
    throw std::string("out and in should have the same number of channels");
  } else if (getTensorB(out) != getTensorB(in)) {
    throw std::string("out and in should have the same batch size");
  }
  uint32_t channels = getTensorC(in);
  // Compute output size based on padding, stride, dilation
  uint32_t outM = (getTensorM(in) + 2 * padding.y - dilation.y * (kernel.dim[2] - 1) - 1) / stride.y + 1;
  uint32_t outN = (getTensorN(in) + 2 * padding.x - dilation.x * (kernel.dim[3] - 1) - 1) / stride.x + 1;
  if (outM != getTensorM(out) || outN != getTensorN(out)) {
    throw std::string("output size is not correct");
  }

  // TODO for smaller tensors, try to launch multiple batches at once
  uint32_t processLen;
  for (uint32_t b = 0; b < getTensorB(in); ++b) {
    double* outPtr = out.mem + batch * getTensorM(out) * getTensorN(out) * channels;
    double* inPtr = in.mem + batch * getTensorM(in) * getTensorN(in) * channels;
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
    config.blockDim.z = channels;
    auto err;
    if (stride.x == 1 && stride.y == 1 && dilation.x == 1 && dilation.y == 1) {
      err = cudaLaunchKernelEx(&config, maxPool2DKernWarp, outPtr, inPtr, Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))},
                               kernS, padding, PaddingMode, pad);
    } else {
      err = cudaLaunchKernelEx(&config, maxPool2DKern, outPtr, inPtr, Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))},
                               kernS, padding, PaddingMode, pad, stride, dilation);
    }
    if (err != cudaSuccess) {
      throw std::string("failed to launch kernel");
    }
  }
}
