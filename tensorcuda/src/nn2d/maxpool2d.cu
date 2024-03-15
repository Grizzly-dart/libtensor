#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

#include <string>

#include "libgpuc_cuda.hpp"
#include "padding.hpp"

template <typename T>
__global__ void maxPool2DKern(T* out, T* inp, Dim2 inpS, Dim2 kernS,
    Dim2 padding, PadMode padMode, T pad, Dim2 stride, Dim2 dilation) {
  Dim2 outS = {r : gridDim.y * blockDim.y, c : gridDim.x * blockDim.x};
  uint32_t outR = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t outC = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t outId = blockIdx.z;
  T* inStart = inp + outId * inpS.c * inpS.r;

  if (outR < outS.r && outC < outS.c) {
    T maxVal = -__DBL_MIN__;
    for (int kRow = 0; kRow < kernS.r; ++kRow) {
      uint32_t inR = outR * stride.r + kRow * dilation.r;
      for (int kCol = 0; kCol < kernS.c; ++kCol) {
        uint32_t inC = outC * stride.c + kCol * dilation.c;
        if (inR < inpS.r && inC < inpS.c) {
          T inVal = padder<T>(inStart, inpS, padding, padMode, pad, inC, inR);
          T val = inp[outId * inpS.c * inpS.r + inR * inpS.c + inC];
          maxVal = max(maxVal, val);
        }
      }
    }
    out[outId * outS.c * outS.r + outR * outS.c + outC] = maxVal;
  }
}

const char* libtcCudaMaxPool2D(libtcCudaStream& stream, double* out, double* inp,
    Dim2 kernS, Dim2 outS, Dim2 inpS, uint32_t matrices, Dim2 padding, 
    PadMode padMode, double pad, Dim2 stride, Dim2 dilation) {
  // TODO validate outS

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
  if (outS.c < props.warpSize) {
    config.blockDim.x = outS.c;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.warpSize;
    config.gridDim.x = (outS.c + props.warpSize - 1) / props.warpSize;
  }
  if (outS.r < props.warpSize) {
    config.blockDim.y = outS.r;
    config.gridDim.y = 1;
  } else {
    config.blockDim.y = props.warpSize;
    config.gridDim.y = (outS.r + props.warpSize - 1) / props.warpSize;
  }
  config.gridDim.z = matrices;
  // TODO launch batches based on the size of the tensor and GPU VRAM
  err = cudaLaunchKernelEx(&config, maxPool2DKern<double>, out, inp, 
    inpS, kernS, padding, padMode, pad, stride, dilation);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

/*
void maxPool2D(Tensor out, Tensor in, Dim2 kernS, Dim2 padding, PadMode PadMode, double pad, Dim2 stride, Dim2 dilation) {
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
                               kernS, padding, PadMode, pad);
    } else {
      err = cudaLaunchKernelEx(&config, maxPool2DKern<double>, outPtr, inPtr, Dim2{x : uint32_t(getTensorM(in)), y : uint32_t(getTensorN(in))},
                               kernS, padding, PadMode, pad, stride, dilation);
    }
    if (err != cudaSuccess) {
      throw std::string(cudaGetErrorString(err));
    }
  }
}
*/