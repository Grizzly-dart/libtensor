#include <cuda_runtime.h>

#include "libgpuc_cuda.hpp"

template <typename T>
__global__ void maxPool2D(const T* input, T* output, Dim2 inS, Dim2 kernS, Dim2 padding, Dim2 stride, Dim2 dilation) {
  Dim2 outS = {x : gridDim.x * blockDim.x, y : gridDim.y * blockDim.y};

  int outIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (outIndex < outS.width * outS.height) {
    int outX = outIndex % outS.width;
    int outY = outIndex / outS.width;
 
    int inXStart = outX * stride.width - padding.width;
    int inYStart = outY * stride.height - padding.height;
    int inXEnd = std::min(inXStart + (kernS.width - 1) * dilation.width + 1, inS.width);
    int inYEnd = std::min(inYStart + (kernS.height - 1) * dilation.height + 1, inS.height);
    inXStart = std::max(inXStart, 0);
    inYStart = std::max(inYStart, 0);

    float maxVal = -FLT_MAX;
    for (int inY = inYStart; inY < inYEnd; inY += dilation.height) {
      for (int inX = inXStart; inX < inXEnd; inX += dilation.width) {
        int inIndex = inY * inS.width + inX;
        maxVal = std::max(maxVal, input[inIndex]);
      }
    }

    int outIndexFlat = outY * outS.width + outX;
    output[outIndexFlat] = maxVal;
  }
}