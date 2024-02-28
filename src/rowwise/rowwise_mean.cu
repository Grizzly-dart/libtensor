#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

template <typename T>
class Mean {
public:
  T mean = 0;
  uint32_t n = 0;

  __global__ void comsume(T sample) {
    n++;
    T delta = sample - mean;
    mean += delta / n;
  }

  __global__ void merge(const Mean<T>& other) {
    if (other.n == 0) {
      return;
    }
    if (n == 0) {
      mean = other.mean;
      n = other.n;
      return;
    }

    n = n + other.n;
    T delta = other.mean - mean;
    mean += delta * other.n / n;
  }
}

template <typename T>
__global__ void mean2DKernel(T* out, T* in, uint32_t numCols) {
  uint32_t numThreads = blockDim.x;
  // uint32_t numRows = gridDim.y;
  uint32_t row = blockIdx.x;
  Mean record();
  for (uint32_t col = threadIdx.x; col < numCols; col += numThreads) {
    uint32_t idx = row * numCols + col;
    record.comsume(in[idx]);
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    record = record.merge(__shfl_down_sync(0xffffffff, record, offset));
  }
  __syncthreads();

  uint8_t lane = threadIdx.x % warpSize;
  uint8_t warp = threadIdx.x / warpSize;

  __shared__ Mean<T> sdata[32];

  if (lane == 0) {
    sdata[warp] = record;
  }
  __syncthreads();

  if (warp == 0) {
    sum = (lane < blockDim.x / warpSize) ? sdata[lane] : 0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      record = record.merge(__shfl_down_sync(0xffffffff, record, offset));
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    out[row] = sum.mean;
  }
}

void mean2DTensor(Tensor out, Tensor in) {
  if (in.ndim != 2) {
    throw std::string("Input tensor must be 2D");
  } else if (out.ndim != 1) {
    throw std::string("Output tensor must be 1D");
  } else if (out.dim[0] != in.dim[0]) {
    throw std::string("Size mismatch between input and output tensors");
  }

  cudaLaunchConfig_t config = {};
  if(in.dim[1] < MAX_THREADS_PER_BLOCK) {
    config.blockDim.x = in.dim[1];
  } else {
    config.blockDim.x = MAX_THREADS_PER_BLOCK;
  }
  config.gridDim.x = in.dim[0];

  auto err = cudaLaunchKernelEx(&config, sum2DKernel<double>, out.mem, in.mem, in.dim[1]);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}