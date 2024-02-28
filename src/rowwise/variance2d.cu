#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

template <typename T>
class Variance {
public:
  T mean = 0;
  uint32_t n = 0;
  T m2 = 0;

  __device__ void comsume(T sample) {
    n++;
    T delta = sample - mean;
    mean += delta / n;
    m2 += delta * (sample - mean);
  }

  __device__ void merge(const Variance<T>& other) {
    if (other.n == 0) {
      return;
    }
    if (n == 0) {
      mean = other.mean;
      n = other.n;
      m2 = other.m2;
      return;
    }

    n = n + other.n;
    T delta = other.mean - mean;
    mean += delta * other.n / n;
    m2 += other.m2 + delta * delta * n * other.n / (n + other.n);
  }

  __device__ Variance<T> shfl_down(int offset) {
    Variance<T> other;
    other.mean = __shfl_down_sync(0xffffffff, mean, offset);
    other.n = __shfl_down_sync(0xffffffff, n, offset);
    other.m2 = __shfl_down_sync(0xffffffff, m2, offset);
    return other;
  }
};

template <typename T>
__global__ void variance2DKernel(T* out, T* in, uint32_t numCols) {
  uint32_t numThreads = blockDim.x;
  // uint32_t numRows = gridDim.y;
  uint32_t row = blockIdx.x;
  Variance<T> record{};
  for (uint32_t col = threadIdx.x; col < numCols; col += numThreads) {
    uint32_t idx = row * numCols + col;
    record.comsume(in[idx]);
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    record.merge(record.shfl_down(offset));
  }
  __syncthreads();

  uint8_t lane = threadIdx.x % warpSize;
  uint8_t warp = threadIdx.x / warpSize;

  __shared__ Variance<T> sdata[32];

  if (lane == 0) {
    sdata[warp] = record;
  }
  __syncthreads();

  if (warp == 0) {
    record = (lane < blockDim.x / warpSize) ? sdata[lane] : Variance<T>{};
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      record.merge(record.shfl_down(offset));
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    out[row] = record.m2;
  }
}

void variance2DTensor(Tensor out, Tensor in) {
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

  auto err = cudaLaunchKernelEx(&config, variance2DKernel<double>, out.mem, in.mem, in.dim[1]);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}