#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

template <typename T, typename I>
__global__ void sum2d(T *out, I *inp, uint64_t numCols) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t row = blockIdx.y;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;
  T sum = 0;
  for (uint64_t col = thId; col < numCols; col += numThreads) {
    if (col < numCols) {
      uint32_t idx = row * numCols + col;
      sum += inp[idx];
    }
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  __syncthreads();

  uint8_t lane = threadIdx.x % warpSize;
  uint8_t warp = threadIdx.x / warpSize;

  __shared__ T sdata[32];
  if (warp == 0) {
    sdata[threadIdx.x] = 0;
  }
  __syncthreads();

  if (lane == 0) {
    sdata[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    sum = sdata[lane];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  } else {
    sum = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(out + row, sum);
  }
}

const char *libtcCudaSum2d(
    libtcCudaStream &stream, void *out, void *inp, Dim2 inpS,
    dtype outType, dtype inpType
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  uint32_t numThreads = props.multiProcessorCount * 128;
  if (numThreads > inpS.c) {
    numThreads = inpS.c;
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if (numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x =
        (numThreads + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }
  config.gridDim.y = inpS.r;

  if (outType == dtype::f64) {
    if (inpType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, double>, (double *)out, (double *)inp, inpS.c
      );
    } else if (inpType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, float>, (double *)out, (float *)inp, inpS.c
      );
    } else if (inpType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, uint64_t>, (double *)out, (uint64_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, int64_t>, (double *)out, (int64_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, uint32_t>, (double *)out, (uint32_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, int32_t>, (double *)out, (int32_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, uint16_t>, (double *)out, (uint16_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, int16_t>, (double *)out, (int16_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, uint8_t>, (double *)out, (uint8_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, sum2d<double, int8_t>, (double *)out, (int8_t *)inp, inpS.c
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inpType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, double>, (float *)out, (double *)inp, inpS.c
      );
    } else if (inpType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, float>, (float *)out, (float *)inp, inpS.c
      );
    } else if (inpType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, uint64_t>, (float *)out, (uint64_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, int64_t>, (float *)out, (int64_t *)inp, inpS.c
      );
    } else if (inpType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, uint32_t>, (float *)out, (uint32_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, int32_t>, (float *)out, (int32_t *)inp, inpS.c
      );
    } else if (inpType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, uint16_t>, (float *)out, (uint16_t *)inp,
          inpS.c
      );
    } else if (inpType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, int16_t>, (float *)out, (int16_t *)inp, inpS.c
      );
    } else if (inpType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, uint8_t>, (float *)out, (uint8_t *)inp, inpS.c
      );
    } else if (inpType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, sum2d<float, int8_t>, (float *)out, (int8_t *)inp, inpS.c
      );
    } else {
      return "Unsupported input type";
    }
  } else {
    return "Unsupported output type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}