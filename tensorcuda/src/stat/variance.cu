#include <cuda_runtime.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <reducers.hpp>
#include <string>

template <typename I>
__global__ void variance(
    double *out, I *inp, uint64_t nel, uint64_t correction
) {
  uint32_t numThreads = blockDim.x;

  Variance<double> record{};
  for (uint64_t col = threadIdx.x; col < nel; col += numThreads) {
    record.consume(inp[col]);
  }
  __syncthreads();

  // Do warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    record.merge(record.shfl_down(offset));
  }
  __syncthreads();

  uint8_t lane = threadIdx.x % warpSize;
  uint8_t warp = threadIdx.x / warpSize;

  __shared__ Variance<double> sdata[32];
  if (warp == 0) {
    sdata[threadIdx.x] = {0};
  }
  __syncthreads();

  if (lane == 0) {
    sdata[warp] = record;
  }
  __syncthreads();

  if (warp == 0) {
    record = sdata[lane];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      record.merge(record.shfl_down(offset));
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *out = record.m2 / (nel - correction);
  }
}

const char *tcuVariance(
    tcuStream &stream, double *out, void *inp, uint64_t nel,
    uint64_t correction, dtype inpType
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if (nel < prop.maxThreadsPerBlock) {
    config.blockDim.x = nel;
  } else {
    config.blockDim.x = prop.maxThreadsPerBlock;
  }

  if (inpType == dtype::f64) {
    err = cudaLaunchKernelEx(
        &config, variance<double>, out, (double *)inp, nel, correction
    );
  } else if (inpType == dtype::f32) {
    err = cudaLaunchKernelEx(
        &config, variance<float>, out, (float *)inp, nel, correction
    );
  } else if (inpType == dtype::i64) {
    err = cudaLaunchKernelEx(
        &config, variance<int64_t>, out, (int64_t *)inp, nel, correction
    );
  } else if (inpType == dtype::i32) {
    err = cudaLaunchKernelEx(
        &config, variance<int32_t>, out, (int32_t *)inp, nel, correction
    );
  } else if (inpType == dtype::i16) {
    err = cudaLaunchKernelEx(
        &config, variance<int16_t>, out, (int16_t *)inp, nel, correction
    );
  } else if (inpType == dtype::i8) {
    err = cudaLaunchKernelEx(
        &config, variance<int8_t>, out, (int8_t *)inp, nel, correction
    );
  } else if (inpType == dtype::u64) {
    err = cudaLaunchKernelEx(
        &config, variance<uint64_t>, out, (uint64_t *)inp, nel, correction
    );
  } else if (inpType == dtype::u32) {
    err = cudaLaunchKernelEx(
        &config, variance<uint32_t>, out, (uint32_t *)inp, nel, correction
    );
  } else if (inpType == dtype::u16) {
    err = cudaLaunchKernelEx(
        &config, variance<uint16_t>, out, (uint16_t *)inp, nel, correction
    );
  } else if (inpType == dtype::u8) {
    err = cudaLaunchKernelEx(
        &config, variance<uint8_t>, out, (uint8_t *)inp, nel, correction
    );
  } else {
    return "Unsupported dtype";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}