#include <cuda_runtime.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <reducers.hpp>
#include <string>

/*
template <typename O, typename I>
__global__ void moments2d(
    O *means, O* variances, I *inp, uint64_t numCols, uint64_t correction, uint8_t calcStd
) {
  uint32_t numThreads = blockDim.x;
  uint32_t row = blockIdx.x;

  inp += row * numCols;

  Variance<double> record{};
  for (uint64_t col = threadIdx.x; col < numCols; col += numThreads) {
    record.consume(inp[col]);
  }
  __syncthreads();

  // Do warp reduction
  for (int indexer = warpSize / 2; indexer > 0; indexer /= 2) {
    record.merge(record.shfl_down(indexer));
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
    for (int indexer = warpSize / 2; indexer > 0; indexer /= 2) {
      record.merge(record.shfl_down(indexer));
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    means[row] = record.mean;
    variances[row] = record.m2 / (numCols - correction);
  }
}

const char *tcuMoments2d(
    tcuStream &stream, void *out, void *inp, Dim2 inpS,
    uint64_t correction, uint8_t calcStd, dtype outType, dtype inpType
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if (inpS.c < 1024) {
    config.blockDim.x = inpS.c;
  } else {
    config.blockDim.x = 1024;
  }
  config.gridDim.x = inpS.r;

  if (outType == dtype::f64) {
    if (inpType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, double>, (double *)out, (double *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, float>, (double *)out, (float *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, int64_t>, (double *)out, (int64_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, int32_t>, (double *)out, (int32_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, int16_t>, (double *)out, (int16_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, int8_t>, (double *)out, (int8_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, uint64_t>, (double *)out, (uint64_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, uint32_t>, (double *)out, (uint32_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, uint16_t>, (double *)out, (uint16_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, moments2d<double, uint8_t>, (double *)out, (uint8_t *)inp,
          inpS.c, correction, calcStd
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inpType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, double>, (float *)out, (double *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, float>, (float *)out, (float *)inp, inpS.c,
          correction, calcStd
      );
    } else if (inpType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, int64_t>, (float *)out, (int64_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, int32_t>, (float *)out, (int32_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, int16_t>, (float *)out, (int16_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, int8_t>, (float *)out, (int8_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, uint64_t>, (float *)out, (uint64_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, uint32_t>, (float *)out, (uint32_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, uint16_t>, (float *)out, (uint16_t *)inp,
          inpS.c, correction, calcStd
      );
    } else if (inpType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, moments2d<float, uint8_t>, (float *)out, (uint8_t *)inp,
          inpS.c, correction, calcStd
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f16) {
    return "Unsupported output type";
  } else {
    return "Unsupported output type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}
*/