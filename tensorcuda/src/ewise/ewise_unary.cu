#include <cmath>
#include <cstdint>
#include <string>
#include <limits>

#include <cuda_runtime.h>

#include <tensorcuda.hpp>

template <typename O, typename I>
__global__ void cast(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp[i];
  }
}

template <typename O, typename I>
__global__ void neg(O *out, I *inp, uint64_t nel, I minVal, I maxVal) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < nel; i += numThreads) {
    if (inp[i] == minVal) {
      out[i] = maxVal;
    } else {
      out[i] = -inp[i];
    }
  }
}

template <typename T, typename O>
__global__ void abs(T *out, O *inp, uint64_t nel) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < nel; i += numThreads) {
    if (inp[i] == std::numeric_limits<T>::min()) {
      out[i] = std::numeric_limits<T>::max();
    } else {
      out[i] = inp[i] >= 0 ? inp[i] : -inp[i];
    }
  }
}

template <typename O, typename I>
__global__ void sqr(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp[i] * inp[i];
  }
}

template <typename O, typename I>
__global__ void sqrt(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::sqrt(inp[i]);
  }
}

template <typename O, typename I>
__global__ void log(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::log(inp[i]);
  }
}

template <typename O, typename I>
__global__ void exp(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::exp(inp[i]);
  }
}

/*
const char *tcuNeg(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (inType == dtype::f64) {
    if (outType != dtype::f64)
      return "Expected f64 output";
    err = cudaLaunchKernelEx(
        &config, neg<double, double>, (double *)out, (double *)inp, n,
        __DBL_MIN__, __DBL_MAX__
    );
  } else if (inType == dtype::f32) {
    if (outType != dtype::f32)
      return "Expected f32 output";
    err = cudaLaunchKernelEx(
        &config, neg<float, float>, (float *)out, (float *)inp, n, __FLT_MIN__,
        __FLT_MAX__
    );
  } else if (inType == dtype::i64) {
    if (outType != dtype::i64)
      return "Expected i64 output";
    err = cudaLaunchKernelEx(
        &config, neg<int64_t, int64_t>, (int64_t *)out, (int64_t *)inp, n,
        INT64_MIN, INT64_MAX
    );
  } else if (inType == dtype::u64) {
    if (outType != dtype::i64)
      return "Expected i64 output";
    err = cudaLaunchKernelEx(
        &config, neg<int64_t, uint64_t>, (int64_t *)out, (uint64_t *)inp, n,
        INT64_MIN, INT64_MAX
    );
  } else if (inType == dtype::i32) {
    if (outType != dtype::i32)
      return "Expected i32 output";
    err = cudaLaunchKernelEx(
        &config, neg<int32_t, int32_t>, (int32_t *)out, (int32_t *)inp, n,
        INT32_MIN, INT32_MAX
    );
  } else if (inType == dtype::u32) {
    if (outType != dtype::i32)
      return "Expected i32 output";
    err = cudaLaunchKernelEx(
        &config, neg<int32_t, uint32_t>, (int32_t *)out, (uint32_t *)inp, n,
        INT32_MIN, INT32_MAX
    );
  } else if (inType == dtype::i16) {
    if (outType != dtype::i16)
      return "Expected i16 output";
    err = cudaLaunchKernelEx(
        &config, neg<int16_t, int16_t>, (int16_t *)out, (int16_t *)inp, n,
        INT16_MIN, INT16_MAX
    );
  } else if (inType == dtype::u16) {
    if (outType != dtype::i16)
      return "Expected i16 output";
    err = cudaLaunchKernelEx(
        &config, neg<int16_t, uint16_t>, (int16_t *)out, (uint16_t *)inp, n,
        INT16_MIN, INT16_MAX
    );
  } else if (inType == dtype::i8) {
    if (outType != dtype::i8)
      return "Expected i8 output";
    err = cudaLaunchKernelEx(
        &config, neg<int8_t, int8_t>, (int8_t *)out, (int8_t *)inp, n, INT8_MIN,
        INT8_MAX
    );
  } else if (inType == dtype::u8) {
    if (outType != dtype::i8)
      return "Expected i8 output";
    err = cudaLaunchKernelEx(
        &config, neg<int8_t, uint8_t>, (int8_t *)out, (uint8_t *)inp, n,
        INT8_MIN, INT8_MAX
    );
  } else {
    return "Unsupported input type";
  }

  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char *tcuAbs(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (inType == dtype::f64) {
    if (outType != dtype::f64)
      return "Expected f64 output";
    err = cudaLaunchKernelEx(
        &config, abs<double, double>, (double *)out, (double *)inp, n
    );
  } else if (inType == dtype::f32) {
    if (outType != dtype::f32)
      return "Expected f32 output";
    err = cudaLaunchKernelEx(
        &config, abs<float, float>, (float *)out, (float *)inp, n
    );
  } else if (inType == dtype::i64) {
    if (outType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, abs<int64_t, int64_t>, (int64_t *)out, (int64_t *)inp, n
      );
    } else if (outType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, abs<uint64_t, int64_t>, (uint64_t *)out, (int64_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (inType == dtype::u64) {
    if (outType != dtype::u64)
      return "Expected u64 output";
    if (out == inp)
      return nullptr;
    err = cudaMemcpyAsync(
        out, inp, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream.stream
    );
  } else if (inType == dtype::i32) {
    if (outType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, abs<int32_t, int32_t>, (int32_t *)out, (int32_t *)inp, n
      );
    } else if (outType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, abs<uint32_t, int32_t>, (uint32_t *)out, (int32_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (inType == dtype::u32) {
    if (outType != dtype::u32)
      return "Expected u32 output";
    if (out == inp)
      return nullptr;
    err = cudaMemcpyAsync(
        out, inp, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream.stream
    );
  } else if (inType == dtype::i16) {
    if (outType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, abs<int16_t, int16_t>, (int16_t *)out, (int16_t *)inp, n
      );
    } else if (outType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, abs<uint16_t, int16_t>, (uint16_t *)out, (int16_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (inType == dtype::u16) {
    if (outType != dtype::u16)
      return "Expected u16 output";
    if (out == inp)
      return nullptr;
    err = cudaMemcpyAsync(
        out, inp, n * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream.stream
    );
  } else if (inType == dtype::i8) {
    if (outType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, abs<int8_t, int8_t>, (int8_t *)out, (int8_t *)inp, n
      );
    } else if (outType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, abs<uint8_t, int8_t>, (uint8_t *)out, (int8_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (inType == dtype::u8) {
    if (outType != dtype::u8)
      return "Expected u8 output";
    if (out == inp)
      return nullptr;
    err = cudaMemcpyAsync(
        out, inp, n * sizeof(uint8_t), cudaMemcpyDeviceToDevice, stream.stream
    );
  } else {
    return "Unsupported input type";
  }

  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char *tcuSqr(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (inType == dtype::f64) {
    if (outType != dtype::f64)
      return "Expected f64 output";
    err = cudaLaunchKernelEx(
        &config, sqr<double, double>, (double *)out, (double *)inp, n
    );
  } else if (inType == dtype::f32) {
    if (outType == dtype::f32)
      err = cudaLaunchKernelEx(
          &config, sqr<float, float>, (float *)out, (float *)inp, n
      );
    else if (outType == dtype::f64)
      err = cudaLaunchKernelEx(
          &config, sqr<double, float>, (double *)out, (float *)inp, n
      );
    else
      return "Unsupported output type";
  } else if (inType == dtype::i64) {
    if (outType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, sqr<int64_t, int64_t>, (int64_t *)out, (int64_t *)inp, n
      );
    else if (outType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqr<uint64_t, int64_t>, (uint64_t *)out, (int64_t *)inp, n
      );
    else
      return "Unsupported output type";
  } else if (inType == dtype::u64) {
    if (outType != dtype::u64)
      return "Expected u64 output";
    err = cudaLaunchKernelEx(
        &config, sqr<uint64_t, uint64_t>, (uint64_t *)out, (uint64_t *)inp, n
    );
  } else if (inType == dtype::i32) {
    if (outType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, sqr<int64_t, int32_t>, (int64_t *)out, (int32_t *)inp, n
      );
    else if (outType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqr<uint64_t, int32_t>, (uint64_t *)out, (int32_t *)inp, n
      );
    else if (outType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, sqr<int32_t, int32_t>, (int32_t *)out, (int32_t *)inp, n
      );
    else if (outType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqr<uint32_t, int32_t>, (uint32_t *)out, (int32_t *)inp, n
      );
    else
      return "Unsupported output type";
  } else if (inType == dtype::u32) {
    if(outType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqr<uint32_t, uint32_t>, (uint32_t *)out, (uint32_t *)inp, n
      );
    else if(outType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqr<uint64_t, uint32_t>, (uint64_t *)out, (uint32_t *)inp, n
      );
    else
      return "Unsupported output type";
  } else if(inType == dtype::i16) {
    if(outType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, sqr<int64_t, int16_t>, (int64_t *)out, (int16_t *)inp, n
      );
    else if(outType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqr<uint64_t, int16_t>, (uint64_t *)out, (int16_t *)inp, n
      );
    else if(outType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, sqr<int32_t, int16_t>, (int32_t *)out, (int16_t *)inp, n
      );
    else if(outType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqr<uint32_t, int16_t>, (uint32_t *)out, (int16_t *)inp, n
      );
    else if(outType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, sqr<int16_t, int16_t>, (int16_t *)out, (int16_t *)inp, n
      );
    else if(outType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, sqr<uint16_t, int16_t>, (uint16_t *)out, (int16_t *)inp, n
      );
    else
      return "Unsupported output type";
  } else if(inType == dtype::u16) {
    if(outType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, sqr<uint16_t, uint16_t>, (uint16_t *)out, (uint16_t *)inp, n
      );
    else if(outType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqr<uint64_t, uint16_t>, (uint64_t *)out, (uint16_t *)inp, n
      );
    else if(outType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqr<uint32_t, uint16_t>, (uint32_t *)out, (uint16_t *)inp, n
      );
    else
      return "Unsupported output type";
  } else if(inType == dtype::i8) {
    if(outType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, sqr<int64_t, int8_t>, (int64_t *)out, (int8_t *)inp, n
      );
    else if(outType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqr<uint64_t, int8_t>, (uint64_t *)out, (int8_t *)inp, n
      );
    else if(outType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, sqr<int32_t, int8_t>, (int32_t *)out, (int8_t *)inp, n
      );
    else if(outType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqr<uint32_t, int8_t>, (uint32_t *)out, (int8_t *)inp, n
      );
    else if(outType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, sqr<int16_t, int8_t>, (int16_t *)out, (int8_t *)inp, n
      );
    else if(outType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, sqr<uint16_t, int8_t>, (uint16_t *)out, (int8_t *)inp, n
      );
    else if(outType == dtype::i8)
      err = cudaLaunchKernelEx(
          &config, sqr<int8_t, int8_t>, (int8_t *)out, (int8_t *)inp, n
      );
    else if(outType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, sqr<uint8_t, int8_t>, (uint8_t *)out, (int8_t *)inp, n
      );
    else
      return "Unsupported output type";
  } else if(inType == dtype::u8) {
    if(outType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, sqr<uint8_t, uint8_t>, (uint8_t *)out, (uint8_t *)inp, n
      );
    else if(outType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqr<uint64_t, uint8_t>, (uint64_t *)out, (uint8_t *)inp, n
      );
    else if(outType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqr<uint32_t, uint8_t>, (uint32_t *)out, (uint8_t *)inp, n
      );
    else if(outType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, sqr<uint16_t, uint8_t>, (uint16_t *)out, (uint8_t *)inp, n
      );
    else
      return "Unsupported output type";
  } else {
    return "Unsupported input type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char *tcuSqrt(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if(outType == dtype::f64) {
    if(inType == dtype::f64)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, double>, (double *)out, (double *)inp, n
      );
    else if(inType == dtype::f32)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, float>, (double *)out, (float *)inp, n
      );
    else if(inType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    else if(inType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, uint64_t>, (double *)out, (uint64_t *)inp, n
      );
    else if(inType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    else if(inType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, uint32_t>, (double *)out, (uint32_t *)inp, n
      );
    else if(inType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    else if(inType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, uint16_t>, (double *)out, (uint16_t *)inp, n
      );
    else if(inType == dtype::i8)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    else if(inType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, sqrt<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    else
      return "Unsupported input type";
  } else if(outType == dtype::f32) {
    if(inType == dtype::f64)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, double>, (float *)out, (double *)inp, n
      );
    else if(inType == dtype::f32)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, float>, (float *)out, (float *)inp, n
      );
    else if(inType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    else if(inType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    else if(inType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    else if(inType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    else if(inType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    else if(inType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    else if(inType == dtype::i8)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    else if(inType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, sqrt<float, uint8_t>, (float *)out, (uint8_t *)inp, n
      );
    else
      return "Unsupported input type";
  } else {
    return "Unsupported output type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char *tcuLog(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }
  cudaError_t err;
  if(outType == dtype::f64) {
    if(inType == dtype::f64)
      err = cudaLaunchKernelEx(
          &config, log<double, double>, (double *)out, (double *)inp, n
      );
    else if(inType == dtype::f32)
      err = cudaLaunchKernelEx(
          &config, log<double, float>, (double *)out, (float *)inp, n
      );
    else if(inType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, log<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    else if(inType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, log<double, uint64_t>, (double *)out, (uint64_t *)inp, n
      );
    else if(inType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, log<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    else if(inType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, log<double, uint32_t>, (double *)out, (uint32_t *)inp, n
      );
    else if(inType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, log<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    else if(inType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, log<double, uint16_t>, (double *)out, (uint16_t *)inp, n
      );
    else if(inType == dtype::i8)
      err = cudaLaunchKernelEx(
          &config, log<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    else if(inType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, log<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    else
      return "Unsupported input type";
  } else if(outType == dtype::f32) {
    if(inType == dtype::f64)
      err = cudaLaunchKernelEx(
          &config, log<float, double>, (float *)out, (double *)inp, n
      );
    else if(inType == dtype::f32)
      err = cudaLaunchKernelEx(
          &config, log<float, float>, (float *)out, (float *)inp, n
      );
    else if(inType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, log<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    else if(inType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, log<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    else if(inType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, log<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    else if(inType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, log<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    else if(inType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, log<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    else if(inType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, log<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    else if(inType == dtype::i8)
      err = cudaLaunchKernelEx(
          &config, log<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    else if(inType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, log<float, uint8_t>, (float *)out, (uint8_t *)inp, n
      );
    else
      return "Unsupported input type";
  } else {
    return "Unsupported output type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char *tcuExp(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }
  cudaError_t err;
  if(outType == dtype::f64) {
    if(inType == dtype::f64)
      err = cudaLaunchKernelEx(
          &config, exp<double, double>, (double *)out, (double *)inp, n
      );
    else if(inType == dtype::f32)
      err = cudaLaunchKernelEx(
          &config, exp<double, float>, (double *)out, (float *)inp, n
      );
    else if(inType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, exp<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    else if(inType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, exp<double, uint64_t>, (double *)out, (uint64_t *)inp, n
      );
    else if(inType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, exp<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    else if(inType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, exp<double, uint32_t>, (double *)out, (uint32_t *)inp, n
      );
    else if(inType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, exp<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    else if(inType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, exp<double, uint16_t>, (double *)out, (uint16_t *)inp, n
      );
    else if(inType == dtype::i8)
      err = cudaLaunchKernelEx(
          &config, exp<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    else if(inType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, exp<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    else
      return "Unsupported input type";
  } else if(outType == dtype::f32) {
    if(inType == dtype::f64)
      err = cudaLaunchKernelEx(
          &config, exp<float, double>, (float *)out, (double *)inp, n
      );
    else if(inType == dtype::f32)
      err = cudaLaunchKernelEx(
          &config, exp<float, float>, (float *)out, (float *)inp, n
      );
    else if(inType == dtype::i64)
      err = cudaLaunchKernelEx(
          &config, exp<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    else if(inType == dtype::u64)
      err = cudaLaunchKernelEx(
          &config, exp<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    else if(inType == dtype::i32)
      err = cudaLaunchKernelEx(
          &config, exp<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    else if(inType == dtype::u32)
      err = cudaLaunchKernelEx(
          &config, exp<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    else if(inType == dtype::i16)
      err = cudaLaunchKernelEx(
          &config, exp<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    else if(inType == dtype::u16)
      err = cudaLaunchKernelEx(
          &config, exp<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    else if(inType == dtype::i8)
      err = cudaLaunchKernelEx(
          &config, exp<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    else if(inType == dtype::u8)
      err = cudaLaunchKernelEx(
          &config, exp<float, uint8_t>, (float *)out, (uint8_t *)inp, n
      );
    else
      return "Unsupported input type";
  }

  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}
 */

// #include "ewise_unary_gen.inc"