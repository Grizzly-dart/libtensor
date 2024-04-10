#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <tensorcuda.hpp>

template <typename O, typename I>
__global__ void sinKernel(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::sin(inp[i]);
  }
}

template <typename O, typename I>
__global__ void cosKernel(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::cos(inp[i]);
  }
}

template <typename O, typename I>
__global__ void tanKernel(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::tan(inp[i]);
  }
}

template <typename O, typename I>
__global__ void sinhKernel(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::sinh(inp[i]);
  }
}

template <typename O, typename I>
__global__ void coshKernel(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::cosh(inp[i]);
  }
}

template <typename O, typename I>
__global__ void tanhKernel(O *out, I *inp, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = std::tanh(inp[i]);
  }
}

/*
extern const char *tcuSin(
    tcuStream &stream, void *out, const void *inp, uint64_t n,
    dtype outType, dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (outType == dtype::f64) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, double>, (double *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, float>, (double *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, uint64_t>, (double *)out, (uint64_t *)inp,
          n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, uint32_t>, (double *)out, (uint32_t *)inp,
          n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, uint16_t>, (double *)out, (uint16_t *)inp,
          n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, double>, (float *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, float>, (float *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, sinKernel<float, uint8_t>, (float *)out, (uint8_t *)inp, n
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

extern const char *tcuCos(
    tcuStream &stream, void *out, const void *inp, uint64_t n,
    dtype outType, dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (outType == dtype::f64) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, double>, (double *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, float>, (double *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, uint64_t>, (double *)out, (uint64_t *)inp,
          n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, uint32_t>, (double *)out, (uint32_t *)inp,
          n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, uint16_t>, (double *)out, (uint16_t *)inp,
          n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, double>, (float *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, float>, (float *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, cosKernel<float, uint8_t>, (float *)out, (uint8_t *)inp, n
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

extern const char *tcuTan(
    tcuStream &stream, void *out, const void *inp, uint64_t n,
    dtype outType, dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (outType == dtype::f64) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, double>, (double *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, float>, (double *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, uint64_t>, (double *)out, (uint64_t *)inp,
          n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, uint32_t>, (double *)out, (uint32_t *)inp,
          n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, uint16_t>, (double *)out, (uint16_t *)inp,
          n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, double>, (float *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, float>, (float *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, tanKernel<float, uint8_t>, (float *)out, (uint8_t *)inp, n
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

extern const char *tcuSinh(
    tcuStream &stream, void *out, const void *inp, uint64_t n,
    dtype outType, dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (outType == dtype::f64) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, double>, (double *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, float>, (double *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, uint64_t>, (double *)out, (uint64_t *)inp,
          n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, uint32_t>, (double *)out, (uint32_t *)inp,
          n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, uint16_t>, (double *)out, (uint16_t *)inp,
          n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, double>, (float *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, float>, (float *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, sinhKernel<float, uint8_t>, (float *)out, (uint8_t *)inp, n
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

extern const char *tcuCosh(
    tcuStream &stream, void *out, const void *inp, uint64_t n,
    dtype outType, dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (outType == dtype::f64) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, double>, (double *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, float>, (double *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, uint64_t>, (double *)out, (uint64_t *)inp,
          n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, uint32_t>, (double *)out, (uint32_t *)inp,
          n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, uint16_t>, (double *)out, (uint16_t *)inp,
          n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, double>, (float *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, float>, (float *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, coshKernel<float, uint8_t>, (float *)out, (uint8_t *)inp, n
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

extern const char *tcuTanh(
    tcuStream &stream, void *out, const void *inp, uint64_t n,
    dtype outType, dtype inType
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (outType == dtype::f64) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, double>, (double *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, float>, (double *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, int64_t>, (double *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, int32_t>, (double *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, int16_t>, (double *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, int8_t>, (double *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, uint64_t>, (double *)out, (uint64_t *)inp,
          n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, uint32_t>, (double *)out, (uint32_t *)inp,
          n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, uint16_t>, (double *)out, (uint16_t *)inp,
          n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<double, uint8_t>, (double *)out, (uint8_t *)inp, n
      );
    } else {
      return "Unsupported input type";
    }
  } else if (outType == dtype::f32) {
    if (inType == dtype::f64) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, double>, (float *)out, (double *)inp, n
      );
    } else if (inType == dtype::f32) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, float>, (float *)out, (float *)inp, n
      );
    } else if (inType == dtype::i64) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, int64_t>, (float *)out, (int64_t *)inp, n
      );
    } else if (inType == dtype::i32) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, int32_t>, (float *)out, (int32_t *)inp, n
      );
    } else if (inType == dtype::i16) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, int16_t>, (float *)out, (int16_t *)inp, n
      );
    } else if (inType == dtype::i8) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, int8_t>, (float *)out, (int8_t *)inp, n
      );
    } else if (inType == dtype::u64) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, uint64_t>, (float *)out, (uint64_t *)inp, n
      );
    } else if (inType == dtype::u32) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, uint32_t>, (float *)out, (uint32_t *)inp, n
      );
    } else if (inType == dtype::u16) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, uint16_t>, (float *)out, (uint16_t *)inp, n
      );
    } else if (inType == dtype::u8) {
      err = cudaLaunchKernelEx(
          &config, tanhKernel<float, uint8_t>, (float *)out, (uint8_t *)inp, n
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
 */