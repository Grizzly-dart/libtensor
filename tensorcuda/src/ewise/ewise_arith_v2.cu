#include <cuda_runtime.h>

#include <tensorcuda.hpp>

#include "caster.hpp"

template <typename O, typename I1, typename I2, bool isRwise>
__global__ void plus(
    O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n, Dim2 rwise
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    if (inp2 != nullptr) {
      if constexpr (!isRwise) {
        scalar = inp2[i];
      } else {
        scalar = inp2[i / rwise.c % rwise.r];
      }
    }
    out[i] = inp1[i] + scalar;
  }
}

template <typename O, typename I1, typename I2, bool isRwise>
__global__ void plusSlow(
    void *out, void *inp1, void *inp2, I2 scalar, uint64_t n, Dim2 rwise,
    uint8_t outType, uint8_t inp1Type, uint8_t inp2Type
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  Caster<O> o;
  if constexpr (std::is_same<O, int64_t>::value) {
    o = i64Casters[outType];
  } else if constexpr (std::is_same<O, double>::value) {
    o = f64Casters[outType];
  }
  Caster<I1> i1;
  if constexpr (std::is_same<I1, int64_t>::value) {
    i1 = i64Casters[inp1Type];
  } else if constexpr (std::is_same<I1, double>::value) {
    i1 = f64Casters[inp1Type];
  }
  Caster<I2> i2;
  if constexpr (std::is_same<I2, int64_t>::value) {
    i2 = i64Casters[inp2Type];
  } else if constexpr (std::is_same<I2, double>::value) {
    i2 = f64Casters[inp2Type];
  }

  for (uint64_t i = thId; i < n; i += numThreads) {
    if (inp2 != nullptr) {
      if constexpr (!isRwise) {
        scalar = i2.loader(inp2, i);
      } else {
        scalar = i2.loader(inp2, i / rwise.c % rwise.r);
      }
    }

    o.storer(out, i, i1.loader(inp1, i) + scalar);
  }
}

template <typename O, typename I1, typename I2>
const char *tcuPlus(
    tcuStream &stream, O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n,
    Dim2 rowWise, uint8_t flip, uint8_t outType, uint8_t inp1Type,
    uint8_t inp2Type
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;

  bool isRwise = rowWise.nel() != 0;
  if (!isRwise) {
    err = cudaLaunchKernelEx(
        &config, plusSlow<O, I1, I2, false>, (void *)out, (void *)inp1,
        (void *)inp2, scalar, n, rowWise, outType, inp1Type, inp2Type
    );
  } else {
    err = cudaLaunchKernelEx(
        &config, plusSlow<O, I1, I2, true>, (void *)out, (void *)inp1,
        (void *)inp2, scalar, n, rowWise, outType, inp1Type, inp2Type
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template <typename O, typename I1, typename I2>
const char *tcuPlusSlow(
    tcuStream &stream, O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n,
    Dim2 rowWise, uint8_t flip, uint8_t outType, uint8_t inp1Type,
    uint8_t inp2Type
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;

  bool isRwise = rowWise.nel() != 0;
  if (!isRwise) {
    err = cudaLaunchKernelEx(
        &config, plus<O, I1, I2, false>, out, inp1, inp2, scalar, n, rowWise
    );
  } else {
    err = cudaLaunchKernelEx(
        &config, plus<O, I1, I2, true>, out, inp1, inp2, scalar, n, rowWise
    );
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

#define UNWIND2(A, B, OP, NAME)                                                \
  OP(A, A, A, NAME)                                                            \
  OP(A, A, B, NAME)                                                            \
  OP(A, B, A, NAME)                                                            \
  OP(A, B, B, NAME)                                                            \
  OP(B, B, B, NAME)                                                            \
  OP(B, A, B, NAME)                                                            \
  OP(B, B, A, NAME)                                                            \
  OP(B, A, A, NAME)

#define PLUS(O, I1, I2, NAME)                                                  \
  template const char *tcu##NAME(                                              \
      tcuStream &stream, O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n,    \
      Dim2 rowWise, uint8_t flip, uint8_t outType, uint8_t inp1Type,           \
      uint8_t inp2Type                                                         \
  );

UNWIND2(double, float, PLUS, Plus)

#define PLUS_SLOW(O, I1, I2, NAME)                                             \
  template const char *tcu##NAME##Slow(                                        \
      tcuStream &stream, O *out, I1 *inp1, I2 *inp2, I2 scalar, uint64_t n,    \
      Dim2 rowWise, uint8_t flip, uint8_t outType, uint8_t inp1Type,           \
      uint8_t inp2Type                                                         \
  );

UNWIND2(double, int64_t, PLUS_SLOW, Plus)