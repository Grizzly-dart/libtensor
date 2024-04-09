#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include <tensorcuda.hpp>

template <typename O, typename I>
__device__ __host__ O castLoader(void *ptr, uint64_t index) {
  return ((I *)ptr)[index];
}

template <typename O, typename I>
__device__ __host__ void castStorer(void *ptr, uint64_t index, O value) {
  ((I *)ptr)[index] = value;
}

template <typename O> using loader = O (*)(void *, uint64_t);
template <typename O> using storer = void (*)(void *, uint64_t, O);

template <typename T>
__device__ __host__ void intCaster(dtype inpType, loader<T> **loader, storer<T> **storer) {
  switch (inpType) {
  case i8:
    *loader = castLoader<int64_t, int8_t>;
    *storer = castStorer<int64_t, int8_t>;
    return;
  case i16:
    *loader = castLoader<int64_t, int16_t>;
    *storer = castStorer<int64_t, int16_t>;
    return;
  case i32:
    *loader = castLoader<int64_t, int32_t>;
    *storer = castStorer<int64_t, int32_t>;
    return;
  case i64:
    *loader = castLoader<int64_t, int64_t>;
    *storer = castStorer<int64_t, int64_t>;
    return;
  case u8:
    *loader = castLoader<int64_t, uint8_t>;
    *storer = castStorer<int64_t, uint8_t>;
    return;
  case u16:
    *loader = castLoader<int64_t, uint16_t>;
    *storer = castStorer<int64_t, uint16_t>;
    return;
  case u32:
    *loader = castLoader<int64_t, uint32_t>;
    *storer = castStorer<int64_t, uint32_t>;
    return;
  case u64:
    *loader = castLoader<int64_t, uint64_t>;
    *storer = castStorer<int64_t, uint64_t>;
    return;
  default:
    return;
  }
}

/// Adds two tensors
template <typename O, typename I1, typename I2>
__global__ void plusV2(
    O *out, I1 *inp1, I2 *inp2, dtype outType, dtype inp1Type, dtype inp2Type,
    uint64_t n, uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  auto i1 = getIntInterpreter(inp1Type, inp1);
  auto i2 = getIntInterpreter(inp2Type, inp2);
  auto o = getIntInterpreter(outType, out);

  for (uint64_t i = thId; i < n; i += numThreads) {
    o->store(i1->load(i) + i2->load(i));
  }
}

/// Adds two tensors
__global__ void plusV3(
    void *out, IntInterpreter *inp1, IntInterpreter *inp2, uint64_t n,
    uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out->store(i, inp1->load(i) + inp2->load(i));
  }
}

void tcplusV3(void *out, void *inp1, void *inp2) {
  auto o = getIntInterpreter(i64, out);
  auto i1 = getIntInterpreter(i64, inp1);
  auto i2 = getIntInterpreter(i64, inp2);
  plusV3<<<4, 5>>>(o, i1, i2, 10, 0);
}