#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include <tensorcuda.hpp>

#include "caster.hpp"



template <typename T> struct [[maybe_unused]] Caster1 {
  void *ptr;

  CastLoader<T> loader = nullptr;
  CastStorer<T> storer = nullptr;
  CastOffsetter indexer = nullptr;

  __device__ __host__ explicit Caster1(void *ptr, dtype type) : ptr(ptr) {
    init(type);
  }

  __device__ __host__ T operator[](uint64_t index) {
    return loader(ptr, index);
  }

  __device__ __host__ void store(uint64_t index, T value) {
    storer(ptr, index, value);
  }

  __device__ __host__ void init(dtype inpType) {
    switch (inpType) {
    case i8:
      loader = castLoader<int64_t, int8_t>;
      storer = castStorer<int64_t, int8_t>;
      indexer = castIndexer<int8_t>;
      return;
    case i16:
      loader = castLoader<int64_t, int16_t>;
      storer = castStorer<int64_t, int16_t>;
      indexer = castIndexer<int8_t>;
      return;
    case i32:
      loader = castLoader<int64_t, int32_t>;
      storer = castStorer<int64_t, int32_t>;
      indexer = castIndexer<int8_t>;
      return;
    case i64:
      loader = castLoader<int64_t, int64_t>;
      storer = castStorer<int64_t, int64_t>;
      indexer = castIndexer<int8_t>;
      return;
    case u8:
      loader = castLoader<int64_t, uint8_t>;
      storer = castStorer<int64_t, uint8_t>;
      indexer = castIndexer<int8_t>;
      return;
    case u16:
      loader = castLoader<int64_t, uint16_t>;
      storer = castStorer<int64_t, uint16_t>;
      indexer = castIndexer<int8_t>;
      return;
    case u32:
      loader = castLoader<int64_t, uint32_t>;
      storer = castStorer<int64_t, uint32_t>;
      indexer = castIndexer<int8_t>;
      return;
    case u64:
      loader = castLoader<int64_t, uint64_t>;
      storer = castStorer<int64_t, uint64_t>;
      indexer = castIndexer<int8_t>;
      return;
    default:
      return;
    }
  }
};

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
template <typename O, typename I1, typename I2>
__global__ void plusV3(
    Caster1<O> &out, Caster1<I1> &inp1, Caster1<I2> &inp2, uint64_t n
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out.store(i, inp1[i] + inp2[i]);
  }
}

template <typename O, typename I1, typename I2>
void tcplusV3(
    void *out, void *inp1, void *inp2, uint64_t nel, dtype outType,
    dtype inp1Type, dtype inp2Type
) {
  Caster1<O> o(out, outType);
  Caster1<I1> i1(inp1, inp1Type);
  Caster1<I2> i2(inp2, inp2Type);
  plusV3<<<4, 5>>>(o, i1, i2, nel);
}