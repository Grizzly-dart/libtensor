#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-dcl21-cpp"

#ifndef TENSORCPU_TYPED_ARRAY_HPP
#define TENSORCPU_TYPED_ARRAY_HPP

#include <experimental/simd>
#include <iterator>
#include <memory>
#include <stdfloat>

#include "range.hpp"
#include "tensorcpu.hpp"

namespace stdx = std::experimental;

template <typename T> constexpr size_t simdSize() {
  return std::experimental::native_simd<T>::size();
}

template <typename T> constexpr bool isAnyInt() {
  return std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
         std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
         std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
         std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value;
}

template <typename T> constexpr bool isInt() {
  return std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
         std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value;
}

template <typename T> constexpr bool isUInt() {
  return std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
         std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value;
}

using Kernel = std::function<void(uint64_t)>;

struct DType {
public:
  uint8_t index;
  uint8_t bytes;
  uint8_t subIndex;

  void *offsetPtr(void *ptr, uint64_t offset) const {
    return (uint8_t *)ptr + offset * bytes;
  }
};

constexpr uint8_t i8Id = 0;
constexpr uint8_t i16Id = 1;
constexpr uint8_t i32Id = 2;
constexpr uint8_t i64Id = 3;
constexpr uint8_t u8Id = 4;
constexpr uint8_t u16Id = 5;
constexpr uint8_t u32Id = 6;
constexpr uint8_t u64Id = 7;
constexpr uint8_t f32Id = 8;
constexpr uint8_t f64Id = 9;

const DType i8 = {i8Id, 1, 0};
const DType i16 = {i16Id, 2, 1};
const DType i32 = {i32Id, 4, 2};
const DType i64 = {i64Id, 8, 3};
const DType u8 = {u8Id, 1, 4};
const DType u16 = {u16Id, 2, 5};
const DType u32 = {u32Id, 4, 6};
const DType u64 = {u64Id, 8, 7};
const DType f32 = {f32Id, 4, 2};
const DType f64 = {f64Id, 4, 3};
/*const DType bf16 = {10, 2, 0};
const DType f16 = {11, 2, 1};*/

const DType dtypes[] = {
    i8, i16, i32, i64, u8, u16, u32, u64, f32, f64,
};

template <typename T> constexpr DType dtypeOf() {
  if constexpr (std::is_same<T, int8_t>::value) {
    return i8;
  } else if constexpr (std::is_same<T, int16_t>::value) {
    return i16;
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return i32;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return i64;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    return u8;
  } else if constexpr (std::is_same<T, uint16_t>::value) {
    return u16;
  } else if constexpr (std::is_same<T, uint32_t>::value) {
    return u32;
  } else if constexpr (std::is_same<T, uint64_t>::value) {
    return u64;
  } else if constexpr (std::is_same<T, float>::value) {
    return f32;
  } else if constexpr (std::is_same<T, double>::value) {
    return f64;
  } else {
    throw std::invalid_argument("Invalid type");
  }
}

template <typename C, uint16_t laneSize> class Caster {
public:
  typedef C SimdType __attribute__((vector_size(sizeof(C) * laneSize)));
  const DType &dtype;
  Caster(const DType &dtype) : dtype(dtype){};

  template <typename T> void load(T &out, const void *inp, uint64_t offset) const {
    out = (T) * (C *)(dtype.offsetPtr((void *)inp, offset));
  }

  template <typename T> void store(void *out, T &inp, uint64_t offset) const {
    *(C *)dtype.offsetPtr(out, offset) = (C)inp;
  }

  template <typename T>
  void simdLoad(
      T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
      const void *inp, uint64_t offset
  ) const {
    typedef T TSimdType __attribute__((vector_size(sizeof(T) * laneSize)));
    SimdType tmp;
    memcpy(&tmp, dtype.offsetPtr((void *)inp, offset), sizeof(SimdType));
    out = __builtin_convertvector(tmp, TSimdType);
  }

  template <typename T>
  void simdStore(
      void *out, T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
      uint64_t offset
  ) const {
    SimdType tmp = __builtin_convertvector(inp, SimdType);
    memcpy(dtype.offsetPtr(out, offset), &tmp, sizeof(SimdType));
  }
};

/*
template <typename T>
using CastLoader = void (*)(T &out, const void *inp, uint64_t offset);

template <typename T>
using CastStorer = void (*)(void *inp, T &out, uint64_t offset);

template <typename T, uint16_t laneSize>
using CastSimdLoader = void (*)(
    T __attribute__((vector_size(sizeof(T) * laneSize))) & out, const void *inp,
    uint64_t offset
);

template <typename T, uint16_t laneSize>
using CastSimdStorer = void (*)(
    void *out, T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
    uint64_t offset
);

template <typename T> extern CastLoader<T> castedLoader(const DType &type);

template <typename T> extern CastStorer<T> castedStorer(const DType &type);

template <typename T, uint16_t laneSize>
extern CastSimdLoader<T, laneSize> castedVectorStore(const DType &type);

template <typename T, uint16_t laneSize>
extern CastSimdStorer<T, laneSize> castedVectorStore(const DType &type);
 */

#endif // TENSORCPU_TYPED_ARRAY_HPP

#pragma clang diagnostic pop