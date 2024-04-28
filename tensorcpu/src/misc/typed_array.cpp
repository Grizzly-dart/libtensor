#include <experimental/simd>
#include <iostream>
#include <iterator>
#include <stdfloat>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

template <typename O, typename I> O castLoader(void *ptr, uint64_t index) {
  return static_cast<O>(((I *)ptr)[index]);
}

template <typename O, typename I>
void castStorer(void *ptr, uint64_t index, O value) {
  I tmp = (I)(value);
  ((I *)ptr)[index] = tmp;
}

template <typename I> void *castIndexer(void *src, int64_t index) {
  return ((I *)src) + index;
}

template <typename O, typename I>
void castSimdLoader(
    void *ptr, uint64_t index, stdx::fixed_size_simd<O, simdSize<O>()> &simd
) {
  stdx::fixed_size_simd<I, simdSize<O>()> tmp;
  tmp.copy_from(((I *)ptr) + index, stdx::vector_aligned);
  stdx::fixed_size_simd<O, simdSize<O>()> simd1 =
      stdx::static_simd_cast<stdx::fixed_size_simd<O, simdSize<O>()>>(tmp);
  simd = simd1;
  // simd = stdx::static_simd_cast<stdx::native_simd<O>>(tmp);
}

template <typename O, typename I>
void castSimdStorer(
    void *ptr, uint64_t index, stdx::fixed_size_simd<O, simdSize<O>()> &simd
) {
  auto tmp =
      stdx::static_simd_cast<stdx::fixed_size_simd<I, simdSize<O>()>>(simd);
  tmp.copy_to(((I *)ptr) + index, stdx::vector_aligned);
}

#define CASTER(TNAME, T)                                                       \
  const Caster<T> TNAME##Casters[10] = {                                       \
      {castLoader<T, int8_t>, castStorer<T, int8_t>, castIndexer<int8_t>,      \
       castSimdLoader<T, int8_t>, castSimdStorer<T, int8_t>},                  \
      {castLoader<T, int16_t>, castStorer<T, int16_t>, castIndexer<int16_t>,   \
       castSimdLoader<T, int16_t>, castSimdStorer<T, int16_t>},                \
      {castLoader<T, int32_t>, castStorer<T, int32_t>, castIndexer<int32_t>,   \
       castSimdLoader<T, int32_t>, castSimdStorer<T, int32_t>},                \
      {castLoader<T, int64_t>, castStorer<T, int64_t>, castIndexer<int64_t>,   \
       castSimdLoader<T, int64_t>, castSimdStorer<T, int64_t>},                \
      {castLoader<T, uint8_t>, castStorer<T, uint8_t>, castIndexer<uint8_t>,   \
       castSimdLoader<T, uint8_t>, castSimdStorer<T, uint8_t>},                \
      {castLoader<T, uint16_t>, castStorer<T, uint16_t>,                       \
       castIndexer<uint16_t>, castSimdLoader<T, uint16_t>,                     \
       castSimdStorer<T, uint16_t>},                                           \
      {castLoader<T, uint32_t>, castStorer<T, uint32_t>,                       \
       castIndexer<uint32_t>, castSimdLoader<T, uint32_t>,                     \
       castSimdStorer<T, uint32_t>},                                           \
      {castLoader<T, uint64_t>, castStorer<T, uint64_t>,                       \
       castIndexer<uint64_t>, castSimdLoader<T, uint64_t>,                     \
       castSimdStorer<T, uint64_t>},                                           \
      {castLoader<T, float>, castStorer<T, float>, castIndexer<float>,         \
       castSimdLoader<T, float>, castSimdStorer<T, float>},                    \
      {castLoader<T, double>, castStorer<T, double>, castIndexer<T>,           \
       castSimdLoader<T, double>, castSimdStorer<T, double>}                   \
  };

CASTER(f64, double)
CASTER(f32, float)
CASTER(i8, int8_t)
CASTER(i16, int16_t)
CASTER(i32, int32_t)
CASTER(i64, int64_t)
CASTER(u8, uint8_t)
CASTER(u16, uint16_t)
CASTER(u32, uint32_t)
CASTER(u64, uint64_t)

template <typename T> const Caster<T> &Caster<T>::lookup(DType type) {
  if constexpr (std::is_same<T, uint8_t>::value) {
    return u8Casters[type.index];
  } else if constexpr (std::is_same<T, uint16_t>::value) {
    return u16Casters[type.index];
  } else if constexpr (std::is_same<T, uint32_t>::value) {
    return u32Casters[type.index];
  } else if constexpr (std::is_same<T, uint64_t>::value) {
    return u64Casters[type.index];
  } else if constexpr (std::is_same<T, int8_t>::value) {
    return i8Casters[type.index];
  } else if constexpr (std::is_same<T, int16_t>::value) {
    return i16Casters[type.index];
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return i32Casters[type.index];
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return i64Casters[type.index];
  } else if constexpr (std::is_same<T, float>::value) {
    return f32Casters[type.index];
  } else if constexpr (std::is_same<T, double>::value) {
    return f64Casters[type.index];
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

template const Caster<double> &Caster<double>::lookup(DType type);
template const Caster<float> &Caster<float>::lookup(DType type);
template const Caster<int8_t> &Caster<int8_t>::lookup(DType type);
template const Caster<int16_t> &Caster<int16_t>::lookup(DType type);
template const Caster<int32_t> &Caster<int32_t>::lookup(DType type);
template const Caster<int64_t> &Caster<int64_t>::lookup(DType type);
template const Caster<uint8_t> &Caster<uint8_t>::lookup(DType type);
template const Caster<uint16_t> &Caster<uint16_t>::lookup(DType type);
template const Caster<uint32_t> &Caster<uint32_t>::lookup(DType type);
template const Caster<uint64_t> &Caster<uint64_t>::lookup(DType type);
