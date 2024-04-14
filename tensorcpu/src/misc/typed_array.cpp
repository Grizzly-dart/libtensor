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

template <typename I> void castIndexer(void **dst, void *src, int64_t index) {
  *dst = ((I *)src) + index;
}

template <typename O, typename I>
void castSimdLoader(void *ptr, uint64_t index, stdx::native_simd<O> &simd) {
  stdx::fixed_size_simd<I, stdx::simd_size<O>::value> tmp;
  tmp.copy_from(((I *)ptr) + index, stdx::element_aligned);
  simd = stdx::static_simd_cast<stdx::native_simd<O>>(tmp);
}

template <typename O, typename I>
void castSimdStorer(void *ptr, uint64_t index, stdx::native_simd<O> &simd) {
  auto tmp = stdx::static_simd_cast<
      stdx::fixed_size_simd<I, stdx::simd_size<O>::value>>(simd);
  tmp.copy_to(((I *)ptr) + index, stdx::element_aligned);
}

const Caster<int64_t> i64Casters[12] = {
    {castLoader<int64_t, int8_t>, castStorer<int64_t, int8_t>,
     castIndexer<int8_t>, castSimdLoader<int64_t, int8_t>},
    {castLoader<int64_t, int16_t>, castStorer<int64_t, int16_t>,
     castIndexer<int16_t>, castSimdLoader<int64_t, int16_t>},
    {castLoader<int64_t, int32_t>, castStorer<int64_t, int32_t>,
     castIndexer<int32_t>, castSimdLoader<int64_t, int32_t>},
    {castLoader<int64_t, int64_t>, castStorer<int64_t, int64_t>,
     castIndexer<int64_t>, castSimdLoader<int64_t, int64_t>},
    {castLoader<int64_t, uint8_t>, castStorer<int64_t, uint8_t>,
     castIndexer<uint8_t>, castSimdLoader<int64_t, uint8_t>},
    {castLoader<int64_t, uint16_t>, castStorer<int64_t, uint16_t>,
     castIndexer<uint16_t>, castSimdLoader<int64_t, uint16_t>},
    {castLoader<int64_t, uint32_t>, castStorer<int64_t, uint32_t>,
     castIndexer<uint32_t>, castSimdLoader<int64_t, uint32_t>},
    {castLoader<int64_t, uint64_t>, castStorer<int64_t, uint64_t>,
     castIndexer<uint64_t>, castSimdLoader<int64_t, uint64_t>},
    {},
    {},
    {castLoader<int64_t, float>, castStorer<int64_t, float>, castIndexer<float>,
     castSimdLoader<int64_t, float>},
    {castLoader<int64_t, double>, castStorer<int64_t, double>,
     castIndexer<double>, castSimdLoader<int64_t, double>},
};

const Caster<double> f64Casters[12] = {
    {castLoader<double, int8_t>, castStorer<double, int8_t>,
     castIndexer<int8_t>, castSimdLoader<double, int8_t>},
    {castLoader<double, int16_t>, castStorer<double, int16_t>,
     castIndexer<int16_t>, castSimdLoader<double, int16_t>},
    {castLoader<double, int32_t>, castStorer<double, int32_t>,
     castIndexer<int32_t>, castSimdLoader<double, int32_t>},
    {castLoader<double, int64_t>, castStorer<double, int64_t>,
     castIndexer<int64_t>, castSimdLoader<double, int64_t>},
    {castLoader<double, uint8_t>, castStorer<double, uint8_t>,
     castIndexer<uint8_t>, castSimdLoader<double, uint8_t>},
    {castLoader<double, uint16_t>, castStorer<double, uint16_t>,
     castIndexer<uint16_t>, castSimdLoader<double, uint16_t>},
    {castLoader<double, uint32_t>, castStorer<double, uint32_t>,
     castIndexer<uint32_t>, castSimdLoader<double, uint32_t>},
    {castLoader<double, uint64_t>, castStorer<double, uint64_t>,
     castIndexer<uint64_t>, castSimdLoader<double, uint64_t>},
    {},
    {},
    {castLoader<double, float>, castStorer<double, float>, castIndexer<float>,
     castSimdLoader<double, float>},
    {castLoader<double, double>, castStorer<double, double>,
     castIndexer<double>, castSimdLoader<double, double>},
};

const Caster<float> f32Casters[12] = {
    {castLoader<float, int8_t>, castStorer<float, int8_t>, castIndexer<int8_t>,
     castSimdLoader<float, int8_t>},
    {castLoader<float, int16_t>, castStorer<float, int16_t>,
     castIndexer<int16_t>, castSimdLoader<float, int16_t>},
    {castLoader<float, int32_t>, castStorer<float, int32_t>,
     castIndexer<int32_t>, castSimdLoader<float, int32_t>},
    {castLoader<float, int64_t>, castStorer<float, int64_t>,
     castIndexer<int64_t>, castSimdLoader<float, int64_t>},
    {castLoader<float, uint8_t>, castStorer<float, uint8_t>,
     castIndexer<uint8_t>, castSimdLoader<float, uint8_t>},
    {castLoader<float, uint16_t>, castStorer<float, uint16_t>,
     castIndexer<uint16_t>, castSimdLoader<float, uint16_t>},
    {castLoader<float, uint32_t>, castStorer<float, uint32_t>,
     castIndexer<uint32_t>, castSimdLoader<float, uint32_t>},
    {castLoader<float, uint64_t>, castStorer<float, uint64_t>,
     castIndexer<uint64_t>, castSimdLoader<float, uint64_t>},
    {},
    {},
    {castLoader<float, float>, castStorer<float, float>, castIndexer<float>,
     castSimdLoader<float, float>},
    {castLoader<float, double>, castStorer<float, double>, castIndexer<double>,
     castSimdLoader<float, double>},
};

const Caster<int16_t> i16Casters[12] = {
    {castLoader<int16_t, int8_t>, castStorer<int16_t, int8_t>,
     castIndexer<int8_t>, castSimdLoader<int16_t, int8_t>},
    {castLoader<int16_t, int16_t>, castStorer<int16_t, int16_t>,
     castIndexer<int16_t>, castSimdLoader<int16_t, int16_t>},
    {},
    {},
    {castLoader<int16_t, uint8_t>, castStorer<int16_t, uint8_t>,
     castIndexer<uint8_t>, castSimdLoader<int16_t, uint8_t>},
    {castLoader<int16_t, uint16_t>, castStorer<int16_t, uint16_t>,
     castIndexer<uint16_t>, castSimdLoader<int16_t, uint16_t>},
    {},
    {},
    {},
    {},
    {},
    {},
};

const Caster<int32_t> i32Casters[12] = {
    {castLoader<int32_t, int8_t>, castStorer<int32_t, int8_t>,
     castIndexer<int8_t>, castSimdLoader<int32_t, int8_t>},
    {castLoader<int32_t, int16_t>, castStorer<int32_t, int16_t>,
     castIndexer<int16_t>, castSimdLoader<int32_t, int16_t>},
    {castLoader<int32_t, int32_t>, castStorer<int32_t, int32_t>,
     castIndexer<int32_t>, castSimdLoader<int32_t, int32_t>},
    {},
    {castLoader<int32_t, uint8_t>, castStorer<int32_t, uint8_t>,
     castIndexer<uint8_t>, castSimdLoader<int32_t, uint8_t>},
    {castLoader<int32_t, uint16_t>, castStorer<int32_t, uint16_t>,
     castIndexer<uint16_t>, castSimdLoader<int32_t, uint16_t>},
    {castLoader<int32_t, uint32_t>, castStorer<int32_t, uint32_t>,
     castIndexer<uint32_t>, castSimdLoader<int32_t, uint32_t>},
    {},
    {},
    {},
    {},
    {},
};

template <typename T> const Caster<T> &Caster<T>::lookup(DType type) {
  if constexpr (std::is_same<T, float>::value) {
    return f32Casters[type.index];
  } else if constexpr (std::is_same<T, double>::value) {
    return f64Casters[type.index];
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return i64Casters[type.index];
  } else if constexpr (std::is_same<T, int16_t>::value) {
    return i16Casters[type.index];
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return i32Casters[type.index];
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

template const Caster<double> &Caster<double>::lookup(DType type);
template const Caster<float> &Caster<float>::lookup(DType type);
template const Caster<int64_t> &Caster<int64_t>::lookup(DType type);
template const Caster<int32_t> &Caster<int32_t>::lookup(DType type);
template const Caster<int16_t> &Caster<int16_t>::lookup(DType type);