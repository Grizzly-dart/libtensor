#include <experimental/simd>
#include <iterator>
#include <stdfloat>

#include "tensorc.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

template <typename O, typename I> O castLoader(void *ptr, uint64_t index) {
  return static_cast<O>(((I *)ptr)[index]);
}

template <typename O, typename I>
void castStorer(void *ptr, uint64_t index, O value) {
  ((I *)ptr)[index] = static_cast<I>(value);
}

template <typename I> void castIndexer(void **dst, void *src, int64_t index) {
  *dst = ((I *)src) + index;
}

const Caster<int64_t> i64Casters[12] = {
    {castLoader<int64_t, int8_t>, castStorer<int64_t, int8_t>,
     castIndexer<int8_t>},
    {castLoader<int64_t, int16_t>, castStorer<int64_t, int16_t>,
     castIndexer<int16_t>},
    {castLoader<int64_t, int32_t>, castStorer<int64_t, int32_t>,
     castIndexer<int32_t>},
    {castLoader<int64_t, int64_t>, castStorer<int64_t, int64_t>,
     castIndexer<int64_t>},
    {castLoader<int64_t, uint8_t>, castStorer<int64_t, uint8_t>,
     castIndexer<uint8_t>},
    {castLoader<int64_t, uint16_t>, castStorer<int64_t, uint16_t>,
     castIndexer<uint16_t>},
    {castLoader<int64_t, uint32_t>, castStorer<int64_t, uint32_t>,
     castIndexer<uint32_t>},
    {castLoader<int64_t, uint64_t>, castStorer<int64_t, uint64_t>,
     castIndexer<uint64_t>},
    {castLoader<int64_t, std::bfloat16_t>, castStorer<int64_t, std::bfloat16_t>,
     castIndexer<std::bfloat16_t>},
    {castLoader<int64_t, std::float16_t>, castStorer<int64_t, std::float16_t>,
     castIndexer<std::float16_t>},
    {castLoader<int64_t, float>, castStorer<int64_t, float>,
     castIndexer<float>},
    {castLoader<int64_t, double>, castStorer<int64_t, double>,
     castIndexer<double>},
};

const Caster<double> f64Casters[12] = {
    {castLoader<double, int8_t>, castStorer<double, int8_t>,
     castIndexer<int8_t>},
    {castLoader<double, int16_t>, castStorer<double, int16_t>,
     castIndexer<int16_t>},
    {castLoader<double, int32_t>, castStorer<double, int32_t>,
     castIndexer<int32_t>},
    {castLoader<double, int64_t>, castStorer<double, int64_t>,
     castIndexer<int64_t>},
    {castLoader<double, uint8_t>, castStorer<double, uint8_t>,
     castIndexer<uint8_t>},
    {castLoader<double, uint16_t>, castStorer<double, uint16_t>,
     castIndexer<uint16_t>},
    {castLoader<double, uint32_t>, castStorer<double, uint32_t>,
     castIndexer<uint32_t>},
    {castLoader<double, uint64_t>, castStorer<double, uint64_t>,
     castIndexer<uint64_t>},
    {castLoader<double, std::bfloat16_t>, castStorer<double, std::bfloat16_t>,
     castIndexer<std::bfloat16_t>},
    {castLoader<double, std::float16_t>, castStorer<double, std::float16_t>,
     castIndexer<std::float16_t>},
    {castLoader<double, float>, castStorer<double, float>, castIndexer<float>},
    {castLoader<double, double>, castStorer<double, double>,
     castIndexer<double>},
};

const Caster<float> f32Casters[12] = {
    {castLoader<float, int8_t>, castStorer<float, int8_t>, castIndexer<int8_t>},
    {castLoader<float, int16_t>, castStorer<float, int16_t>,
     castIndexer<int16_t>},
    {castLoader<float, int32_t>, castStorer<float, int32_t>,
     castIndexer<int32_t>},
    {castLoader<float, int64_t>, castStorer<float, int64_t>,
     castIndexer<int64_t>},
    {castLoader<float, uint8_t>, castStorer<float, uint8_t>,
     castIndexer<uint8_t>},
    {castLoader<float, uint16_t>, castStorer<float, uint16_t>,
     castIndexer<uint16_t>},
    {castLoader<float, uint32_t>, castStorer<float, uint32_t>,
     castIndexer<uint32_t>},
    {castLoader<float, uint64_t>, castStorer<float, uint64_t>,
     castIndexer<uint64_t>},
    {castLoader<float, std::bfloat16_t>, castStorer<float, std::bfloat16_t>,
     castIndexer<std::bfloat16_t>},
    {castLoader<float, std::float16_t>, castStorer<float, std::float16_t>,
     castIndexer<std::float16_t>},
    {castLoader<float, float>, castStorer<float, float>, castIndexer<float>},
    {castLoader<float, double>, castStorer<float, double>, castIndexer<double>},
};

const Caster<int16_t> i16Casters[12] = {
    {castLoader<int16_t, int8_t>, castStorer<int16_t, int8_t>,
     castIndexer<int8_t>},
    {castLoader<int16_t, int16_t>, castStorer<int16_t, int16_t>,
     castIndexer<int16_t>},
    {},
    {},
    {castLoader<int16_t, uint8_t>, castStorer<int16_t, uint8_t>,
     castIndexer<uint8_t>},
    {castLoader<int16_t, uint16_t>, castStorer<int16_t, uint16_t>,
     castIndexer<uint16_t>},
    {},
    {},
    {},
    {},
    {},
    {},
};

const Caster<int32_t> i32Casters[12] = {
    {castLoader<int32_t, int8_t>, castStorer<int32_t, int8_t>,
     castIndexer<int8_t>},
    {castLoader<int32_t, int16_t>, castStorer<int32_t, int16_t>,
     castIndexer<int16_t>},
    {castLoader<int32_t, int32_t>, castStorer<int32_t, int32_t>,
     castIndexer<int32_t>},
    {},
    {castLoader<int32_t, uint8_t>, castStorer<int32_t, uint8_t>,
     castIndexer<uint8_t>},
    {castLoader<int32_t, uint16_t>, castStorer<int32_t, uint16_t>,
     castIndexer<uint16_t>},
    {castLoader<int32_t, uint32_t>, castStorer<int32_t, uint32_t>,
     castIndexer<uint32_t>},
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