//
// Created by tejag on 2024-04-09.
//

#ifndef TENSORCUDA_CASTER_HPP
#define TENSORCUDA_CASTER_HPP

#include <cstdint>
#include <functional>

template <typename T> constexpr bool isRealNum() {
  return std::is_same<T, float>::value || std::is_same<T, double>::value;
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

template <typename O, typename I>
__device__ __host__ O castLoader(void *ptr, uint64_t index);

template <typename O, typename I>
__device__ __host__ void castStorer(void *ptr, uint64_t index, O value);

template <typename I>
__device__ __host__ void castIndexer(void **dst, void *src, int64_t index);

template <typename O> using CastLoader = O (*)(void *, uint64_t);
template <typename O> using CastStorer = void (*)(void *, uint64_t, O);
using CastOffsetter = void (*)(void **dst, void *src, int64_t offset);

template <typename T> struct Caster {
  CastLoader<T> loader = nullptr;
  CastStorer<T> storer = nullptr;
  CastOffsetter indexer = nullptr;

  __device__ __host__ Caster()
      : loader(nullptr), storer(nullptr), indexer(nullptr) {}

  __device__ __host__
  constexpr Caster(CastLoader<T> loader, CastStorer<T> storer, CastOffsetter indexer)
      : loader(loader), storer(storer), indexer(indexer) {}

  __device__ __host__ Caster(Caster<T> &other)
      : loader(other.loader), storer(other.storer), indexer(other.indexer) {}
};

extern __constant__ Caster<int64_t> i64Casters[8];

extern __constant__ Caster<double> f64Casters[4];

#endif // TENSORCUDA_CASTER_HPP
