//
// Created by tejag on 2024-04-09.
//

#ifndef TENSORCUDA_CASTER_HPP
#define TENSORCUDA_CASTER_HPP

#include <cstdint>
#include <functional>

template <typename O, typename I>
__device__ __host__ O castLoader(void *ptr, uint64_t index) {
  return ((I *)ptr)[index];
}

template <typename O, typename I>
__device__ __host__ void castStorer(void *ptr, uint64_t index, O value) {
  ((I *)ptr)[index] = value;
}

template <typename I>
__device__ __host__ void castIndexer(void **dst, void *src, int64_t index) {
  *dst = ((I *)src) + index;
}

template <typename O> using CastLoader = O (*)(void *, uint64_t);
template <typename O> using CastStorer = void (*)(void *, uint64_t, O);
using CastOffsetter = void (*)(void **dst, void *src, int64_t offset);

template <typename T> struct [[maybe_unused]] Caster {
  CastLoader<T> loader = nullptr;
  CastStorer<T> storer = nullptr;
  CastOffsetter indexer = nullptr;
};

__constant__ std::function loaders[10] = {
    castLoader<int64_t, int8_t>,

};

__constant__ std::function intCasters[10] = {
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
    {castLoader<double, double>, castStorer<double, double>,
     castIndexer<double>},
    {castLoader<double, float>, castStorer<double, float>,
     castIndexer<float>},
};

#endif // TENSORCUDA_CASTER_HPP
