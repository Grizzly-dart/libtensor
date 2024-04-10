//
// Created by tejag on 2024-04-10.
//

#include "caster.hpp"

#include "tensorcuda.hpp"

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

__constant__ Caster<int64_t> i64Casters[8] = {
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
};

__constant__ Caster<double> f64Casters[4] = {
    {castLoader<double, float>, castStorer<double, float>,
     castIndexer<float>}, // TODO convert to bg16
    {castLoader<double, float>, castStorer<double, float>,
     castIndexer<float>}, // TODO covnert to f16
    {castLoader<double, float>, castStorer<double, float>, castIndexer<float>},
    {castLoader<double, double>, castStorer<double, double>,
     castIndexer<double>},
};