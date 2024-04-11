#include <experimental/simd>
#include <iterator>
#include <stdfloat>

#include "tensorc.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

const Caster<int64_t> i64Casters[8] = {
        {castLoader<int64_t, int8_t>,   castStorer<int64_t, int8_t>,
                castIndexer<int8_t>},
        {castLoader<int64_t, int16_t>,  castStorer<int64_t, int16_t>,
                castIndexer<int16_t>},
        {castLoader<int64_t, int32_t>,  castStorer<int64_t, int32_t>,
                castIndexer<int32_t>},
        {castLoader<int64_t, int64_t>,  castStorer<int64_t, int64_t>,
                castIndexer<int64_t>},
        {castLoader<int64_t, uint8_t>,  castStorer<int64_t, uint8_t>,
                castIndexer<uint8_t>},
        {castLoader<int64_t, uint16_t>, castStorer<int64_t, uint16_t>,
                castIndexer<uint16_t>},
        {castLoader<int64_t, uint32_t>, castStorer<int64_t, uint32_t>,
                castIndexer<uint32_t>},
        {castLoader<int64_t, uint64_t>, castStorer<int64_t, uint64_t>,
                castIndexer<uint64_t>},
};

const Caster<double> f64Casters[4] = {
        {castLoader<double, float>,  castStorer<double, float>,
                                                                castIndexer<std::bfloat16_t>},
        {castLoader<double, float>,  castStorer<double, float>,
                                                                castIndexer<std::float16_t>},
        {castLoader<double, float>,  castStorer<double, float>, castIndexer<float>},
        {castLoader<double, double>, castStorer<double, double>,
                                                                castIndexer<double>},
};