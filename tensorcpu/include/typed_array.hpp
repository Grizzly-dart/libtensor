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

template <typename T, uint16_t laneSize> class Caster {
public:
  typedef T SimdType __attribute__((vector_size(sizeof(T) * laneSize)));

  using CastLoader = void (*)(T &out, const void *inp, uint64_t offset);

  using CastStorer = void (*)(void *inp, T &out, uint64_t offset);

  using CastSimdLoader =
      void (*)(SimdType &out, const void *inp, uint64_t offset);

  using CastSimdStorer = void (*)(void *out, SimdType &inp, uint64_t offset);

  CastLoader load = nullptr;
  CastStorer store = nullptr;
  CastSimdLoader loadSimd = nullptr;
  CastSimdStorer storeSimd = nullptr;

  Caster() = default;
};

struct DType {
public:
  uint8_t index;
  uint8_t bytes;
  uint8_t subIndex;

  void *offsetPtr(void *ptr, uint64_t offset) const {
    return (uint8_t *)ptr + offset * bytes;
  }

  template <typename T, uint16_t laneSize>
  void caster(Caster<T, laneSize> &caster) const {
    typedef T TSimd __attribute__((vector_size(sizeof(T) * laneSize)));
    if (index == f32Id) {
      typedef float CSimd
          __attribute__((vector_size(sizeof(float) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset
                    ) { out = (T) * ((float *)inp + offset); },
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((float *)out + offset) = (float)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (float *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy(((float *)out + offset), &tmp, sizeof(CSimd));
      };
    } else if (index == f64Id) {
      typedef double CSimd
          __attribute__((vector_size(sizeof(double) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((double *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset
                     ) { *((double *)out + offset) = (double)inp; },
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (double *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((double *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == i8Id) {
      typedef int8_t CSimd
          __attribute__((vector_size(sizeof(int8_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((int8_t *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((int8_t *)out + offset) = (int8_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (int8_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((int8_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == i16Id) {
      typedef int16_t CSimd
          __attribute__((vector_size(sizeof(int16_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset
                    ) { out = (T) * ((int16_t *)inp + offset); },
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((int16_t *)out + offset) = (int16_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (int16_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((int16_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == i32Id) {
      typedef int32_t CSimd
          __attribute__((vector_size(sizeof(int32_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((int32_t *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((int32_t *)out + offset) = (int32_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (int32_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((int32_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == i64Id) {
      typedef int64_t CSimd
          __attribute__((vector_size(sizeof(int64_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((int64_t *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((int64_t *)out + offset) = (int64_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (int64_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((int64_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == u8Id) {
      typedef uint8_t CSimd
          __attribute__((vector_size(sizeof(uint8_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((uint8_t *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((uint8_t *)out + offset) = (uint8_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (uint8_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((uint8_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == u16Id) {
      typedef uint16_t CSimd
          __attribute__((vector_size(sizeof(uint16_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((uint16_t *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((uint16_t *)out + offset) = (uint16_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (uint16_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((uint16_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == u32Id) {
      typedef uint32_t CSimd
          __attribute__((vector_size(sizeof(uint32_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((uint32_t *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((uint32_t *)out + offset) = (uint32_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (uint32_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((uint32_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else if (index == u64Id) {
      typedef uint64_t CSimd
          __attribute__((vector_size(sizeof(uint64_t) * laneSize)));
      caster.load = [](T &out, const void *inp, uint64_t offset) {
        out = (T) * ((uint64_t *)inp + offset);
      };
      caster.store = [](void *out, T &inp, uint64_t offset) {
        *((uint64_t *)out + offset) = (uint64_t)inp;
      };
      caster.loadSimd = [](TSimd &out, const void *inp, uint64_t offset) {
        CSimd tmp;
        memcpy(&tmp, (uint64_t *)inp + offset, sizeof(CSimd));
        out = __builtin_convertvector(tmp, TSimd);
      };
      caster.storeSimd = [](void *out, TSimd &inp, uint64_t offset) {
        CSimd tmp = __builtin_convertvector(inp, CSimd);
        memcpy((uint64_t *)out + offset, &tmp, sizeof(CSimd));
      };
    } else {
      throw std::invalid_argument("Invalid type");
    }
  }
};

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

#endif // TENSORCPU_TYPED_ARRAY_HPP

#pragma clang diagnostic pop