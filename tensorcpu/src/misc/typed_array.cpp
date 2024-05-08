#include <experimental/simd>
#include <iostream>
#include <iterator>
#include <stdfloat>

#include "macro_unwind.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

/*
template <typename T> CastLoader<T> castedLoader(const DType &type) {
  if (type.index == f32.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((float *)inp)[offset]);
    };
  } else if (type.index == f64.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((double *)inp)[offset]);
    };
  } else if (type.index == i8.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int8_t *)inp)[offset]);
    };
  } else if (type.index == i16.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int16_t *)inp)[offset]);
    };
  } else if (type.index == i32.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int32_t *)inp)[offset]);
    };
  } else if (type.index == i64.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int64_t *)inp)[offset]);
    };
  } else if (type.index == u8.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint8_t *)inp)[offset]);
    };
  } else if (type.index == u16.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint16_t *)inp)[offset]);
    };
  } else if (type.index == u32.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint32_t *)inp)[offset]);
    };
  } else if (type.index == u64.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint64_t *)inp)[offset]);
    };
  }
}

template <typename T> CastStorer<T> castedStorer(const DType &type) {
  if (type.index == f32.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((float *)out)[offset] = inp;
    };
  } else if (type.index == f64.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((double *)out)[offset] = inp;
    };
  } else if (type.index == i8.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int8_t *)out)[offset] = inp;
    };
  } else if (type.index == i16.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int16_t *)out)[offset] = inp;
    };
  } else if (type.index == i32.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int32_t *)out)[offset] = inp;
    };
  } else if (type.index == i64.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int64_t *)out)[offset] = inp;
    };
  } else if (type.index == u8.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint8_t *)out)[offset] = inp;
    };
  } else if (type.index == u16.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint16_t *)out)[offset] = inp;
    };
  } else if (type.index == u32.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint32_t *)out)[offset] = inp;
    };
  } else if (type.index == u64.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint64_t *)out)[offset] = inp;
    };
  }
}

template <typename T, uint16_t laneSize>
CastSimdLoader<T, laneSize> castedVectorLoader(const DType &type) {
  typedef T Simd __attribute__((vector_size(sizeof(T) * laneSize)));
  if (type.index == f32.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      float __attribute__((vector_size(sizeof(float) * laneSize))) tmp;
      memcpy(&tmp, ((float *)inp) + offset, sizeof(float) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == f64.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      double __attribute__((vector_size(sizeof(double) * laneSize))) tmp;
      memcpy(&tmp, ((double *)inp) + offset, sizeof(double) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == i8.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      int8_t __attribute__((vector_size(sizeof(int8_t) * laneSize))) tmp;
      memcpy(&tmp, ((int8_t *)inp) + offset, sizeof(int8_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == i16.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      int16_t __attribute__((vector_size(sizeof(int16_t) * laneSize))) tmp;
      memcpy(&tmp, ((int16_t *)inp) + offset, sizeof(int16_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == i32.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      int32_t __attribute__((vector_size(sizeof(int32_t) * laneSize))) tmp;
      memcpy(&tmp, ((int32_t *)inp) + offset, sizeof(int32_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == i64.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      int64_t __attribute__((vector_size(sizeof(int64_t) * laneSize))) tmp;
      memcpy(&tmp, ((int64_t *)inp) + offset, sizeof(int64_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == u8.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      uint8_t __attribute__((vector_size(sizeof(uint8_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint8_t *)inp) + offset, sizeof(uint8_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == u16.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      uint16_t __attribute__((vector_size(sizeof(uint16_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint16_t *)inp) + offset, sizeof(uint16_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == u32.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      uint32_t __attribute__((vector_size(sizeof(uint32_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint32_t *)inp) + offset, sizeof(uint32_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  } else if (type.index == u64.index) {
    return [](Simd &out, const void *inp, uint64_t offset) {
      uint64_t __attribute__((vector_size(sizeof(uint64_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint64_t *)inp) + offset, sizeof(uint64_t) * laneSize);
      out = __builtin_convertvector(tmp, Simd);
    };
  }
}

template <typename T, uint16_t laneSize>
CastSimdStorer<T, laneSize> castedVectorStore(const DType &type) {
  if (type.index == f32.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      float __attribute__((vector_size(sizeof(float) * laneSize))) tmp =
          __builtin_convertvector(
              inp, float __attribute__((vector_size(sizeof(float) * laneSize)))
          );
      memcpy(((float *)out) + offset, &tmp, sizeof(float) * laneSize);
    };
  } else if (type.index == f64.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      double __attribute__((vector_size(sizeof(double) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              double __attribute__((vector_size(sizeof(double) * laneSize)))
          );
      memcpy(((double *)out) + offset, &tmp, sizeof(double) * laneSize);
    };
  } else if (type.index == i8.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int8_t __attribute__((vector_size(sizeof(int8_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              int8_t __attribute__((vector_size(sizeof(int8_t) * laneSize)))
          );
      memcpy(((int8_t *)out) + offset, &tmp, sizeof(int8_t) * laneSize);
    };
  } else if (type.index == i16.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int16_t __attribute__((vector_size(sizeof(int16_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              int16_t __attribute__((vector_size(sizeof(int16_t) * laneSize)))
          );
      memcpy(((int16_t *)out) + offset, &tmp, sizeof(int16_t) * laneSize);
    };
  } else if (type.index == i32.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int32_t __attribute__((vector_size(sizeof(int32_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              int32_t __attribute__((vector_size(sizeof(int32_t) * laneSize)))
          );
      memcpy(((int32_t *)out) + offset, &tmp, sizeof(int32_t) * laneSize);
    };
  } else if (type.index == i64.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int64_t __attribute__((vector_size(sizeof(int64_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              int64_t __attribute__((vector_size(sizeof(int64_t) * laneSize)))
          );
      memcpy(((int64_t *)out) + offset, &tmp, sizeof(int64_t) * laneSize);
    };
  } else if (type.index == u8.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint8_t __attribute__((vector_size(sizeof(uint8_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              uint8_t __attribute__((vector_size(sizeof(uint8_t) * laneSize)))
          );
      memcpy(((uint8_t *)out) + offset, &tmp, sizeof(uint8_t) * laneSize);
    };
  } else if (type.index == u16.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint16_t __attribute__((vector_size(sizeof(uint16_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              uint16_t __attribute__((vector_size(sizeof(uint16_t) * laneSize)))
          );
      memcpy(((uint16_t *)out) + offset, &tmp, sizeof(uint16_t) * laneSize);
    };
  } else if (type.index == u32.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint32_t __attribute__((vector_size(sizeof(uint32_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              uint32_t __attribute__((vector_size(sizeof(uint32_t) * laneSize)))
          );
      memcpy(((uint32_t *)out) + offset, &tmp, sizeof(uint32_t) * laneSize);
    };
  } else if (type.index == u64.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint64_t __attribute__((vector_size(sizeof(uint64_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              uint64_t __attribute__((vector_size(sizeof(uint64_t) * laneSize)))
          );
      memcpy(((uint64_t *)out) + offset, &tmp, sizeof(uint64_t) * laneSize);
    };
  }
}

#define CASTEDFUNCS(T)                                                         \
  template CastLoader<T> castedLoader<T>(const DType &type);                   \
  template CastStorer<T> castedStorer<T>(const DType &type);                   \
  template CastSimdLoader<T, simdSize<T>()>                                    \
  castedVectorLoader<T, simdSize<T>()>(const DType &type);                     \
  template CastSimdStorer<T, simdSize<T>()>                                    \
  castedVectorStore<T, simdSize<T>()>(const DType &type);

UNWIND1_ALL_TYPES(CASTEDFUNCS)
 */