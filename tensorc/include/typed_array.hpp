#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-dcl21-cpp"
//
// Created by tejag on 2024-04-06.
//

#ifndef TENSORC_TYPED_ARRAY_HPP
#define TENSORC_TYPED_ARRAY_HPP

#include <experimental/simd>
#include <iterator>
#include <memory>
#include <stdfloat>

#include "range.hpp"
#include "tensorc.hpp"

namespace stdx = std::experimental;

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

struct DType {
public:
  uint8_t index;
  uint8_t bytes;
  uint8_t subIndex;
};

const DType i8 = {0, 1, 0};
const DType i16 = {1, 2, 1};
const DType i32 = {2, 4, 2};
const DType i64 = {3, 8, 3};
const DType u8 = {4, 1, 4};
const DType u16 = {5, 2, 5};
const DType u32 = {6, 4, 6};
const DType u64 = {7, 8, 7};
const DType bf16 = {8, 2, 0};
const DType f16 = {9, 2, 1};
const DType f32 = {10, 4, 2};
const DType f64 = {11, 4, 3};

const DType dtypes[] = {i8,  i16, i32,  i64, u8,  u16,
                        u32, u64, bf16, f16, f32, f64};

template <typename T> constexpr DType dtypeOf()  {
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
  } else if constexpr (std::is_same<T, std::bfloat16_t>::value) {
    return bf16;
  } else if constexpr (std::is_same<T, std::float16_t>::value) {
    return f16;
  } else if constexpr (std::is_same<T, float>::value) {
    return f32;
  } else if constexpr (std::is_same<T, double>::value) {
    return f64;
  } else {
    throw std::invalid_argument("Invalid type");
  }
}

template <typename O, typename I> O castLoader(void *ptr, uint64_t index);

template <typename O, typename I>
void castStorer(void *ptr, uint64_t index, O value);

template <typename I> void castIndexer(void **dst, void *src, int64_t index);

template <typename O> using CastLoader = O (*)(void *, uint64_t);
template <typename O> using CastStorer = void (*)(void *, uint64_t, O);
using CastIndexer = void (*)(void **dst, void *src, int64_t offset);

template <typename T> struct Caster {
  CastLoader<T> loader = nullptr;
  CastStorer<T> storer = nullptr;
  CastIndexer indexer = nullptr;

  Caster() : loader(nullptr), storer(nullptr), indexer(nullptr) {}

  constexpr Caster(
      CastLoader<T> loader, CastStorer<T> storer, CastIndexer indexer
  )
      : loader(loader), storer(storer), indexer(indexer) {}

  Caster(Caster<T> &other)
      : loader(other.loader), storer(other.storer), indexer(other.indexer) {}

  static const Caster<T> &lookup(DType dtype);
};

extern const Caster<int64_t> i64Casters[12];

extern const Caster<double> f64Casters[12];

extern const Caster<float> f32Casters[12];

template <typename T> class ISimd {
public:
  uint16_t width;
  uint64_t length;

  ISimd(uint16_t width, uint64_t length) : width(width), length(length){};

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((length + width - 1) / width);
  }

  stdx::native_simd<T> load(uint64_t i) const {
    stdx::native_simd<T> simd;
    return load(i, simd);
  }

  virtual stdx::native_simd<T> &load(uint64_t i, stdx::native_simd<T> &simd)
      const = 0;

  virtual ~ISimd() = default;

  [[nodiscard]] uint16_t calcRemainingElements(uint64_t ind) const {
    int64_t diff = int64_t(length) - ind * width;
    if (diff < 0) {
      return 0;
    } else if (diff < width) {
      return diff;
    } else {
      return width;
    }
  }
};

template <typename T> class Simd : public ISimd<T> {
public:
  T *ptr;

  using ISimd<T>::width;
  using ISimd<T>::length;
  using ISimd<T>::calcRemainingElements;

  Simd(T *ptr, uint16_t width, int64_t length)
      : ptr(ptr), ISimd<T>(width, length){};

  Simd(const Simd &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      simd[i] = ptr[start + i];
    }
    return simd;
  }

  template <typename F> void store(uint64_t ind, const stdx::simd<F> &simd) {
    uint64_t start = ind * width;
    uint16_t elements = calcRemainingElements(ind);
    for (uint64_t i = 0; i < elements; i++) {
      ptr[start + i] = static_cast<T>(static_cast<F>(simd[i]));
    }
  }
};

template <typename T> class CastSimd : public ISimd<T> {
public:
  const Caster<T> &caster;
  void *ptr;

  using ISimd<T>::width;
  using ISimd<T>::length;
  using ISimd<T>::calcRemainingElements;

  CastSimd(const Caster<T> &caster, void *ptr, uint16_t width, int64_t length)
      : caster(caster), ptr(ptr), ISimd<T>(width, length){};

  CastSimd(const CastSimd &other) = default;

  static CastSimd<T> create(
      DType dtype, void *ptr, uint16_t width, int64_t length
  ) {
    if constexpr (isRealNum<T>()) {
      return CastSimd<T>(f64Casters[dtype.index], ptr, width, length);
    } else if constexpr (isAnyInt<T>()) {
      return CastSimd<T>(i64Casters[dtype.index], ptr, width, length);
    } else {
      throw std::invalid_argument("Invalid type");
    }
  }

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      simd[i] = caster.loader(ptr, start + i);
    }
    return simd;
  }

  template <typename F> void store(uint64_t ind, const stdx::native_simd<F> &simd) {
    auto elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      caster.storer(ptr, start + i, static_cast<T>(static_cast<F>(simd[i])));
    }
  }
};

template <typename T> class RwiseSimd : public ISimd<T> {
public:
  T *ptr;
  Dim2 size;

  using ISimd<T>::width;
  using ISimd<T>::length;
  using ISimd<T>::calcRemainingElements;

  RwiseSimd(T *ptr, uint16_t width, uint64_t length, Dim2 size)
      : ptr(ptr), size(size), ISimd<T>(width, length){};

  RwiseSimd(const RwiseSimd &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      simd[i] = ptr[((start + i) / size.c) % size.r];
    }
    return simd;
  }
};

template <typename T> class CastRwiseSimd : public ISimd<T> {
public:
  const Caster<T> &caster;
  Dim2 size;
  void *ptr;

  using ISimd<T>::width;
  using ISimd<T>::length;
  using ISimd<T>::calcRemainingElements;

  CastRwiseSimd(
      const Caster<T> &caster, void *ptr, uint16_t width, uint64_t length,
      Dim2 size
  )
      : caster(caster), ptr(ptr), size(size), ISimd<T>(width, length){};

  CastRwiseSimd(const CastRwiseSimd &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      simd[i] = caster.loader(ptr, ((start + i) / size.c) % size.r);
    }
    return simd;
  }
};

template <typename T> class SameSimd : public ISimd<T> {
public:
  T value;

  using ISimd<T>::width;
  using ISimd<T>::length;

  SameSimd(T value, uint16_t width, int64_t length)
      : value(value), ISimd<T>(width, length){};

  SameSimd(const SameSimd &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    int16_t diff = length - ind * width;
    if (diff >= width) {
      simd = value;
    } else if (diff > 0) {
      for (int i = 0; i < diff; i++) {
        simd[i] = value;
      }
    }
    return simd;
  }
};

#define UNWIND3_SAME(A, OP, NAME) OP(A, A, A, NAME)

#define UNWIND2_SAME(A, OP, NAME) OP(A, A, NAME)

#define UWIND3_ALL_TYPES(OP, NAME)                                             \
  UNWIND3_SAME(int8_t, OP, NAME)                                               \
  UNWIND3_SAME(int16_t, OP, NAME)                                              \
  UNWIND3_SAME(int32_t, OP, NAME)                                              \
  UNWIND3_SAME(int64_t, OP, NAME)                                              \
  UNWIND3_SAME(uint8_t, OP, NAME)                                              \
  UNWIND3_SAME(uint16_t, OP, NAME)                                             \
  UNWIND3_SAME(uint32_t, OP, NAME)                                             \
  UNWIND3_SAME(uint64_t, OP, NAME)                                             \
  UNWIND3_SAME(float, OP, NAME)                                                \
  UNWIND3_SAME(double, OP, NAME)

#define UNWIND2_ALL_TYPES(OP, NAME)                                            \
  UNWIND2_SAME(int8_t, OP, NAME)                                               \
  UNWIND2_SAME(int16_t, OP, NAME)                                              \
  UNWIND2_SAME(int32_t, OP, NAME)                                              \
  UNWIND2_SAME(int64_t, OP, NAME)                                              \
  UNWIND2_SAME(uint8_t, OP, NAME)                                              \
  UNWIND2_SAME(uint16_t, OP, NAME)                                             \
  UNWIND2_SAME(uint32_t, OP, NAME)                                             \
  UNWIND2_SAME(uint64_t, OP, NAME)                                             \
  UNWIND2_SAME(float, OP, NAME)                                                \
  UNWIND2_SAME(double, OP, NAME)

#define UNWIND3_2(A, B, OP, NAME)                                              \
  OP(A, A, A, NAME)                                                            \
  OP(A, A, B, NAME)                                                            \
  OP(A, B, A, NAME)                                                            \
  OP(A, B, B, NAME)                                                            \
  OP(B, B, B, NAME)                                                            \
  OP(B, A, B, NAME)                                                            \
  OP(B, B, A, NAME)                                                            \
  OP(B, A, A, NAME)

#define UNWIND2_2(A, B, OP, NAME)                                              \
  OP(A, A, NAME)                                                               \
  OP(A, B, NAME)                                                               \
  OP(B, A, NAME)                                                               \
  OP(B, B, NAME)

#endif // TENSORC_TYPED_ARRAY_HPP

#pragma clang diagnostic pop