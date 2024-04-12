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

#include "range.hpp"
#include "tensorc.hpp"

namespace stdx = std::experimental;

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
};

extern const Caster<int64_t> i64Casters[8];

extern const Caster<double> f64Casters[4];

template <typename T> class SimdIter {
public:
  uint16_t width;
  uint64_t length;

  SimdIter(uint16_t width, uint64_t length) : width(width), length(length){};

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

  virtual ~SimdIter() = default;
};

template <typename T> class SimdIterator : public SimdIter<T> {
public:
  T *ptr;

  using SimdIter<T>::width;
  using SimdIter<T>::length;

  SimdIterator(T *ptr, uint16_t width, int64_t length)
      : ptr(ptr), SimdIter<T>(width, length){};

  SimdIterator(const SimdIterator &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    for (uint64_t i = 0; i < width; i++) {
      if (ind * width + i >= length) {
        break;
      }
      simd[i] = ptr[ind * width + i];
    }
    return simd;
  }

  template<typename F>
  void store(uint64_t ind, const stdx::simd<F> &simd) {
    uint64_t ptrIndex = ind * width;
    for (uint64_t i = 0; i < width; i++) {
      if (ptrIndex + i >= length) {
        break;
      }
      ptr[ptrIndex + i] = static_cast<T>(static_cast<F>(simd[i]));
    }
  }
};

template <typename T> class CastingSimdIterator : public SimdIter<T> {
public:
  Caster<T> caster;
  void *ptr;

  using SimdIter<T>::width;
  using SimdIter<T>::length;

  CastingSimdIterator(
      Caster<T> caster, void *ptr, uint16_t width, int64_t length
  )
      : caster(caster), ptr(ptr), SimdIter<T>(width, length){};

  CastingSimdIterator(const CastingSimdIterator &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    auto elements = std::min(width, length - ind * width);
    for (uint64_t i = 0; i < elements; i++) {
      simd[i] = caster.loader(ptr, ind * width + i);
    }
  }

  void store(uint64_t ind, const stdx::native_simd<T> &simd) {
    auto elements = std::min(width, length - ind * width);
    for (uint64_t i = 0; i < elements; i++) {
      caster.storer(ptr, ind * width + i, simd[i]);
    }
  }
};

template <typename T> class RwiseSimdIterator : public SimdIter<T> {
public:
  T *ptr;
  Dim2 size;

  using SimdIter<T>::width;
  using SimdIter<T>::length;

  RwiseSimdIterator(T *ptr, uint16_t width, uint64_t length, Dim2 size)
      : ptr(ptr), size(size), SimdIter<T>(width, length){};

  RwiseSimdIterator(const RwiseSimdIterator &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    for (uint64_t i = 0; i < width; i++) {
      if (ind * width + i >= size.nel()) {
        break;
      }
      simd[i] = ptr[((ind * width + i) / size.c) % size.r];
    }
    return simd;
  }
};

template <typename T> class CastingRwiseSimdIterator : public SimdIter<T> {
public:
  Caster<T> caster;
  Dim2 size;
  void *ptr;

  using SimdIter<T>::width;
  using SimdIter<T>::length;

  CastingRwiseSimdIterator(
      Caster<T> caster, void *ptr, uint16_t width, uint64_t length, Dim2 size
  )
      : caster(caster), ptr(ptr), size(size), SimdIter<T>(width, length){};

  CastingRwiseSimdIterator(const CastingRwiseSimdIterator &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    for (uint64_t i = 0; i < width; i++) {
      if (ind * width + i >= size.nel()) {
        break;
      }
      simd[i] = caster.loader(
          ptr, ((ind * width + i) / size.c) % size.r
      );
    }
    return simd;
  }
};

template <typename T> class ScalarSimdInpIter : public SimdIter<T> {
public:
  T value;

  using SimdIter<T>::width;
  using SimdIter<T>::length;

  ScalarSimdInpIter(T value, uint16_t width, int64_t length)
      : value(value), SimdIter<T>(width, length){};

  ScalarSimdInpIter(const ScalarSimdInpIter &other) = default;

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    uint16_t diff = length - ind * width;
    if (diff >= width) {
      simd = value;
      return simd;
    } else {
      for (int i = 0; i < diff; i++) {
        simd[i] = value;
      }
      return simd;
    }
  }
};

#define UNWIND2(A, B, OP, NAME)                                                \
  OP(A, A, A, NAME)                                                            \
  OP(A, A, B, NAME)                                                            \
  OP(A, B, A, NAME)                                                            \
  OP(A, B, B, NAME)                                                            \
  OP(B, B, B, NAME)                                                            \
  OP(B, A, B, NAME)                                                            \
  OP(B, B, A, NAME)                                                            \
  OP(B, A, A, NAME)

#endif // TENSORC_TYPED_ARRAY_HPP

#pragma clang diagnostic pop