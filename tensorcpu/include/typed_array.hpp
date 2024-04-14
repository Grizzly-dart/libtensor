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

namespace stdx = std::experimental;

template <typename T> constexpr bool isRealNum() {
  // TODO add bfloat16_t
  return std::is_same<T, float>::value || std::is_same<T, double>::value ||
         std::is_same<T, std::float16_t>::value;
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

template <typename O, typename I>
void castSimdLoader(void *ptr, uint64_t index, stdx::native_simd<O> &simd);

template <typename O> using CastLoader = O (*)(void *, uint64_t);
template <typename O> using CastStorer = void (*)(void *, uint64_t, O);
using CastIndexer = void (*)(void **dst, void *src, int64_t offset);
template <typename O>
using CastSimdLoader = void (*)(void *, uint64_t, stdx::native_simd<O> &);

template <typename T> struct Caster {
  CastLoader<T> loader = nullptr;
  CastStorer<T> storer = nullptr;
  CastIndexer indexer = nullptr;
  CastSimdLoader<T> simdLoader = nullptr;

  Caster()
      : loader(nullptr), storer(nullptr), indexer(nullptr),
        simdLoader(nullptr) {}

  constexpr Caster(
      CastLoader<T> loader, CastStorer<T> storer, CastIndexer indexer,
      CastSimdLoader<T> simdLoader
  )
      : loader(loader), storer(storer), indexer(indexer),
        simdLoader(simdLoader) {}

  Caster(Caster<T> &other)
      : loader(other.loader), storer(other.storer), indexer(other.indexer),
        simdLoader(other.simdLoader) {}

  static const Caster<T> &lookup(DType dtype);
};

extern const Caster<int64_t> i64Casters[12];

extern const Caster<double> f64Casters[12];

extern const Caster<float> f32Casters[12];

extern const Caster<int16_t> i16Casters[12];

extern const Caster<int32_t> i32Casters[12];

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

  virtual void load(uint64_t ind, T *p) const = 0;

  virtual T get(uint64_t i) const = 0;

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
    simd.copy_from(ptr + ind * width, stdx::vector_aligned);
    return simd;
  }

  void load(uint64_t ind, T *p) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
#pragma GCC ivdep
    for (uint64_t i = 0; i < elements; i++) {
      p[i] = ptr[start + i];
    }
  }

  template <typename F>
  void store(uint64_t ind, const stdx::native_simd<F> &simd) {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    if (elements == width) {
      if constexpr (std::is_same<T, F>::value) {
        simd.copy_to(ptr + start, stdx::vector_aligned);
        return;
      } else if constexpr (!std::is_same<T, F>::value) {
        stdx::simd_cast<T>(simd).copy_to(ptr + start, stdx::vector_aligned);
        return;
      }
    }
    for (uint64_t i = 0; i < elements; i++) {
      ptr[start + i] = static_cast<T>(static_cast<F>(simd[i]));
    }
  }

  template <typename F>
  void store(uint64_t ind, const F *vec, uint64_t elements) {
    uint64_t start = ind * width;
#pragma GCC ivdep
    for (uint64_t i = 0; i < elements; i++) {
      ptr[start + i] = vec[i];
    }
  }

  T get(uint64_t i) const { return ptr[i]; }

  void set(uint64_t i, T value) { ptr[i] = value; }
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

  stdx::native_simd<T> &load(uint64_t ind, stdx::native_simd<T> &simd) const {
    caster.simdLoader(ptr, ind * width, simd);
    return simd;
  }

  void load(uint64_t ind, T *vec) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      vec[i] = caster.loader(ptr, start + i);
    }
  }

  template <typename F>
  void store(uint64_t ind, const stdx::native_simd<F> &simd) {
    auto elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      caster.storer(ptr, start + i, static_cast<T>(static_cast<F>(simd[i])));
    }
  }

  template <typename F>
  void store(uint64_t ind, const F *vec, uint64_t elements) {
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      caster.storer(ptr, start + i, vec[i]);
    }
  }

  T get(uint64_t i) const { return caster.loader(ptr, i); }

  void set(uint64_t i, T value) { caster.storer(ptr, i, value); }
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

  void load(uint64_t ind, T *vec) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      vec[i] = ptr[((start + i) / size.c) % size.r];
    }
  }

  T get(uint64_t i) const { return ptr[(i / size.c) % size.r]; }
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

  void load(uint64_t ind, T *vec) const {
    uint16_t elements = calcRemainingElements(ind);
    uint64_t start = ind * width;
    for (uint64_t i = 0; i < elements; i++) {
      vec[i] = caster.loader(ptr, ((start + i) / size.c) % size.r);
    }
  }

  T get(uint64_t i) const { return caster.loader(ptr, (i / size.c) % size.r); }
};

template <typename T> class SameSimd : public ISimd<T> {
public:
  T value;

  using ISimd<T>::width;
  using ISimd<T>::length;

  SameSimd(T value, uint16_t width, int64_t length)
      : value(value), ISimd<T>(width, length){};

  SameSimd(const SameSimd &other) = default;

  stdx::native_simd<T> &load(uint64_t, stdx::native_simd<T> &simd) const {
    simd = value;
    return simd;
  }

  void load(uint64_t ind, T *vec) const {
    int16_t diff = length - ind * width;
    if (diff >= width) {
#pragma GCC ivdep
      for (int i = 0; i < width; i++) {
        vec[i] = value;
      }
    } else if (diff > 0) {
#pragma GCC ivdep
      for (int i = 0; i < diff; i++) {
        vec[i] = value;
      }
    }
  }

  T get(uint64_t) const { return value; }
};

#endif // TENSORCPU_TYPED_ARRAY_HPP

#pragma clang diagnostic pop