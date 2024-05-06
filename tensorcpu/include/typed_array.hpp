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

struct DType {
public:
  uint8_t index;
  uint8_t bytes;
  uint8_t subIndex;

  void *offsetPtr(void *ptr, uint64_t offset) const {
    return (uint8_t *)ptr + offset * bytes;
  }
};

const DType i8 = {0, 1, 0};
const DType i16 = {1, 2, 1};
const DType i32 = {2, 4, 2};
const DType i64 = {3, 8, 3};
const DType u8 = {4, 1, 4};
const DType u16 = {5, 2, 5};
const DType u32 = {6, 4, 6};
const DType u64 = {7, 8, 7};
const DType f32 = {8, 4, 2};
const DType f64 = {9, 4, 3};
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

template <typename T>
using CastLoader1 = void (*)(T &out, const void *inp, uint64_t offset);

template <typename T>
using CastStorer1 = void (*)(void *inp, T &out, uint64_t offset);

template <typename T, uint16_t laneSize>
using CastSimdLoader1 = void (*)(
    T __attribute__((vector_size(sizeof(T) * laneSize))) & out, const void *inp,
    uint64_t offset
);

template <typename T, uint16_t laneSize>
using CastSimdStorer1 = void (*)(
    void *out, T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
    uint64_t offset
);

template <typename T> extern CastLoader1<T> castedLoader(const DType &type);

template <typename T> extern CastStorer1<T> castedStorer(const DType &type);

template <typename T, uint16_t laneSize>
extern CastSimdLoader1<T, laneSize> castedVectorStore(const DType &type);

template <typename T, uint16_t laneSize>
extern CastSimdStorer1<T, laneSize> castedVectorStore(const DType &type);

template <typename O, typename I> O castLoader(void *ptr, uint64_t index);

template <typename O, typename I>
void castStorer(void *ptr, uint64_t index, O value);

template <typename I> void *castIndexer(void *src, int64_t index);

template <typename O, typename I>
void castSimdLoader(
    void *ptr, uint64_t index, stdx::fixed_size_simd<O, simdSize<O>()> &simd
);

template <typename O, typename I>
void castSimdStorer(
    void *ptr, uint64_t index, stdx::fixed_size_simd<O, simdSize<O>()> &simd
);

template <typename O> using CastLoader = O (*)(void *, uint64_t);
template <typename O> using CastStorer = void (*)(void *, uint64_t, O);
using CastIndexer = void *(*)(void *src, int64_t offset);
template <typename O>
using CastSimdLoader =
    void (*)(void *, uint64_t, stdx::fixed_size_simd<O, simdSize<O>()> &);
template <typename O>
using CastSimdStorer =
    void (*)(void *, uint64_t, stdx::fixed_size_simd<O, simdSize<O>()> &);

template <typename T> struct Caster {
  CastLoader<T> loader = nullptr;
  CastStorer<T> storer = nullptr;
  CastIndexer indexer = nullptr;
  CastSimdLoader<T> simdLoader = nullptr;
  CastSimdStorer<T> simdStorer = nullptr;

  Caster()
      : loader(nullptr), storer(nullptr), indexer(nullptr), simdLoader(nullptr),
        simdStorer(nullptr) {}

  constexpr Caster(
      CastLoader<T> loader, CastStorer<T> storer, CastIndexer indexer,
      CastSimdLoader<T> simdLoader, CastSimdStorer<T> simdStorer
  )
      : loader(loader), storer(storer), indexer(indexer),
        simdLoader(simdLoader), simdStorer(simdStorer) {}

  Caster(Caster<T> &other)
      : loader(other.loader), storer(other.storer), indexer(other.indexer),
        simdLoader(other.simdLoader), simdStorer(other.simdStorer) {}

  static const Caster<T> &lookup(DType dtype);
};

template <typename T> class IAccessor {
public:
  uint16_t width;
  uint64_t length;

  IAccessor(uint16_t width, uint64_t length) : width(width), length(length){};

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((length + width - 1) / width);
  }

  stdx::fixed_size_simd<T, simdSize<T>()> load(uint64_t i) const {
    stdx::fixed_size_simd<T, simdSize<T>()> simd;
    return load(i, simd);
  }

  virtual stdx::fixed_size_simd<T, simdSize<T>()> &load(
      uint64_t i, stdx::fixed_size_simd<T, simdSize<T>()> &simd
  ) const = 0;

  virtual void load(uint64_t ind, T *p) const = 0;

  virtual T get(uint64_t i) const = 0;

  virtual ~IAccessor() = default;

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

template <typename T> class Accessor : public IAccessor<T> {
public:
  T *ptr;

  using IAccessor<T>::width;
  using IAccessor<T>::length;
  using IAccessor<T>::calcRemainingElements;

  Accessor(T *ptr, uint16_t width, int64_t length)
      : ptr(ptr), IAccessor<T>(width, length){};

  Accessor(const Accessor &other) = default;

  [[gnu::always_inline]] stdx::fixed_size_simd<T, simdSize<T>()> &load(
      uint64_t ind, stdx::fixed_size_simd<T, simdSize<T>()> &simd
  ) const {
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
  [[gnu::always_inline]] void store(
      uint64_t ind, const stdx::fixed_size_simd<F, simdSize<F>()> simd
  ) const {
    uint64_t start = ind * width;
    uint16_t elements = calcRemainingElements(ind);
    if (elements == width) {
      if constexpr (std::is_same<T, F>::value) {
        simd.copy_to(ptr + start, stdx::vector_aligned);
        return;
      } else if constexpr (!std::is_same<T, F>::value) {
        stdx::static_simd_cast<stdx::fixed_size_simd<T, simdSize<F>()>>(simd)
            .copy_to(ptr + start, stdx::vector_aligned);
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

template <typename T> class CastAccessor : public IAccessor<T> {
public:
  const Caster<T> &caster;
  void *ptr;

  using IAccessor<T>::width;
  using IAccessor<T>::length;
  using IAccessor<T>::calcRemainingElements;

  CastAccessor(
      const Caster<T> &caster, void *ptr, uint16_t width, int64_t length
  )
      : caster(caster), ptr(ptr), IAccessor<T>(width, length){};

  CastAccessor(const CastAccessor &other) = default;

  stdx::fixed_size_simd<T, simdSize<T>()> &load(
      uint64_t ind, stdx::fixed_size_simd<T, simdSize<T>()> &simd
  ) const {
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
  void store(
      uint64_t ind, const stdx::fixed_size_simd<F, simdSize<F>()> &simd
  ) {
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

template <typename T> class RwiseAccessor : public IAccessor<T> {
public:
  T *ptr;
  Dim2 size;

  using IAccessor<T>::width;
  using IAccessor<T>::length;
  using IAccessor<T>::calcRemainingElements;

  RwiseAccessor(T *ptr, uint16_t width, uint64_t length, Dim2 size)
      : ptr(ptr), size(size), IAccessor<T>(width, length){};

  RwiseAccessor(const RwiseAccessor &other) = default;

  stdx::fixed_size_simd<T, simdSize<T>()> &load(
      uint64_t ind, stdx::fixed_size_simd<T, simdSize<T>()> &simd
  ) const {
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

template <typename T> class CastRwiseAccessor : public IAccessor<T> {
public:
  const Caster<T> &caster;
  Dim2 size;
  void *ptr;

  using IAccessor<T>::width;
  using IAccessor<T>::length;
  using IAccessor<T>::calcRemainingElements;

  CastRwiseAccessor(
      const Caster<T> &caster, void *ptr, uint16_t width, uint64_t length,
      Dim2 size
  )
      : caster(caster), ptr(ptr), size(size), IAccessor<T>(width, length){};

  CastRwiseAccessor(const CastRwiseAccessor &other) = default;

  stdx::fixed_size_simd<T, simdSize<T>()> &load(
      uint64_t ind, stdx::fixed_size_simd<T, simdSize<T>()> &simd
  ) const {
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

template <typename T> class SameAccessor : public IAccessor<T> {
public:
  T value;

  using IAccessor<T>::width;
  using IAccessor<T>::length;

  SameAccessor(T value, uint16_t width, int64_t length)
      : value(value), IAccessor<T>(width, length){};

  SameAccessor(const SameAccessor &other) = default;

  stdx::fixed_size_simd<T, simdSize<T>()> &load(
      uint64_t, stdx::fixed_size_simd<T, simdSize<T>()> &simd
  ) const {
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