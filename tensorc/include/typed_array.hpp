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

#include "tensorc.hpp"

namespace stdx = std::experimental;

struct DType {
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

class Range {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = int64_t;
  using pointer = value_type *;
  using reference = value_type &;

  int64_t index;

  explicit Range(int64_t index) : index(index){};

  Range(const Range &other) = default;

  reference operator*() { return index; }

  pointer operator->() { return &index; }

  Range &operator++() {
    index += 1;
    return *this;
  }

  Range operator++(int) & {
    Range tmp = *this;
    ++(*this);
    return tmp;
  }

  Range &operator--() {
    index -= 1;
    return *this;
  }

  Range operator--(int) & {
    Range tmp = *this;
    --(*this);
    return tmp;
  }

  Range &operator+=(difference_type n) {
    index += n;
    return *this;
  }

  Range &operator-=(difference_type n) {
    index -= n;
    return *this;
  }

  Range operator+(difference_type n) const {
    Range tmp = *this;
    return tmp += n;
  }

  Range operator-(difference_type n) const {
    Range tmp = *this;
    return tmp -= n;
  }

  difference_type operator-(const Range &rhs) const {
    difference_type ret = index - rhs.index;
    return ret;
  }

  friend bool operator==(const Range &a, const Range &b) {
    return a.index == b.index;
  };

  friend bool operator!=(const Range &a, const Range &b) {
    return a.index != b.index;
  };

  friend bool operator<(const Range &a, const Range &b) {
    return a.index < b.index;
  };

  friend bool operator>(const Range &a, const Range &b) {
    return a.index > b.index;
  };

  friend bool operator<=(const Range &a, const Range &b) {
    return a.index <= b.index;
  };

  friend bool operator>=(const Range &a, const Range &b) {
    return a.index >= b.index;
  };
};

template <typename T> class Caster {
public:
  virtual ~Caster() = default;

  virtual T at(void *ptr, uint64_t index) = 0;
  virtual void store(void *ptr, uint64_t index, T value) = 0;

  virtual int64_t indexOffset(int64_t index, int64_t offset) = 0;

  static Caster<T> typed(DType type) {
    if (type.index == f32.index) {
      return CasterImpl<T, float>();
    } else if (type.index == f64.index) {
      return CasterImpl<T, double>();
    }
  }
};

template <typename T, typename F> class ScalarCaster : public Caster<T> {
public:
  F value;

  explicit ScalarCaster(F value) : value(value) {}

  ~ScalarCaster() override = default;

  stdx::native_simd<T> &loadSimd(
      uint64_t index, stdx::native_simd<T> &simd, uint16_t elements
  ) override {
    for (int i = 0; i < elements; i++) {
      simd[i] = static_cast<T>(value);
    }
    return simd;
  }

  void storeSimd(uint64_t index, stdx::native_simd<T> &simd, uint16_t elements)
      override {}

  T at(uint64_t index) override { return static_cast<T>(value); }

  void store(uint64_t index, T v) override {}

  int64_t indexOffset(int64_t index, int64_t offset) override {
    return index + offset * sizeof(F);
  }
};

template <typename T, typename F> class CasterImpl : public Caster<T> {
public:
  explicit CasterImpl(F *ptr) : ptr(ptr) {}

  ~CasterImpl() override = default;

  stdx::native_simd<T> &loadSimd(
      uint64_t index, stdx::native_simd<T> &simd, uint16_t elements
  ) {
    for (int i = 0; i < elements; i++) {
      simd[i] = static_cast<T>(ptr[index + i]);
    }
    return simd;
  }

  void storeSimd(uint64_t index, stdx::native_simd<T> &simd, uint16_t elements)
      override {
    for (int i = 0; i < elements; i++) {
      ptr[index + i] = static_cast<F>(simd[i]);
    }
  }

  T at(uint64_t index) { return static_cast<T>(ptr[index]); }
  void store(uint64_t index, T value) { ptr[index] = static_cast<F>(value); }

  int64_t indexOffset(int64_t index, int64_t offset) {
    return index + offset * sizeof(F);
  }
};

template <typename T> class SimdIter {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;

  uint16_t width;

  explicit SimdIter(uint16_t width) : width(width){};

  [[nodiscard]] virtual Range countBegin() const = 0;

  [[nodiscard]] virtual Range countEnd() const = 0;

  virtual value_type operator*() const = 0;

  virtual value_type at(size_type i) const = 0;

  virtual value_type &loadAt(size_type i, value_type &simd) const = 0;

  virtual ~SimdIter() = default;
};

template <typename T> class SimdIterator : public SimdIter<T> {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;

  T *ptr;
  int64_t index;
  int64_t length;

  SimdIterator(T *ptr, uint16_t width, int64_t length, int64_t index = 0)
      : ptr(ptr), length(length), index(index), SimdIter<T>(width){};

  SimdIterator(const SimdIterator &other) = default;

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((length + this->width - 1) / this->width);
  }

  value_type operator*() const {
    value_type simd;
    return this->loadAt(0, simd);
  }

  value_type at(size_type i) const {
    value_type simd;
    return this->loadAt(i, simd);
  }

  value_type &loadAt(size_type ind, value_type &simd) const {
    for (size_type i = 0; i < this->width; i++) {
      if (index + ind * this->width + i >= length) {
        break;
      }
      simd[i] = ptr[index + ind * this->width + i];
    }
    return simd;
  }

  void storeAt(size_type ind, const value_type &simd) {
    size_type ptrIndex = index + ind * this->width;
    for (size_type i = 0; i < this->width; i++) {
      if (ptrIndex + i >= length) {
        break;
      }
      ptr[ptrIndex + i] = simd[i];
    }
  }

  SimdIterator &operator=(const value_type &simd) {
    for (int i = 0; i < this->width; i++) {
      if (index + i >= length) {
        break;
      }
      ptr[index + i] = simd[i];
    }
    return *this;
  }

  SimdIterator operator[](size_type i) const {
    return SimdIterator(ptr, this->width, length, index + i * this->width);
  }

  SimdIterator &operator++() {
    index += this->width;
    return *this;
  }

  SimdIterator operator++(int) {
    SimdIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  SimdIterator &operator--() {
    index -= this->width;
    return *this;
  }

  SimdIterator operator--(int) {
    SimdIterator tmp = *this;
    --(*this);
    return tmp;
  }

  SimdIterator &operator+=(difference_type n) {
    index += this->width * n;
    return *this;
  }

  SimdIterator &operator-=(difference_type n) {
    index -= this->width * n;
    return *this;
  }

  SimdIterator operator+(difference_type n) const {
    return SimdIterator(ptr, this->width, length, index + this->width * n);
  }

  SimdIterator operator-(difference_type n) const {
    return SimdIterator(ptr, this->width, length, index - this->width * n);
  }

  difference_type operator-(const SimdIterator &rhs) const {
    return (index - rhs.index + this->width - 1) / this->width;
  }

  friend bool operator==(const SimdIterator &a, const SimdIterator &b) {
    return a.index == b.index;
  };

  friend bool operator!=(const SimdIterator &a, const SimdIterator &b) {
    return a.index != b.index;
  };

  friend bool operator<(const SimdIterator &a, const SimdIterator &b) {
    return a.index < b.index;
  };

  friend bool operator>(const SimdIterator &a, const SimdIterator &b) {
    return a.index > b.index;
  };

  friend bool operator<=(const SimdIterator &a, const SimdIterator &b) {
    return a.index <= b.index;
  };

  friend bool operator>=(const SimdIterator &a, const SimdIterator &b) {
    return a.index >= b.index;
  };
};

template <typename T> class CastingSimdIterator : public SimdIter<T> {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;

  Caster<T> *caster;
  int64_t index;
  int64_t length;

  CastingSimdIterator(
      Caster<T> *caster, uint16_t width, int64_t length, int64_t index = 0
  )
      : caster(caster), length(length), index(index), SimdIter<T>(width){};

  CastingSimdIterator(const CastingSimdIterator &other) = default;

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((length + this->width - 1) / this->width);
  }

  value_type operator*() const {
    value_type simd;
    return this->loadAt(0, simd);
  }

  value_type at(size_type i) const {
    value_type simd;
    return this->loadAt(i, simd);
  }

  value_type &loadAt(size_type ind, value_type &simd) const {
    auto elements = std::min(this->width, length - index - ind * this->width);
    return caster->loadSimd(index + ind * this->width, simd, elements);
  }

  void storeAt(size_type ind, const value_type &simd) {
    auto elements = std::min(this->width, length - index - ind * this->width);
    caster->storeSimd(index + ind * this->width, simd, elements);
  }

  CastingSimdIterator &operator=(const value_type &simd) {
    storeAt(0, simd);
    return *this;
  }

  CastingSimdIterator operator[](size_type i) const {
    return CastingSimdIterator(
        caster, this->width, length, index + i * this->width
    );
  }

  CastingSimdIterator &operator++() {
    index += this->width;
    return *this;
  }

  CastingSimdIterator operator++(int) {
    CastingSimdIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  CastingSimdIterator &operator--() {
    index -= this->width;
    return *this;
  }

  CastingSimdIterator operator--(int) {
    CastingSimdIterator tmp = *this;
    --(*this);
    return tmp;
  }

  CastingSimdIterator &operator+=(difference_type n) {
    index += this->width * n;
    return *this;
  }

  CastingSimdIterator &operator-=(difference_type n) {
    index -= this->width * n;
    return *this;
  }

  CastingSimdIterator operator+(difference_type n) const {
    return CastingSimdIterator(
        caster, this->width, length, index + this->width * n
    );
  }

  CastingSimdIterator operator-(difference_type n) const {
    return CastingSimdIterator(
        caster, this->width, length, index - this->width * n
    );
  }

  difference_type operator-(const CastingSimdIterator &rhs) const {
    return (index - rhs.index + this->width - 1) / this->width;
  }

  friend bool operator==(
      const CastingSimdIterator &a, const CastingSimdIterator &b
  ) {
    return a.index == b.index;
  };

  friend bool operator!=(
      const CastingSimdIterator &a, const CastingSimdIterator &b
  ) {
    return a.index != b.index;
  };

  friend bool operator<(
      const CastingSimdIterator &a, const CastingSimdIterator &b
  ) {
    return a.index < b.index;
  };

  friend bool operator>(
      const CastingSimdIterator &a, const CastingSimdIterator &b
  ) {
    return a.index > b.index;
  };

  friend bool operator<=(
      const CastingSimdIterator &a, const CastingSimdIterator &b
  ) {
    return a.index <= b.index;
  };

  friend bool operator>=(
      const CastingSimdIterator &a, const CastingSimdIterator &b
  ) {
    return a.index >= b.index;
  };
};

template <typename T> class RwiseSimdIterator : public SimdIter<T> {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;

  T *ptr;
  Dim3 size;
  int64_t index;

  RwiseSimdIterator(T *ptr, uint16_t width, Dim3 size, int64_t index = 0)
      : ptr(ptr), size(size), index(index), SimdIter<T>(width){};

  RwiseSimdIterator(const RwiseSimdIterator &other) = default;

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((size.nel() + this->width - 1) / this->width);
  }

  value_type operator*() const {
    value_type ret;
    return this->loadAt(0, ret);
  }

  value_type at(size_type i) const {
    value_type ret;
    return (*this + i).loadAt(i, ret);
  }

  value_type &loadAt(size_type ind, value_type &simd) const {
    for (size_type i = 0; i < this->width; i++) {
      if (this->index + ind * this->width + i >= size.nel()) {
        break;
      }
      simd[i] = ptr[((this->index + ind * this->width + i) / size.c) % size.r];
    }
    return simd;
  }

  RwiseSimdIterator operator[](size_type i) const {
    return RwiseSimdIterator(ptr, this->width, size, index + i * this->width);
  }

  RwiseSimdIterator &operator++() {
    index += this->width;
    return *this;
  }

  RwiseSimdIterator operator++(int) {
    RwiseSimdIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  RwiseSimdIterator &operator--() {
    index -= this->width;
    return *this;
  }

  RwiseSimdIterator operator--(int) {
    RwiseSimdIterator tmp = *this;
    --(*this);
    return tmp;
  }

  RwiseSimdIterator &operator+=(difference_type n) {
    index += this->width * n;
    return *this;
  }

  RwiseSimdIterator &operator-=(difference_type n) {
    index -= this->width * n;
    return *this;
  }

  RwiseSimdIterator operator+(difference_type n) const {
    return RwiseSimdIterator(ptr, this->width, size, index + this->width * n);
  }

  RwiseSimdIterator operator-(difference_type n) const {
    return RwiseSimdIterator(ptr, this->width, size, index - this->width * n);
  }

  difference_type operator-(const RwiseSimdIterator &rhs) const {
    return index - rhs.index;
  }

  friend bool operator==(
      const RwiseSimdIterator &a, const RwiseSimdIterator &b
  ) {
    return a.index == b.index;
  };

  friend bool operator!=(
      const RwiseSimdIterator &a, const RwiseSimdIterator &b
  ) {
    return a.index != b.index;
  };

  friend bool operator<(
      const RwiseSimdIterator &a, const RwiseSimdIterator &b
  ) {
    return a.index < b.index;
  };

  friend bool operator>(
      const RwiseSimdIterator &a, const RwiseSimdIterator &b
  ) {
    return a.index > b.index;
  };

  friend bool operator<=(
      const RwiseSimdIterator &a, const RwiseSimdIterator &b
  ) {
    return a.index <= b.index;
  };

  friend bool operator>=(
      const RwiseSimdIterator &a, const RwiseSimdIterator &b
  ) {
    return a.index >= b.index;
  };
};

template <typename T> class CastingRwiseSimdIterator : public SimdIter<T> {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;

  Caster<T> *caster;
  Dim3 size;
  int64_t index;

  CastingRwiseSimdIterator(
      Caster<T> *caster, uint16_t width, Dim3 size, int64_t index = 0
  )
      : caster(caster), size(size), index(index), SimdIter<T>(width){};

  CastingRwiseSimdIterator(const CastingRwiseSimdIterator &other) = default;

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((size.nel() + this->width - 1) / this->width);
  }

  value_type operator*() const {
    value_type ret;
    return this->loadAt(0, ret);
  }

  value_type at(size_type i) const {
    value_type ret;
    return (*this + i).loadAt(i, ret);
  }

  value_type &loadAt(size_type ind, value_type &simd) const {
    for (size_type i = 0; i < this->width; i++) {
      if (this->index + ind * this->width + i >= size.nel()) {
        break;
      }
      simd[i] =
          caster->at(((this->index + ind * this->width + i) / size.c) % size.r);
    }
    return simd;
  }

  CastingRwiseSimdIterator operator[](size_type i) const {
    return CastingRwiseSimdIterator(
        caster, this->width, size, index + i * this->width
    );
  }

  CastingRwiseSimdIterator &operator++() {
    index += this->width;
    return *this;
  }

  CastingRwiseSimdIterator operator++(int) {
    CastingRwiseSimdIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  CastingRwiseSimdIterator &operator--() {
    index -= this->width;
    return *this;
  }

  CastingRwiseSimdIterator operator--(int) {
    CastingRwiseSimdIterator tmp = *this;
    --(*this);
    return tmp;
  }

  CastingRwiseSimdIterator &operator+=(difference_type n) {
    index += this->width * n;
    return *this;
  }

  CastingRwiseSimdIterator &operator-=(difference_type n) {
    index -= this->width * n;
    return *this;
  }

  CastingRwiseSimdIterator operator+(difference_type n) const {
    return CastingRwiseSimdIterator(
        caster, this->width, size, index + this->width * n
    );
  }

  CastingRwiseSimdIterator operator-(difference_type n) const {
    return CastingRwiseSimdIterator(
        caster, this->width, size, index - this->width * n
    );
  }

  difference_type operator-(const CastingRwiseSimdIterator &rhs) const {
    return index - rhs.index;
  }

  friend bool operator==(
      const CastingRwiseSimdIterator &a, const CastingRwiseSimdIterator &b
  ) {
    return a.index == b.index;
  };

  friend bool operator!=(
      const CastingRwiseSimdIterator &a, const CastingRwiseSimdIterator &b
  ) {
    return a.index != b.index;
  };

  friend bool operator<(
      const CastingRwiseSimdIterator &a, const CastingRwiseSimdIterator &b
  ) {
    return a.index < b.index;
  };

  friend bool operator>(
      const CastingRwiseSimdIterator &a, const CastingRwiseSimdIterator &b
  ) {
    return a.index > b.index;
  };

  friend bool operator<=(
      const CastingRwiseSimdIterator &a, const CastingRwiseSimdIterator &b
  ) {
    return a.index <= b.index;
  };

  friend bool operator>=(
      const CastingRwiseSimdIterator &a, const CastingRwiseSimdIterator &b
  ) {
    return a.index >= b.index;
  };
};

template <typename T> class ScalarSimdInpIter : public SimdIter<T> {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;
  using reference = value_type &;

  T value;
  int64_t index;
  int64_t length;

  ScalarSimdInpIter(T value, uint16_t width, int64_t length, int64_t index = 0)
      : value(value), length(length), index(index), SimdIter<T>(width){};

  ScalarSimdInpIter(const ScalarSimdInpIter &other) = default;

  value_type operator*() const {
    value_type ret;
    this->loadAt(0, ret);
    return ret;
  }

  value_type at(size_type i) const {
    value_type ret;
    this->loadAt(i, ret);
    return ret;
  }

  value_type &loadAt(size_type ind, value_type &simd) const {
    uint16_t diff = length - index - ind * this->width;
    if (diff >= this->width) {
      simd = value;
      return simd;
    } else {
      for (int i = 0; i < diff; i++) {
        simd[i] = value;
      }
      return simd;
    }
  }

  ScalarSimdInpIter operator[](size_type i) const { return *this + i; }

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((length + this->width - 1) / this->width);
  }

  ScalarSimdInpIter &operator++() {
    index += this->width;
    length -= this->width;
    return *this;
  }

  ScalarSimdInpIter operator++(int) {
    ScalarSimdInpIter tmp = *this;
    ++(*this);
    return tmp;
  }

  ScalarSimdInpIter &operator--() {
    index -= this->width;
    length += this->width;
    return *this;
  }

  ScalarSimdInpIter operator--(int) {
    ScalarSimdInpIter tmp = *this;
    --(*this);
    return tmp;
  }

  ScalarSimdInpIter &operator+=(difference_type n) {
    index += this->width * n;
    length -= this->width * n;
    return *this;
  }

  ScalarSimdInpIter &operator-=(difference_type n) {
    index -= this->width * n;
    length += this->width * n;
    return *this;
  }

  ScalarSimdInpIter operator+(difference_type n) const {
    return ScalarSimdInpIter(
        value, this->width, length - this->width * n, index + this->width * n
    );
  }

  ScalarSimdInpIter operator-(difference_type n) const {
    return ScalarSimdInpIter(
        value, this->width, length + this->width * n, index - this->width * n
    );
  }

  difference_type operator-(const ScalarSimdInpIter &rhs) const {
    return index - rhs.index;
  }

  friend bool operator==(
      const ScalarSimdInpIter &a, const ScalarSimdInpIter &b
  ) {
    return a.index == b.index;
  };

  friend bool operator!=(
      const ScalarSimdInpIter &a, const ScalarSimdInpIter &b
  ) {
    return a.index != b.index;
  };

  friend bool operator<(
      const ScalarSimdInpIter &a, const ScalarSimdInpIter &b
  ) {
    return a.index < b.index;
  };

  friend bool operator>(
      const ScalarSimdInpIter &a, const ScalarSimdInpIter &b
  ) {
    return a.index > b.index;
  };

  friend bool operator<=(
      const ScalarSimdInpIter &a, const ScalarSimdInpIter &b
  ) {
    return a.index <= b.index;
  };

  friend bool operator>=(
      const ScalarSimdInpIter &a, const ScalarSimdInpIter &b
  ) {
    return a.index >= b.index;
  };
};

template <typename T> class CastingScalarSimdIterator : public SimdIter<T> {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;
  using reference = value_type &;

  T value;
  int64_t index;
  int64_t length;

  CastingScalarSimdIterator(
      T value, uint16_t width, int64_t length, int64_t index = 0
  )
      : value(value), length(length), index(index), SimdIter<T>(width){};

  CastingScalarSimdIterator(const CastingScalarSimdIterator &other) = default;

  value_type operator*() const {
    value_type ret;
    this->loadAt(0, ret);
    return ret;
  }

  value_type at(size_type i) const {
    value_type ret;
    this->loadAt(i, ret);
    return ret;
  }

  value_type &loadAt(size_type ind, value_type &simd) const {
    uint16_t diff = length - index - ind * this->width;
    if (diff >= this->width) {
      simd = value;
      return simd;
    } else {
      for (int i = 0; i < diff; i++) {
        simd[i] = value;
      }
      return simd;
    }
  }

  CastingScalarSimdIterator operator[](size_type i) const { return *this + i; }

  [[nodiscard]] Range countBegin() const { return Range(0); }

  [[nodiscard]] Range countEnd() const {
    return Range((length + this->width - 1) / this->width);
  }

  CastingScalarSimdIterator &operator++() {
    index += this->width;
    length -= this->width;
    return *this;
  }

  CastingScalarSimdIterator operator++(int) {
    CastingScalarSimdIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  CastingScalarSimdIterator &operator--() {
    index -= this->width;
    length += this->width;
    return *this;
  }

  CastingScalarSimdIterator operator--(int) {
    CastingScalarSimdIterator tmp = *this;
    --(*this);
    return tmp;
  }

  CastingScalarSimdIterator &operator+=(difference_type n) {
    index += this->width * n;
    length -= this->width * n;
    return *this;
  }

  CastingScalarSimdIterator &operator-=(difference_type n) {
    index -= this->width * n;
    length += this->width * n;
    return *this;
  }

  CastingScalarSimdIterator operator+(difference_type n) const {
    return CastingScalarSimdIterator(
        value, this->width, length - this->width * n, index + this->width * n
    );
  }

  CastingScalarSimdIterator operator-(difference_type n) const {
    return CastingScalarSimdIterator(
        value, this->width, length + this->width * n, index - this->width * n
    );
  }

  difference_type operator-(const CastingScalarSimdIterator &rhs) const {
    return index - rhs.index;
  }

  friend bool operator==(
      const CastingScalarSimdIterator &a, const CastingScalarSimdIterator &b
  ) {
    return a.index == b.index;
  };

  friend bool operator!=(
      const CastingScalarSimdIterator &a, const CastingScalarSimdIterator &b
  ) {
    return a.index != b.index;
  };

  friend bool operator<(
      const CastingScalarSimdIterator &a, const CastingScalarSimdIterator &b
  ) {
    return a.index < b.index;
  };

  friend bool operator>(
      const CastingScalarSimdIterator &a, const CastingScalarSimdIterator &b
  ) {
    return a.index > b.index;
  };

  friend bool operator<=(
      const CastingScalarSimdIterator &a, const CastingScalarSimdIterator &b
  ) {
    return a.index <= b.index;
  };

  friend bool operator>=(
      const CastingScalarSimdIterator &a, const CastingScalarSimdIterator &b
  ) {
    return a.index >= b.index;
  };
};

#endif // TENSORC_TYPED_ARRAY_HPP

#pragma clang diagnostic pop