#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-dcl21-cpp"
//
// Created by tejag on 2024-04-06.
//

#ifndef TENSORC_TYPED_ARRAY_HPP
#define TENSORC_TYPED_ARRAY_HPP

#include <experimental/simd>
#include <iterator>

#include "tensorc.hpp"

namespace stdx = std::experimental;

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

template <typename T> class Val {
public:
  T *ptr;
  uint64_t width;

  Val(T *ptr, uint64_t width) : ptr(ptr), width(width){};

  Val(const Val &other) : ptr(other.ptr), width(other.width){};

  template <typename U> Val &operator=(stdx::native_simd<U> simd) {
    for (int i = 0; i < width; i++) {
      ptr[i] = static_cast<T>(static_cast<U>(simd[i]));
    }
    return *this;
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

#endif // TENSORC_TYPED_ARRAY_HPP

#pragma clang diagnostic pop