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

template <typename T> class SimdInpIterator {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;

  T *ptr;
  uint16_t width;
  int64_t length;

  SimdInpIterator(T *ptr, uint16_t width, int64_t length)
      : ptr(ptr), width(width), length(length){};

  SimdInpIterator(const SimdInpIterator &other) = default;

  Range countBegin() { return Range(0); }

  Range countEnd() { return Range((length + width - 1) / width); }

  value_type operator*() const {
    value_type simd;
    for (int i = 0; i < width; i++) {
      if (i >= length) {
        break;
      }
      simd[i] = ptr[i];
    }
    return simd;
  }

  SimdInpIterator &operator=(const value_type &simd) {
    for (int i = 0; i < width; i++) {
      if (i >= length) {
        break;
      }
      ptr[i] = simd[i];
    }
    return *this;
  }

  SimdInpIterator operator[](size_type i) {
    return SimdInpIterator(ptr + i * width, width, length - i * width);
  }

  SimdInpIterator &operator++() {
    ptr += width;
    length -= width;
    return *this;
  }

  SimdInpIterator operator++(int) {
    SimdInpIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  SimdInpIterator &operator--() {
    ptr -= width;
    length += width;
    return *this;
  }

  SimdInpIterator operator--(int) {
    SimdInpIterator tmp = *this;
    --(*this);
    return tmp;
  }

  SimdInpIterator &operator+=(difference_type n) {
    ptr += width * n;
    length -= width * n;
    return *this;
  }

  SimdInpIterator &operator-=(difference_type n) {
    ptr -= width * n;
    length += width * n;
    return *this;
  }

  SimdInpIterator operator+(difference_type n) const { return *this += n; }

  SimdInpIterator operator-(difference_type n) const { return *this -= n; }

  difference_type operator-(const SimdInpIterator &rhs) const {
    return ptr - rhs.ptr;
  }

  friend bool operator==(const SimdInpIterator &a, const SimdInpIterator &b) {
    return a.ptr == b.ptr;
  };

  friend bool operator!=(const SimdInpIterator &a, const SimdInpIterator &b) {
    return a.ptr != b.ptr;
  };

  friend bool operator<(const SimdInpIterator &a, const SimdInpIterator &b) {
    return a.ptr < b.ptr;
  };

  friend bool operator>(const SimdInpIterator &a, const SimdInpIterator &b) {
    return a.ptr > b.ptr;
  };

  friend bool operator<=(const SimdInpIterator &a, const SimdInpIterator &b) {
    return a.ptr <= b.ptr;
  };

  friend bool operator>=(const SimdInpIterator &a, const SimdInpIterator &b) {
    return a.ptr >= b.ptr;
  };
};

template <typename T> class RwiseSimdIterator {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;

  T *ptr;
  uint16_t width;
  Dim3 size;
  int64_t index;

  RwiseSimdIterator(T *ptr, uint16_t width, Dim3 size, int64_t index)
      : ptr(ptr), width(width), size(size), index(index){};

  RwiseSimdIterator(const RwiseSimdIterator &other) = default;

  Range countBegin() { return Range(0); }

  Range countEnd() { return Range((size.nel() + width - 1) / width); }

  value_type operator*() const {
    value_type simd{0};
    for (int i = 0; i < width; i++) {
      if (index + i >= size.nel()) {
        break;
      }
      simd[i] = ptr[((index + i) / size.c) % size.r];
    }
    return simd;
  }

  RwiseSimdIterator operator[](size_type i) const {
    return RwiseSimdIterator(ptr, width, size, index + i * width);
  }

  RwiseSimdIterator &operator++() {
    index += width;
    return *this;
  }

  RwiseSimdIterator operator++(int) {
    RwiseSimdIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  RwiseSimdIterator &operator--() {
    index -= width;
    return *this;
  }

  RwiseSimdIterator operator--(int) {
    RwiseSimdIterator tmp = *this;
    --(*this);
    return tmp;
  }

  RwiseSimdIterator &operator+=(difference_type n) {
    index += width * n;
    return *this;
  }

  RwiseSimdIterator &operator-=(difference_type n) {
    index -= width * n;
    return *this;
  }

  RwiseSimdIterator operator+(difference_type n) const {
        return RwiseSimdIterator(ptr, width, size, index + width * n);
  }

  RwiseSimdIterator operator-(difference_type n) const {
    return RwiseSimdIterator(ptr, width, size, index - width * n);
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

template <typename T> class ScalarSimdInpIter {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = stdx::native_simd<T>;
  using pointer = value_type *;
  using reference = value_type &;

  T value;
  value_type simd;
  uint16_t width;
  int64_t index;
  int64_t length;

  ScalarSimdInpIter(T value, uint16_t width, int64_t length, int64_t index)
      : value(value), simd(value), width(width), length(length), index(index){};

  ScalarSimdInpIter(const ScalarSimdInpIter &other) = default;

  value_type operator*() const {
    uint16_t diff = length - index;
    if (diff >= width) {
      return simd;
    } else {
      value_type ret{0};
      for (int i = 0; i < diff; i++) {
        ret[i] = value;
      }
      return ret;
    }
  }
  ScalarSimdInpIter operator[](size_type i) const { return *this + i; }

  Range countBegin() { return Range(0); }

  Range countEnd() { return Range((length + width - 1) / width); }

  ScalarSimdInpIter &operator++() {
    index += width;
    length -= width;
    return *this;
  }

  ScalarSimdInpIter operator++(int) {
    ScalarSimdInpIter tmp = *this;
    ++(*this);
    return tmp;
  }

  ScalarSimdInpIter &operator--() {
    index -= width;
    length += width;
    return *this;
  }

  ScalarSimdInpIter operator--(int) {
    ScalarSimdInpIter tmp = *this;
    --(*this);
    return tmp;
  }

  ScalarSimdInpIter &operator+=(difference_type n) {
    index += width * n;
    length -= width * n;
    return *this;
  }

  ScalarSimdInpIter &operator-=(difference_type n) {
    index -= width * n;
    length += width * n;
    return *this;
  }

  ScalarSimdInpIter operator+(difference_type n) const {
    return ScalarSimdInpIter(
        value, width, length - width * n, index + width * n
    );
  }

  ScalarSimdInpIter operator-(difference_type n) const {
    return ScalarSimdInpIter(
        value, width, length + width * n, index - width * n
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