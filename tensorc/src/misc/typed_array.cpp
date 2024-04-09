#include <experimental/simd>
#include <iterator>

#include "tensorc.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;



/*
template <typename F> class TypedIterator {
public:
  using iterator_category = std::contiguous_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using value_type = F;
  using pointer = F *;
  using reference = F &;

  F *ptr;

  TypedIterator(void *ptr) : ptr(ptr) {}

  double getF64(uint64_t index) { return ((F *)ptr)[index]; }
  uint64_t getU64(uint64_t index) { return ((F *)ptr)[index]; }
  int64_t getI64(uint64_t index) { return ((F *)ptr)[index]; }
  void *offset(uint64_t index) { return ((F *)ptr) + index; }

  reference operator*() const { return *ptr; }
  pointer operator->() { return ptr; }

  TypedIterator &operator++() {
    ptr++;
    return *this;
  }

  TypedIterator operator++(int) {
    TypedIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  TypedIterator &operator--() {
    ptr--;
    return *this;
  }

  TypedIterator operator--(int) {
    TypedIterator tmp = *this;
    --(*this);
    return tmp;
  }

  TypedIterator &operator+=(difference_type n) {
    ptr += n;
    return *this;
  }

  TypedIterator operator+(difference_type n) const {
    TypedIterator tmp = *this;
    return tmp += n;
  }

  TypedIterator &operator-=(difference_type n) {
    ptr -= n;
    return *this;
  }

  TypedIterator operator-(difference_type n) const {
    TypedIterator tmp = *this;
    return tmp -= n;
  }

  difference_type operator-(const TypedIterator &rhs) const {
    return ptr - rhs.ptr;
  }

  reference operator[](size_type n) const { return *(*this + n); }

  friend bool operator==(const TypedIterator &a, const TypedIterator &b) {
    return a.ptr == b.ptr;
  };
  friend bool operator!=(const TypedIterator &a, const TypedIterator &b) {
    return a.ptr != b.ptr;
  };
};
*/

/*
class TypeArray {
public:
  virtual double getF64(void *ptr, uint64_t index) = 0;
  virtual uint64_t getU64(void *ptr, uint64_t index) = 0;
  virtual int64_t getI64(void *ptr, uint64_t index) = 0;
  virtual void *offset(void *ptr, uint64_t index);
};

class U8Array : public TypeArray {
public:

  double getF64(void *ptr, uint64_t index) { return ((uint8_t *)ptr)[index]; }
  uint64_t getU64(void *ptr, uint64_t index) { return ((uint8_t *)ptr)[index]; }
  int64_t getI64(void *ptr, uint64_t index) { return ((uint8_t *)ptr)[index]; }
  void *offset(void *ptr, uint64_t index) { return ((uint8_t *)ptr) + index; }
};

class U16Array : public TypeArray {
public:
  double getF64(void *ptr, uint64_t index) { return ((uint16_t *)ptr)[index]; }
  uint64_t getU64(void *ptr, uint64_t index) {
    return ((uint16_t *)ptr)[index];
  }
  int64_t getI64(void *ptr, uint64_t index) { return ((uint16_t *)ptr)[index]; }
  void *offset(void *ptr, uint64_t index) { return ((uint16_t *)ptr) + index; }
};
*/