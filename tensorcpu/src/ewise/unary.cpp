#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <stdfloat>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

enum UnaryOp : uint8_t {
  Neg,
  Abs,
};

template <typename O, typename I>
const char *tcUnary(O *out, I *inp, UnaryOp op, uint64_t nel) {
  size_t width = stdx::native_simd<I>::size();
  printf("width: %zu\n", width);
  auto i1 = Simd<I>(inp, width, nel);

  Kernel kernel;
  // TODO

  std::for_each(std::execution::par_unseq, i1.countBegin(), i1.countEnd(), kernel);
  return nullptr;
}

template <typename O, typename I>
const char *tcCast(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return static_cast<O>(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcCastPlain(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return static_cast<O>(a);
  });
  return nullptr;
}

/*
template <typename O, typename I>
const char *tcCast(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return static_cast<O>(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcNeg(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return -a;
  });
  return nullptr;
}

template <typename T>
const char *tcAbs(T *out, const T *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](T a) {
    if (a == std::numeric_limits<T>::min()) {
      return std::numeric_limits<T>::max();
    } else {
      return a >= 0 ? a : -a;
    }
  });
  return nullptr;
}
 */