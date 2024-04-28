#include <algorithm>
#include <cmath>
#include <execution>
#include <future>
#include <limits>
#include <stdfloat>

#include "tensorcpu.hpp"
#include "typed_array.hpp"
#include "macro_unwind.hpp"

enum UnaryOp : uint8_t {
  Neg,
  Abs,
};

template <typename O, typename I>
const char *tcUnary(O *out, I *inp, UnaryOp op, uint64_t nel) {
  size_t width = stdx::native_simd<I>::size();
  printf("width: %zu\n", width);
  auto i1 = Accessor<I>(inp, width, nel);

  Kernel kernel;
  switch (op) {
  case Neg:
    kernel = makeKernel(out, inp, i1, width, [](I a) { return -a; });
    break;
  case Abs:
    kernel = makeKernel(out, inp, i1, width, [](I a) {
      if (a == std::numeric_limits<I>::min()) {
        return std::numeric_limits<I>::max();
      } else {
        return a >= 0 ? a : -a;
      }
    });
    break;
  }

  std::for_each(
      std::execution::par_unseq, i1.countBegin(), i1.countEnd(), kernel
  );
  return nullptr;
}

template <typename O, typename I>
const char *tcCast(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return static_cast<O>(a);
  });
  return nullptr;
}

#define TCCAST(O, I) \
  template const char *tcCast(O *out, const I *inp, uint64_t nel);

// UNWIND2_ALL_TYPES(TCCAST)

/*
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