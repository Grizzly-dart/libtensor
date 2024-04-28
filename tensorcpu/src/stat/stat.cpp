#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename T>
class Mean {
public:
  T mean = 0;
  uint32_t n = 0;

  void consume(T sample)  {
    n++;
    auto delta = sample - mean;
    mean += delta / n;
  }

  void ConsumeSimd(stdx::native_simd<T> &simd, uint16_t width) {
    n += width;
    auto delta = simd - mean;
    mean += delta / n;
  }

  void merge(const Mean<T>& other) {
    if (other.n == 0) {
      return;
    }
    if (n == 0) {
      mean = other.mean;
      n = other.n;
      return;
    }

    n = n + other.n;
    auto delta = other.mean - mean;
    mean += delta * other.n / n;
  }
};

template <typename O, typename I> void tcSum(O *out, I *inp, uint64_t nel) {
  size_t width =
      std::min(stdx::native_simd<O>::size(), stdx::native_simd<I>::size());
  std::unique_ptr<IAccessor<I>> inpIt =
      std::make_unique<Accessor<I>>(inp, width, nel);

  auto red = std::reduce(
      std::execution::par, (*inpIt).countBegin(), (*inpIt).countEnd(),
      stdx::native_simd<O>(0),
      [&inpIt](uint64_t a, uint64_t b) { return a + inpIt->load(b); }
  );

  O redScalar = 0;
  for (size_t i = 0; i < width; i++) {
    redScalar += static_cast<O>(red[i]);
  }
  *out = redScalar;
}

template <typename O, typename I> void tcMean(O *out, I *inp, uint64_t nel) {
  size_t width =
      std::min(stdx::native_simd<O>::size(), stdx::native_simd<I>::size());
  std::unique_ptr<IAccessor<I>> inpIt =
      std::make_unique<Accessor<I>>(inp, width, nel);

  // TODO
}