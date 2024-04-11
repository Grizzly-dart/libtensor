#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>

#include "tensorc.hpp"
#include "typed_array.hpp"

template <typename O, typename I1, typename I2>
void tcPlus(
    O *out, I1 *inp1, I2 *inp2, uint64_t nel, uint8_t flip, Dim2 i2broadcaster
) {
  size_t width = std::min(
      std::min(stdx::native_simd<O>::size(), stdx::native_simd<I1>::size()),
      stdx::native_simd<I2>::size()
  );
  printf("width: %zu\n", width);
  std::unique_ptr<SimdIter<I1>> in1It =
      std::make_unique<SimdIterator<I1>>(inp1, width, nel);
  std::unique_ptr<SimdIter<I2>> in2It;
  uint64_t snel = i2broadcaster.nel();
  if (snel == 0) {
    in2It = std::make_unique<SimdIterator<I2>>(inp2, width, nel);
  } else if (snel == 1) {
    in2It = std::make_unique<ScalarSimdInpIter<I2>>(
        ScalarSimdInpIter<I2>(5, width, nel)
    );
  } else {
    in2It = std::make_unique<RwiseSimdIterator<I2>>(inp2, width, size);
  }
  auto outIt = SimdIterator<O>(out, width, nel);

  std::for_each(
      std::execution::par, (*in1It).countBegin(), (*in1It).countEnd(),
      [&in1It, &in2It, &outIt, flip](uint64_t a) {
        if (flip == 0) {
          outIt[a] = in1It->at(a) + in2It->at(a);
        } else {
          outIt[a] = in2It->at(a) - in1It->at(a);
        }
      }
  );
}

/* TODO
template <typename O, typename I1, typename I2>
void tcPlusSlow(
    O *out, I1 *inp1, I2 *inp2, uint64_t nel, uint8_t flip, Dim2 i2broadcaster,
    DType outType, DType inp1Type, DType inp2Type
) {
  CastingSimdIterator
  // TODO
}
*/

template <typename O, typename I1, typename I2>
const char *tcPlus(
    O *out, const I1 *inp1, const I2 *inp2, const I2 *scalar, uint64_t nel,
    uint8_t flip
) {
  if ((inp2 == nullptr) == (scalar == nullptr)) {
    return "Both inp2 and scalar cannot be null or non-null at the same time";
  }
  if (inp2 != nullptr) {
    std::transform(
        std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
        [](I1 a, I2 b) { return a + b; }
    );
  } else {
    std::transform(
        std::execution::par_unseq, inp1, inp1 + nel, out,
        [scalar](I1 a) { return a + *scalar; }
    );
  }
  return nullptr;
}

template <typename O, typename I1, typename I2>
const char *tcMinus(
    O *out, const I1 *inp1, const I2 *inp2, const I2 *scalar, uint64_t nel,
    uint8_t flip
) {
  if ((inp2 == nullptr) == (scalar == nullptr)) {
    return "Both inp2 and scalar cannot be null or non-null at the same time";
  }
  if (inp2 != nullptr) {
    if (!flip) {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
          [](I1 a, I2 b) { return a + b; }
      );
    } else {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
          [](I1 a, I2 b) { return b - a; }
      );
    }
  } else {
    if (!flip) {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, out,
          [scalar](I1 a) { return a - *scalar; }
      );
    } else {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, out,
          [scalar](I1 a) { return *scalar - a; }
      );
    }
  }
  return nullptr;
}

template <typename O, typename I1, typename I2>
const char *tcMul(
    O *out, const I1 *inp1, const I2 *inp2, const I2 *scalar, uint64_t nel,
    uint8_t flip
) {
  if ((inp2 == nullptr) == (scalar == nullptr)) {
    return "Both inp2 and scalar cannot be null or non-null at the same time";
  }
  if (inp2 != nullptr) {
    std::transform(
        std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
        [](I1 a, I2 b) { return a * b; }
    );
  } else {
    std::transform(
        std::execution::par_unseq, inp1, inp1 + nel, out,
        [scalar](I1 a) { return a * *scalar; }
    );
  }
  return nullptr;
}

template <typename O, typename I1, typename I2>
const char *tcDiv(
    O *out, const I1 *inp1, const I2 *inp2, const I2 *scalar, uint64_t nel,
    uint8_t flip
) {
  if ((inp2 == nullptr) == (scalar == nullptr)) {
    return "Both inp2 and scalar cannot be null or non-null at the same time";
  }
  if (inp2 != nullptr) {
    if (!flip) {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
          [](I1 a, I2 b) { return a / b; }
      );
    } else {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
          [](I1 a, I2 b) { return b / a; }
      );
    }
  } else {
    if (!flip) {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, out,
          [scalar](I1 a) { return a / *scalar; }
      );
    } else {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, out,
          [scalar](I1 a) { return *scalar / a; }
      );
    }
  }
  return nullptr;
}

template <typename O, typename I1, typename I2>
const char *tcPow(
    O *out, const I1 *inp1, const I2 *inp2, const I2 *scalar, uint64_t nel,
    uint8_t flip
) {
  if ((inp2 == nullptr) == (scalar == nullptr)) {
    return "Both inp2 and scalar cannot be null or non-null at the same time";
  }
  if (inp2 != nullptr) {
    if (!flip) {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
          [](I1 a, I2 b) { return std::pow(a, b); }
      );
    } else {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
          [](I1 a, I2 b) { return std::pow(b, a); }
      );
    }
  } else {
    if (!flip) {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, out,
          [scalar](I1 a) { return std::pow(a, *scalar); }
      );
    } else {
      std::transform(
          std::execution::par_unseq, inp1, inp1 + nel, out,
          [scalar](I1 a) { return std::pow(*scalar, a); }
      );
    }
  }
  return nullptr;
}

// #include "binary_arith_gen.inc"