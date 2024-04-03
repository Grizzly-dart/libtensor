#include <algorithm>
#include <execution>
#include <stdint.h>
#include <stdlib.h>
#include <cmath>

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

#include "binary_arith_gen.inc"