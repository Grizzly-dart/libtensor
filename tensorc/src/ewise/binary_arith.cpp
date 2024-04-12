#include <algorithm>
#include <cmath>
#include <execution>

#include "tensorc.hpp"
#include "typed_array.hpp"

template <typename O, typename I>
void tcPlus(
    O *out, I *inp1, I *inp2, uint64_t nel, uint8_t flip, Dim2 i2broadcaster
) {
  size_t width = std::min(
      std::min(stdx::native_simd<O>::size(), stdx::native_simd<I>::size()),
      stdx::native_simd<I>::size()
  );
  printf("width: %zu\n", width);
  auto i1 = Simd<I>(inp1, width, nel);
  std::unique_ptr<ISimd<I>> i2;
  uint64_t snel = i2broadcaster.nel();
  if (snel == 0) {
    i2 = std::make_unique<Simd<I>>(inp2, width, nel);
  } else if (snel == 1) {
    i2 = std::make_unique<SameSimd<I>>(SameSimd<I>(*inp2, width, nel));
  } else {
    i2 = std::make_unique<RwiseSimd<I>>(inp2, width, nel, i2broadcaster);
  }
  auto o = Simd<O>(out, width, nel);

  std::for_each(
      std::execution::par, i1.countBegin(), i1.countEnd(),
      [&i1, &i2, &o, flip](uint64_t i) {
        stdx::native_simd<I> a, b;
        o.store(i, i1.load(i, a) + i2->load(i, b));
      }
  );
}

#define BINARYARITH(O, I, NAME)                                                \
  template void tc##NAME(                                                      \
      O *out, I *inp1, I *inp2, uint64_t nel, uint8_t flip, Dim2 i2broadcaster \
  );

UNWIND2_ALL_TYPES(BINARYARITH, Plus)
BINARYARITH(double, float, Plus)
BINARYARITH(float, double, Plus)

template <typename O, typename I>
void tcPlusSlow(
    O *out, I *inp1, I *inp2, uint64_t nel, uint8_t flip, Dim2 i2broadcaster,
    uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  size_t width = stdx::native_simd<I>::size();
  printf("width: %zu\n", width);
  DType outType = dtypes[outTID];
  DType inp1Type = dtypes[i1TID];
  DType inp2Type = dtypes[i2TID];
  auto i1 = CastSimd<I>(Caster<I>::lookup(inp1Type), inp1, width, nel);
  auto o = CastSimd<O>(Caster<O>::lookup(outType), out, width, nel);
  std::unique_ptr<ISimd<I>> i2;
  uint64_t snel = i2broadcaster.nel();
  const Caster<I> &i2Caster = Caster<I>::lookup(inp2Type);
  if (snel == 0) {
    i2 = std::make_unique<CastSimd<I>>(i2Caster, inp2, width, nel);
  } else if (snel == 1) {
    i2 = std::make_unique<SameSimd<I>>(i2Caster.loader(inp2, 0), width, nel);
  } else {
    i2 = std::make_unique<CastRwiseSimd<I>>(
        i2Caster, inp2, width, nel, i2broadcaster
    );
  }
  std::for_each(
      std::execution::par, i1.countBegin(), i1.countEnd(),
      [&i1, &i2, &o](uint64_t i) {
        stdx::native_simd<I> a, b;
        o.store(i, i1.load(i, a) + i2->load(i, b));
      }
  );
}

#define BINARYARITH_SLOW(O, I, NAME)                                           \
  template void tc##NAME##Slow(                                                \
      O *out, I *inp1, I *inp2, uint64_t nel, uint8_t flip,                    \
      Dim2 i2broadcaster, uint8_t outType, uint8_t inp1Type, uint8_t inp2Type  \
  );

BINARYARITH_SLOW(float, float, Plus)
UNWIND2_2(double, int64_t, BINARYARITH_SLOW, Plus)

/*
template <typename O, typename I1, typename I2>
const char *tcPlus(
    O *out, const I1 *inp1, const I2 *inp2, const I2 *scalar, uint64_t nel,
    uint8_t flip
) {
  if ((inp2 == nullptr) == (scalar == nullptr)) {
    return "Both inp2 and scalar cannot be null or non-null load the same time";
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
    return "Both inp2 and scalar cannot be null or non-null load the same time";
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
    return "Both inp2 and scalar cannot be null or non-null load the same time";
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
    return "Both inp2 and scalar cannot be null or non-null load the same time";
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
    return "Both inp2 and scalar cannot be null or non-null load the same time";
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
 */

// #include "binary_arith_gen.inc"