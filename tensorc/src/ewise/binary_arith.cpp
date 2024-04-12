#include <algorithm>
#include <cmath>
#include <execution>

#include "tensorc.hpp"
#include "typed_array.hpp"

enum BinaryOp : uint8_t {
  Plus,
  Minus,
  Mul,
  Div,
  Pow,
};

template <typename O, typename I, BinaryOp op>
void tcBinaryArith(
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
        if constexpr (op == BinaryOp::Plus) {
          o.store(i, i1.load(i, a) + i2->load(i, b));
        } else if constexpr (op == BinaryOp::Minus) {
          if (!flip) {
            o.store(i, i1.load(i, a) - i2->load(i, b));
          } else {
            o.store(i, i2->load(i, b) - i1.load(i, a));
          }
        } else if constexpr (op == BinaryOp::Mul) {
          o.store(i, i1.load(i, a) * i2->load(i, b));
        } else if constexpr (op == BinaryOp::Div) {
          if (!flip) {
            o.store(i, i1.load(i, a) / i2->load(i, b));
          } else {
            o.store(i, i2->load(i, b) / i1.load(i, a));
          }
        } else if constexpr (op == BinaryOp::Pow) {
          auto elements = i1.calcRemainingElements(i);
          using std::pow;
          if (!flip) {
            for (int j = 0; j < elements; j++) {
              uint64_t ind = i * i1.width + j;
              o.set(ind, pow(i1.get(ind), i2->get(ind)));
            }
          } else {
            for (int j = 0; j < elements; j++) {
              uint64_t ind = i * i1.width + j;
              o.set(ind, pow(i2->get(ind), i1.get(ind)));
            }
          }
        }
      }
  );
}

#define BINARYARITH(O, I)                                                      \
  template void tcBinaryArith<O, I, BinaryOp::Plus>(                           \
      O * out, I * inp1, I * inp2, uint64_t nel, uint8_t flip,                 \
      Dim2 i2broadcaster                                                       \
  );                                                                           \
  template void tcBinaryArith<O, I, BinaryOp::Minus>(                          \
      O * out, I * inp1, I * inp2, uint64_t nel, uint8_t flip,                 \
      Dim2 i2broadcaster                                                       \
  );                                                                           \
  template void tcBinaryArith<O, I, BinaryOp::Mul>(                            \
      O * out, I * inp1, I * inp2, uint64_t nel, uint8_t flip,                 \
      Dim2 i2broadcaster                                                       \
  );                                                                           \
  template void tcBinaryArith<O, I, BinaryOp::Div>(                            \
      O * out, I * inp1, I * inp2, uint64_t nel, uint8_t flip,                 \
      Dim2 i2broadcaster                                                       \
  );                                                                           \
  template void tcBinaryArith<O, I, BinaryOp::Pow>(                            \
      O * out, I * inp1, I * inp2, uint64_t nel, uint8_t flip,                 \
      Dim2 i2broadcaster                                                       \
  );

UNWIND2_ALL_TYPES(BINARYARITH)
// BINARYARITH(double, float, Plus)
// BINARYARITH(float, double, Plus)

template <typename O, typename I, BinaryOp op>
void tcPlusSlow(
    void *out, void *inp1, void *inp2, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID
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
      [&i1, &i2, &o, flip](uint64_t i) {
        stdx::native_simd<I> a, b;
        if constexpr (op == BinaryOp::Plus) {
          o.store(i, i1.load(i, a) + i2->load(i, b));
        } else if constexpr (op == BinaryOp::Minus) {
          if (!flip) {
            o.store(i, i1.load(i, a) - i2->load(i, b));
          } else {
            o.store(i, i2->load(i, b) - i1.load(i, a));
          }
        } else if constexpr (op == BinaryOp::Mul) {
          o.store(i, i1.load(i, a) * i2->load(i, b));
        } else if constexpr (op == BinaryOp::Div) {
          if (!flip) {
            o.store(i, i1.load(i, a) / i2->load(i, b));
          } else {
            o.store(i, i2->load(i, b) / i1.load(i, a));
          }
        } else if constexpr (op == BinaryOp::Pow) {
          auto elements = i1.calcRemainingElements(i);
          using std::pow;
          if (!flip) {
            for (int j = 0; j < elements; j++) {
              uint64_t ind = i * i1.width + j;
              o.set(ind, pow(i1.get(ind), i2->get(ind)));
            }
          } else {
            for (int j = 0; j < elements; j++) {
              uint64_t ind = i * i1.width + j;
              o.set(ind, pow(i2->get(ind), i1.get(ind)));
            }
          }
        }
      }
  );
}

#define BINARYARITH_SLOW(O, I)                                                 \
  template void tcPlusSlow<O, I, BinaryOp::Plus>(                              \
      void *out, void *inp1, void *inp2, uint64_t nel, uint8_t flip,           \
      Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID         \
  );                                                                           \
  template void tcPlusSlow<O, I, BinaryOp::Minus>(                             \
      void *out, void *inp1, void *inp2, uint64_t nel, uint8_t flip,           \
      Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID         \
  );                                                                           \
  template void tcPlusSlow<O, I, BinaryOp::Mul>(                               \
      void *out, void *inp1, void *inp2, uint64_t nel, uint8_t flip,           \
      Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID         \
  );                                                                           \
  template void tcPlusSlow<O, I, BinaryOp::Div>(                               \
      void *out, void *inp1, void *inp2, uint64_t nel, uint8_t flip,           \
      Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID         \
  );                                                                           \
  template void tcPlusSlow<O, I, BinaryOp::Pow>(                               \
      void *out, void *inp1, void *inp2, uint64_t nel, uint8_t flip,           \
      Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID         \
  );

UNWIND2_2(double, int64_t, BINARYARITH_SLOW)