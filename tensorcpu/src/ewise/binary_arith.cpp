#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <stdfloat>
#include <typeinfo>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename O, typename I>
void tcBinaryArith(
    O *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster
) {
  size_t width = stdx::native_simd<I>::size();
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

  Kernel kernel;

  if (op == BinaryOp::Plus) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::native_simd<I> a, b;
      o.store(i, i1.load(i, a) + i2->load(i, b));
    };
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::native_simd<I> a, b;
        o.store(i, i1.load(i, a) - i2->load(i, b));
      };
    } else {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::native_simd<I> a, b;
        o.store(i, i2->load(i, b) - i1.load(i, a));
      };
    }
  } else if (op == BinaryOp::Mul) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::native_simd<I> a, b;
      o.store(i, i1.load(i, a) * i2->load(i, b));
    };
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      kernel = [&i1, &i2, &o, &out, &inp1, &width](uint64_t i) {
        if constexpr (isRealNum<I>()) {
          stdx::native_simd<I> a, b;
          o.store(i, i1.load(i, a) / i2->load(i, b));
        } else {
          auto elements = i1.calcRemainingElements(i);
          I b[elements];
          O *oPtr = out + i * width;
          I *i1Ptr = inp1 + i * width;
          i2->load(i, b);
#pragma GCC ivdep
          for (int j = 0; j < elements; j++) {
            oPtr[j] = i1Ptr[j] / b[j];
          }
        }
      };
    } else {
      kernel = [&i1, &i2, &o, &out, &inp1, &width](uint64_t i) {
        if constexpr (isRealNum<I>()) {
          stdx::native_simd<I> a, b;
          o.store(i, i2->load(i, b) / i1.load(i, a));
        } else {
          auto elements = i1.calcRemainingElements(i);
          I b[elements];
          O *oPtr = out + i * width;
          I *i1Ptr = inp1 + i * width;
          i2->load(i, b);
#pragma GCC ivdep
          for (int j = 0; j < elements; j++) {
            oPtr[j] = b[j] / i1Ptr[j];
          }
        }
      };
    }
  } else if (op == BinaryOp::Pow) {
    if (!flip) {
      kernel = [&i1, &i2, &o, &out, &inp1, &width](uint64_t i) {
        using std::pow;
        auto elements = i1.calcRemainingElements(i);
        I b[elements];
        O *oPtr = out + i * width;
        I *i1Ptr = inp1 + i * width;
        i2->load(i, b);
#pragma GCC ivdep
        for (int j = 0; j < elements; j++) {
          oPtr[j] = pow(i1Ptr[j], b[j]);
        }
      };
    } else {
      kernel = [&i1, &i2, &o, &out, &inp1, &width](uint64_t i) {
        using std::pow;
        auto elements = i1.calcRemainingElements(i);
        I b[elements];
        O *oPtr = out + i * width;
        I *i1Ptr = inp1 + i * width;
        i2->load(i, b);
#pragma GCC ivdep
        for (int j = 0; j < elements; j++) {
          oPtr[j] = pow(b[j], i1Ptr[j]);
        }
      };
    }
  }

  std::for_each(std::execution::par, i1.countBegin(), i1.countEnd(), kernel);
}

#define BINARYARITH(O, I)                                                      \
  template void tcBinaryArith<O, I>(                                           \
      O * out, I * inp1, I * inp2, BinaryOp op, uint64_t nel, uint8_t flip,    \
      Dim2 i2broadcaster                                                       \
  );

template <typename O, typename I>
void tcBinaryArithCasted(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
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

  Kernel kernel;
  if (op == BinaryOp::Plus) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::native_simd<I> a, b;
      o.store(i, i1.load(i, a) + i2->load(i, b));
    };
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::native_simd<I> a, b;
        o.store(i, i1.load(i, a) - i2->load(i, b));
      };
    } else {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::native_simd<I> a, b;
        o.store(i, i2->load(i, b) - i1.load(i, a));
      };
    }
  } else if (op == BinaryOp::Mul) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::native_simd<I> a, b;
      o.store(i, i1.load(i, a) * i2->load(i, b));
    };
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      kernel = [&i1, &i2, &o](uint64_t i) {
        // WORKAROUND: SIMD integer division not working for some reason
        if constexpr (isRealNum<I>()) {
          stdx::native_simd<I> a, b;
          o.store(i, i1.load(i, a) / i2->load(i, b));
        } else {
          auto elements = i1.calcRemainingElements(i);
          I a[elements], b[elements];
          i1.load(i, a);
          i2->load(i, b);
          O res[elements];
#pragma GCC ivdep
          for (int j = 0; j < elements; j++) {
            res[j] = a[j] / b[j];
          }
          o.store(i, res, elements);
        }
      };
    } else {
      kernel = [&i1, &i2, &o](uint64_t i) {
        if constexpr (isRealNum<I>()) {
          stdx::native_simd<I> a, b;
          o.store(i, i2->load(i, b) / i1.load(i, a));
        } else {
          auto elements = i1.calcRemainingElements(i);
          I a[elements], b[elements];
          i1.load(i, a);
          i2->load(i, b);
          O res[elements];
#pragma GCC ivdep
          for (int j = 0; j < elements; j++) {
            res[j] = b[j] / a[j];
          }
          o.store(i, res, elements);
        }
      };
    }
  } else if (op == BinaryOp::Pow) {
    if (!flip) {
      kernel = [&i1, &i2, &o](uint64_t i) {
        using std::pow;
        auto elements = i1.calcRemainingElements(i);
        I a[elements], b[elements];
        i1.load(i, a);
        i2->load(i, b);
        O res[elements];
#pragma GCC ivdep
        for (int j = 0; j < elements; j++) {
          res[j] = pow(a[j], b[j]);
        }
        o.store(i, res, elements);
      };
    } else {
      kernel = [&i1, &i2, &o](uint64_t i) {
        using std::pow;
        auto elements = i1.calcRemainingElements(i);
        I a[elements], b[elements];
        i1.load(i, a);
        i2->load(i, b);
        O res[elements];
#pragma GCC ivdep
        for (int j = 0; j < elements; j++) {
          res[j] = pow(b[j], a[j]);
        }
        o.store(i, res, elements);
      };
    }
  }

  std::for_each(std::execution::par, i1.countBegin(), i1.countEnd(), kernel);
}

#define BINARYARITH_CASTED(O, I)                                               \
  template void tcBinaryArithCasted<O, I>(                                     \
      void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel,            \
      uint8_t flip, Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID,         \
      uint8_t i2TID                                                            \
  );

template <typename O, typename I1, typename I2>
void tcBinaryArithCastedPlain(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  DType outType = dtypes[outTID];
  DType inp1Type = dtypes[i1TID];
  DType inp2Type = dtypes[i2TID];
  const Caster<I1> &i1Caster = Caster<I1>::lookup(inp1Type);
  const Caster<O> &oCaster = Caster<O>::lookup(outType);
  std::unique_ptr<ISimd<I2>> i2;
  uint64_t snel = i2broadcaster.nel();
  const Caster<I2> &i2Caster = Caster<I2>::lookup(inp2Type);
  if (snel == 0) {
    i2 = std::make_unique<CastSimd<I2>>(i2Caster, inp2, 1, nel);
  } else if (snel == 1) {
    i2 = std::make_unique<SameSimd<I2>>(i2Caster.loader(inp2, 0), 1, nel);
  } else {
    i2 = std::make_unique<CastRwiseSimd<I2>>(
        i2Caster, inp2, 1, nel, i2broadcaster
    );
  }

  Kernel kernel;
  if (op == BinaryOp::Plus) {
    kernel = [&inp1, &i2, &out, &i1Caster, &oCaster](uint64_t i) {
      oCaster.storer(out, i, i1Caster.loader(inp1, i) + i2->get(i));
    };
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      kernel = [&inp1, &i2, &out, &i1Caster, &oCaster](uint64_t i) {
        oCaster.storer(out, i, i1Caster.loader(inp1, i) - i2->get(i));
      };
    } else {
      kernel = [&inp1, &i2, &out, &i1Caster, &oCaster](uint64_t i) {
        oCaster.storer(out, i, i2->get(i) - i1Caster.loader(inp1, i));
      };
    }
  } else if (op == BinaryOp::Mul) {
    kernel = [&inp1, &i2, &out, &i1Caster, &oCaster](uint64_t i) {
      oCaster.storer(out, i, i1Caster.loader(inp1, i) * i2->get(i));
    };
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      kernel = [&inp1, &i2, &out, &i1Caster, &oCaster](uint64_t i) {
        oCaster.storer(out, i, i1Caster.loader(inp1, i) / i2->get(i));
      };
    } else {
      kernel = [&inp1, &i2, &out, &i1Caster, &oCaster](uint64_t i) {
        oCaster.storer(out, i, i2->get(i) / i1Caster.loader(inp1, i));
      };
    }
  } else if (op == BinaryOp::Pow) {
    if (!flip) {
      kernel = [&inp1, &i2, &out, &i1Caster, &oCaster, flip](uint64_t i) {
        using std::pow;
        I1 a = i1Caster.loader(inp1, i);
        I2 b = i2->get(i);
        O res = pow(a, b);
        oCaster.storer(out, i, res);
      };
    } else {
      kernel = [&inp1, &i2, &out, &i1Caster, &oCaster, flip](uint64_t i) {
        using std::pow;
        I1 a = i1Caster.loader(inp1, i);
        I2 b = i2->get(i);
        O res = pow(b, a);
        oCaster.storer(out, i, res);
      };
    }
  }

  std::for_each(std::execution::par, Range(0), Range(nel), kernel);
}

#define BINARYARITH_CASTED_PLAIN(O, I1, I2)                                    \
  template void tcBinaryArithCastedPlain<O, I1, I2>(                           \
      void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel,            \
      uint8_t flip, Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID,         \
      uint8_t i2TID                                                            \
  );

#define UNWIND3_SAME(A, OP, NAME) OP(A, A, A, NAME)

#define UNWIND2_SAME(A, OP) OP(A, A)

#define UWIND3_ALL_TYPES(OP, NAME)                                             \
  UNWIND3_SAME(int8_t, OP, NAME)                                               \
  UNWIND3_SAME(int16_t, OP, NAME)                                              \
  UNWIND3_SAME(int32_t, OP, NAME)                                              \
  UNWIND3_SAME(int64_t, OP, NAME)                                              \
  UNWIND3_SAME(uint8_t, OP, NAME)                                              \
  UNWIND3_SAME(uint16_t, OP, NAME)                                             \
  UNWIND3_SAME(uint32_t, OP, NAME)                                             \
  UNWIND3_SAME(uint64_t, OP, NAME)                                             \
  UNWIND3_SAME(float, OP, NAME)                                                \
  UNWIND3_SAME(double, OP, NAME)

#define UNWIND2_ALL_TYPES(OP)                                                  \
  UNWIND2_SAME(int8_t, OP)                                                     \
  UNWIND2_SAME(int16_t, OP)                                                    \
  UNWIND2_SAME(int32_t, OP)                                                    \
  UNWIND2_SAME(int64_t, OP)                                                    \
  UNWIND2_SAME(uint8_t, OP)                                                    \
  UNWIND2_SAME(uint16_t, OP)                                                   \
  UNWIND2_SAME(uint32_t, OP)                                                   \
  UNWIND2_SAME(uint64_t, OP)                                                   \
  UNWIND2_SAME(float, OP)                                                      \
  UNWIND2_SAME(double, OP)                                                     \
  UNWIND2_SAME(std::float16_t, OP)                                             \
  UNWIND2_SAME(std::bfloat16_t, OP)

#define UNWIND3_2(A, B, OP, NAME)                                              \
  OP(A, A, A, NAME)                                                            \
  OP(A, A, B, NAME)                                                            \
  OP(A, B, A, NAME)                                                            \
  OP(A, B, B, NAME)                                                            \
  OP(B, B, B, NAME)                                                            \
  OP(B, A, B, NAME)                                                            \
  OP(B, B, A, NAME)                                                            \
  OP(B, A, A, NAME)

#define UNWIND2_2(A, B, OP)                                                    \
  OP(A, A)                                                                     \
  OP(A, B)                                                                     \
  OP(B, A)                                                                     \
  OP(B, B)

UNWIND2_ALL_TYPES(BINARYARITH)

UNWIND2_2(double, int64_t, BINARYARITH_CASTED)
BINARYARITH_CASTED(float, float)
BINARYARITH_CASTED(float, int16_t)
BINARYARITH_CASTED(float, int32_t)
BINARYARITH_CASTED(double, int16_t)
BINARYARITH_CASTED(double, int32_t)
BINARYARITH_CASTED(int32_t, int32_t)
BINARYARITH_CASTED(int16_t, int16_t)

/*
BINARYARITH_CASTED_PLAIN(int16_t, int16_t, int16_t)
BINARYARITH_CASTED_PLAIN(int32_t, int32_t, int32_t)
BINARYARITH_CASTED_PLAIN(int64_t, int64_t, int64_t)
 */