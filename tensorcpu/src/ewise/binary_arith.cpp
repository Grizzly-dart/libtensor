#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <stdfloat>
#include <typeinfo>

#include "macro_unwind.hpp"
#include "reducer.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename I>
void binaryarith_1thread(
    I *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster
) {
  if (op == BinaryOp::Plus) {
#pragma GCC ivdep
    for (uint64_t i = 0; i < nel; i++) {
      out[i] = inp1[i] + inp2[i];
    }
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
#pragma GCC ivdep
      for (uint64_t i = 0; i < nel; i++) {
        out[i] = inp1[i] - inp2[i];
      }
    } else {
#pragma GCC ivdep
      for (uint64_t i = 0; i < nel; i++) {
        out[i] = inp2[i] - inp1[i];
      }
    }
  } else if (op == BinaryOp::Mul) {
#pragma GCC ivdep
    for (uint64_t i = 0; i < nel; i++) {
      out[i] = inp1[i] * inp2[i];
    }
  } else if (op == BinaryOp::Div) {
    if (!flip) {
#pragma GCC ivdep
      for (uint64_t i = 0; i < nel; i++) {
        out[i] = inp1[i] / inp2[i];
      }
    } else {
#pragma GCC ivdep
      for (uint64_t i = 0; i < nel; i++) {
        out[i] = inp2[i] / inp1[i];
      }
    }
  } else if (op == BinaryOp::Pow) {
    if (!flip) {
#pragma GCC ivdep
      for (uint64_t i = 0; i < nel; i++) {
        out[i] = std::pow(inp1[i], inp2[i]);
      }
    } else {
#pragma GCC ivdep
      for (uint64_t i = 0; i < nel; i++) {
        out[i] = std::pow(inp2[i], inp1[i]);
      }
    }
  }
}

#define BINARYARITH1THREAD(I)                                                  \
  template void binaryarith_1thread<I>(                                        \
      I * out, I * inp1, I * inp2, BinaryOp op, uint64_t nel, uint8_t flip,    \
      Dim2 i2broadcaster                                                       \
  );

UNWIND1_ALL_TYPES(BINARYARITH1THREAD)

template <typename I>
void binaryarith_parallel(
    I *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster
) {
  constexpr uint64_t laneSize = simdSize<I>();
  typedef I SimdType __attribute__((vector_size(sizeof(I) * laneSize)));
  using KernelType = std::function<void(uint64_t, uint64_t)>;
  KernelType kernel;
  if (op == BinaryOp::Plus) {
    kernel = [&](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i += laneSize) {
        SimdType a, b;
        memcpy(&a, inp1 + i, sizeof(SimdType));
        memcpy(&b, inp2 + i, sizeof(SimdType));
        SimdType res = a + b;
        memcpy(out + i, &res, sizeof(SimdType));
      }
    };
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      kernel = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          memcpy(&a, inp1 + i, sizeof(SimdType));
          memcpy(&b, inp2 + i, sizeof(SimdType));
          SimdType res = a - b;
          memcpy(out + i, &res, sizeof(SimdType));
        }
      };
    } else {
      kernel = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          memcpy(&a, inp1 + i, sizeof(SimdType));
          memcpy(&b, inp2 + i, sizeof(SimdType));
          SimdType res = b - a;
          memcpy(out + i, &res, sizeof(SimdType));
        }
      };
    }
  } else if (op == BinaryOp::Mul) {
    kernel = [&](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i += laneSize) {
        SimdType a, b;
        memcpy(&a, inp1 + i, sizeof(SimdType));
        memcpy(&b, inp2 + i, sizeof(SimdType));
        SimdType res = a * b;
        memcpy(out + i, &res, sizeof(SimdType));
      }
    };
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      kernel = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          memcpy(&a, inp1 + i, sizeof(SimdType));
          memcpy(&b, inp2 + i, sizeof(SimdType));
          SimdType res = a / b;
          memcpy(out + i, &res, sizeof(SimdType));
        }
      };
    } else {
      kernel = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          memcpy(&a, inp1 + i, sizeof(SimdType));
          memcpy(&b, inp2 + i, sizeof(SimdType));
          SimdType res = b / a;
          memcpy(out + i, &res, sizeof(SimdType));
        }
      };
    }
  } else if (op == BinaryOp::Pow) {
    if (!flip) {
      kernel = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          memcpy(&a, inp1 + i, sizeof(SimdType));
          memcpy(&b, inp2 + i, sizeof(SimdType));
          SimdType res;
          for (uint64_t j = 0; j < laneSize; j++) {
            res[j] = std::pow(a[j], b[j]);
          }
          memcpy(out + i, &res, sizeof(SimdType));
        }
      };
    } else {
      kernel = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          memcpy(&a, inp1 + i, sizeof(SimdType));
          memcpy(&b, inp2 + i, sizeof(SimdType));
          SimdType res;
          for (uint64_t j = 0; j < laneSize; j++) {
            res[j] = std::pow(b[j], a[j]);
          }
          memcpy(out + i, &res, sizeof(SimdType));
        }
      };
    }
  }

  parallelSimdTransform(nel, laneSize, kernel);

  uint64_t tail = nel % laneSize;
  uint64_t offset = nel - tail;
  binaryarith_1thread(
      out + offset, inp1 + offset, inp2 + offset, op, tail, flip, i2broadcaster
  );
}

#define BINARYARITHPARALLEL(I)                                                 \
  template void binaryarith_parallel<I>(                                       \
      I * out, I * inp1, I * inp2, BinaryOp op, uint64_t nel, uint8_t flip,    \
      Dim2 i2broadcaster                                                       \
  );

UNWIND1_ALL_TYPES(BINARYARITHPARALLEL)

template<typename T, uint16_t laneSize>
[[gnu::always_inline]]
void castedVectorLoad(T __attribute__((vector_size(sizeof(T) * laneSize))) &out, const void *inp, uint64_t offset, DType type) {
  memcpy(&out, inp, sizeof(T) * laneSize);
}

template<typename T, uint16_t laneSize>
[[gnu::always_inline]]
void castedVectorStore(void *inp, T __attribute__((vector_size(sizeof(T) * laneSize))) &out) {
  memcpy(inp, &out, sizeof(T) * laneSize);
}

template <typename O, typename I>
void binaryarith_casted_parallel(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  constexpr uint64_t laneSize = std::min(simdSize<O>(), simdSize<I>());
  typedef I SimdType __attribute__((vector_size(sizeof(I) * laneSize)));
  typedef O SimdType __attribute__((vector_size(sizeof(O) * laneSize)));
  using KernelType = std::function<void(uint64_t, uint64_t)>;
  KernelType kernel;
  if (op == BinaryOp::Plus) {
    kernel = [&](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i += laneSize) {
        SimdType a, b;
        memcpy(&a, inp1 + i, sizeof(SimdType));
        memcpy(&b, inp2 + i, sizeof(SimdType));
        SimdType res = a + b;
        memcpy(out + i, &res, sizeof(SimdType));
      }
    };
  }

  // TODO
}

template <typename O, typename I>
void tcBinaryArith(
    O *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster
) {
  constexpr size_t width = simdSize<I>();
  auto i1 = Accessor<I>(inp1, width, nel);
  std::unique_ptr<IAccessor<I>> i2;
  uint64_t snel = i2broadcaster.nel();
  if (snel == 0) {
    i2 = std::make_unique<Accessor<I>>(inp2, width, nel);
  } else if (snel == 1) {
    i2 = std::make_unique<SameAccessor<I>>(SameAccessor<I>(*inp2, width, nel));
  } else {
    i2 = std::make_unique<RwiseAccessor<I>>(inp2, width, nel, i2broadcaster);
  }
  auto o = Accessor<O>(out, width, nel);

  Kernel kernel;

  if (op == BinaryOp::Plus) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::fixed_size_simd<I, width> a, b;
      o.store(i, i1.load(i, a) + i2->load(i, b));
    };
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::fixed_size_simd<I, width> a, b;
        o.store(i, i1.load(i, a) - i2->load(i, b));
      };
    } else {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::fixed_size_simd<I, width> a, b;
        o.store(i, i2->load(i, b) - i1.load(i, a));
      };
    }
  } else if (op == BinaryOp::Mul) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::fixed_size_simd<I, width> a, b;
      o.store(i, i1.load(i, a) * i2->load(i, b));
    };
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      kernel = [&i1, &i2, &o, &out, &inp1, &width](uint64_t i) {
        if constexpr (std::is_floating_point<I>::value) {
          stdx::fixed_size_simd<I, width> a, b;
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
        if constexpr (std::is_floating_point<I>::value) {
          stdx::fixed_size_simd<I, width> a, b;
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
  constexpr size_t width = simdSize<I>();
  DType outType = dtypes[outTID];
  DType inp1Type = dtypes[i1TID];
  DType inp2Type = dtypes[i2TID];
  auto i1 = CastAccessor<I>(Caster<I>::lookup(inp1Type), inp1, width, nel);
  auto o = CastAccessor<O>(Caster<O>::lookup(outType), out, width, nel);
  std::unique_ptr<IAccessor<I>> i2;
  uint64_t snel = i2broadcaster.nel();
  const Caster<I> &i2Caster = Caster<I>::lookup(inp2Type);
  if (snel == 0) {
    i2 = std::make_unique<CastAccessor<I>>(i2Caster, inp2, width, nel);
  } else if (snel == 1) {
    i2 =
        std::make_unique<SameAccessor<I>>(i2Caster.loader(inp2, 0), width, nel);
  } else {
    i2 = std::make_unique<CastRwiseAccessor<I>>(
        i2Caster, inp2, width, nel, i2broadcaster
    );
  }

  Kernel kernel;
  if (op == BinaryOp::Plus) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::fixed_size_simd<I, width> a, b;
      o.store(i, i1.load(i, a) + i2->load(i, b));
    };
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::fixed_size_simd<I, width> a, b;
        o.store(i, i1.load(i, a) - i2->load(i, b));
      };
    } else {
      kernel = [&i1, &i2, &o](uint64_t i) {
        stdx::fixed_size_simd<I, width> a, b;
        o.store(i, i2->load(i, b) - i1.load(i, a));
      };
    }
  } else if (op == BinaryOp::Mul) {
    kernel = [&i1, &i2, &o](uint64_t i) {
      stdx::fixed_size_simd<I, width> a, b;
      o.store(i, i1.load(i, a) * i2->load(i, b));
    };
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      kernel = [&i1, &i2, &o](uint64_t i) {
        // WORKAROUND: SIMD integer division not working for some reason
        if constexpr (std::is_floating_point<I>::value) {
          stdx::fixed_size_simd<I, width> a, b;
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
        if constexpr (std::is_floating_point<I>::value) {
          stdx::fixed_size_simd<I, width> a, b;
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
  std::unique_ptr<IAccessor<I2>> i2;
  uint64_t snel = i2broadcaster.nel();
  const Caster<I2> &i2Caster = Caster<I2>::lookup(inp2Type);
  if (snel == 0) {
    i2 = std::make_unique<CastAccessor<I2>>(i2Caster, inp2, 1, nel);
  } else if (snel == 1) {
    i2 = std::make_unique<SameAccessor<I2>>(i2Caster.loader(inp2, 0), 1, nel);
  } else {
    i2 = std::make_unique<CastRwiseAccessor<I2>>(
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

#define UNWIND2_SAME(A, OP) OP(A, A)

UNWIND2_ALL_TYPES(BINARYARITH)

UNWIND2_2(BINARYARITH_CASTED, double, int64_t)

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