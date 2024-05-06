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

template <typename T>
using CastLoader1 = void (*)(T &out, const void *inp, uint64_t offset);

template <typename T> CastLoader1<T> castedLoader(const DType &type) {
  if (type.index == f32.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((float *)inp)[offset]);
    };
  } else if (type.index == f64.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((double *)inp)[offset]);
    };
  } else if (type.index == i8.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int8_t *)inp)[offset]);
    };
  } else if (type.index == i16.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int16_t *)inp)[offset]);
    };
  } else if (type.index == i32.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int32_t *)inp)[offset]);
    };
  } else if (type.index == i64.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((int64_t *)inp)[offset]);
    };
  } else if (type.index == u8.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint8_t *)inp)[offset]);
    };
  } else if (type.index == u16.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint16_t *)inp)[offset]);
    };
  } else if (type.index == u32.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint32_t *)inp)[offset]);
    };
  } else if (type.index == u64.index) {
    return [](T &out, const void *inp, uint64_t offset) {
      out = T(((uint64_t *)inp)[offset]);
    };
  }
}

template <typename T>
using CastStorer1 = void (*)(void *inp, T &out, uint64_t offset);

template <typename T> CastStorer<T> castedStorer(const DType &type) {
  if (type.index == f32.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((float *)out)[offset] = inp;
    };
  } else if (type.index == f64.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((double *)out)[offset] = inp;
    };
  } else if (type.index == i8.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int8_t *)out)[offset] = inp;
    };
  } else if (type.index == i16.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int16_t *)out)[offset] = inp;
    };
  } else if (type.index == i32.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int32_t *)out)[offset] = inp;
    };
  } else if (type.index == i64.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((int64_t *)out)[offset] = inp;
    };
  } else if (type.index == u8.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint8_t *)out)[offset] = inp;
    };
  } else if (type.index == u16.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint16_t *)out)[offset] = inp;
    };
  } else if (type.index == u32.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint32_t *)out)[offset] = inp;
    };
  } else if (type.index == u64.index) {
    return [](void *out, T &inp, uint64_t offset) {
      ((uint64_t *)out)[offset] = inp;
    };
  }
}

template <typename T, uint16_t laneSize>
using CastSimdLoader1 = void (*)(
    T __attribute__((vector_size(sizeof(T) * laneSize))) & out, const void *inp,
    uint64_t offset
);

template <typename T, uint16_t laneSize>
CastSimdLoader1<T, laneSize> castedVectorStore(const DType &type) {
  if (type.index == f32.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      float __attribute__((vector_size(sizeof(float) * laneSize))) tmp;
      memcpy(&tmp, ((float *)inp) + offset, sizeof(float) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == f64.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      double __attribute__((vector_size(sizeof(double) * laneSize))) tmp;
      memcpy(&tmp, ((double *)inp) + offset, sizeof(double) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == i8.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      int8_t __attribute__((vector_size(sizeof(int8_t) * laneSize))) tmp;
      memcpy(&tmp, ((int8_t *)inp) + offset, sizeof(int8_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == i16.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      int16_t __attribute__((vector_size(sizeof(int16_t) * laneSize))) tmp;
      memcpy(&tmp, ((int16_t *)inp) + offset, sizeof(int16_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == i32.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      int32_t __attribute__((vector_size(sizeof(int32_t) * laneSize))) tmp;
      memcpy(&tmp, ((int32_t *)inp) + offset, sizeof(int32_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == i64.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      int64_t __attribute__((vector_size(sizeof(int64_t) * laneSize))) tmp;
      memcpy(&tmp, ((int64_t *)inp) + offset, sizeof(int64_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == u8.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      uint8_t __attribute__((vector_size(sizeof(uint8_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint8_t *)inp) + offset, sizeof(uint8_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == u16.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      uint16_t __attribute__((vector_size(sizeof(uint16_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint16_t *)inp) + offset, sizeof(uint16_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == u32.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      uint32_t __attribute__((vector_size(sizeof(uint32_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint32_t *)inp) + offset, sizeof(uint32_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  } else if (type.index == u64.index) {
    return [](T __attribute__((vector_size(sizeof(T) * laneSize))) & out,
              const void *inp, uint64_t offset) {
      uint64_t __attribute__((vector_size(sizeof(uint64_t) * laneSize))) tmp;
      memcpy(&tmp, ((uint64_t *)inp) + offset, sizeof(uint64_t) * laneSize);
      out = __builtin_convertvector(tmp, T);
    };
  }
}

template <typename T, uint16_t laneSize>
using CastSimdStorer1 = void (*)(
    void *out, T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
    uint64_t offset
);

template <typename T, uint16_t laneSize>
CastSimdStorer1<T, laneSize> castedVectorStore(const DType &type) {
  if (type.index == f32.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      float __attribute__((vector_size(sizeof(float) * laneSize))) tmp =
          __builtin_convertvector(
              inp, float __attribute__((vector_size(sizeof(float) * laneSize)))
          );
      memcpy(((float *)out) + offset, &tmp, sizeof(float) * laneSize);
    };
  } else if (type.index == f64.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      double __attribute__((vector_size(sizeof(double) * laneSize)))
      tmp = __builtin_convertvector(
          inp, double __attribute__((vector_size(sizeof(double) * laneSize)))
      );
      memcpy(((double *)out) + offset, &tmp, sizeof(double) * laneSize);
    };
  } else if (type.index == i8.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int8_t __attribute__((vector_size(sizeof(int8_t) * laneSize)))
      tmp = __builtin_convertvector(
          inp, int8_t __attribute__((vector_size(sizeof(int8_t) * laneSize)))
      );
      memcpy(((int8_t *)out) + offset, &tmp, sizeof(int8_t) * laneSize);
    };
  } else if (type.index == i16.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int16_t __attribute__((vector_size(sizeof(int16_t) * laneSize)))
      tmp = __builtin_convertvector(
          inp, int16_t __attribute__((vector_size(sizeof(int16_t) * laneSize)))
      );
      memcpy(((int16_t *)out) + offset, &tmp, sizeof(int16_t) * laneSize);
    };
  } else if (type.index == i32.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int32_t __attribute__((vector_size(sizeof(int32_t) * laneSize)))
      tmp = __builtin_convertvector(
          inp, int32_t __attribute__((vector_size(sizeof(int32_t) * laneSize)))
      );
      memcpy(((int32_t *)out) + offset, &tmp, sizeof(int32_t) * laneSize);
    };
  } else if (type.index == i64.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      int64_t __attribute__((vector_size(sizeof(int64_t) * laneSize)))
      tmp = __builtin_convertvector(
          inp, int64_t __attribute__((vector_size(sizeof(int64_t) * laneSize)))
      );
      memcpy(((int64_t *)out) + offset, &tmp, sizeof(int64_t) * laneSize);
    };
  } else if (type.index == u8.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint8_t __attribute__((vector_size(sizeof(uint8_t) * laneSize)))
      tmp = __builtin_convertvector(
          inp, uint8_t __attribute__((vector_size(sizeof(uint8_t) * laneSize)))
      );
      memcpy(((uint8_t *)out) + offset, &tmp, sizeof(uint8_t) * laneSize);
    };
  } else if (type.index == u16.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint16_t __attribute__((vector_size(sizeof(uint16_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              uint16_t __attribute__((vector_size(sizeof(uint16_t) * laneSize)))
          );
      memcpy(((uint16_t *)out) + offset, &tmp, sizeof(uint16_t) * laneSize);
    };
  } else if (type.index == u32.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint32_t __attribute__((vector_size(sizeof(uint32_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              uint32_t __attribute__((vector_size(sizeof(uint32_t) * laneSize)))
          );
      memcpy(((uint32_t *)out) + offset, &tmp, sizeof(uint32_t) * laneSize);
    };
  } else if (type.index == u64.index) {
    return [](void *out,
              T __attribute__((vector_size(sizeof(T) * laneSize))) & inp,
              uint64_t offset) {
      uint64_t __attribute__((vector_size(sizeof(uint64_t) * laneSize))) tmp =
          __builtin_convertvector(
              inp,
              uint64_t __attribute__((vector_size(sizeof(uint64_t) * laneSize)))
          );
      memcpy(((uint64_t *)out) + offset, &tmp, sizeof(uint64_t) * laneSize);
    };
  }
}

template <typename I>
void binaryarith_casted_1thread(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  const auto &i1Loader = castedLoader<I>(dtypes[i1TID]);
  const auto &i2Loader = castedLoader<I>(dtypes[i2TID]);
  const auto &oStorer = castedStorer<I>(dtypes[outTID]);

  if (op == BinaryOp::Plus) {
    for (uint64_t i = 0; i < nel; i++) {
      I a, b;
      i1Loader(a, inp1, i);
      i2Loader(b, inp2, i);
      I res = a + b;
      oStorer(out, res, i);
    }
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      for (uint64_t i = 0; i < nel; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = a - b;
        oStorer(out, res, i);
      }
    } else {
      for (uint64_t i = 0; i < nel; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = b - a;
        oStorer(out, res, i);
      }
    }
  } else if (op == BinaryOp::Mul) {
    for (uint64_t i = 0; i < nel; i++) {
      I a, b;
      i1Loader(a, inp1, i);
      i2Loader(b, inp2, i);
      I res = a * b;
      oStorer(out, res, i);
    }
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      for (uint64_t i = 0; i < nel; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = a / b;
        oStorer(out, res, i);
      }
    } else {
      for (uint64_t i = 0; i < nel; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = b / a;
        oStorer(out, res, i);
      }
    }
  } else if (op == BinaryOp::Pow) {
    if (!flip) {
      for (uint64_t i = 0; i < nel; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = std::pow(a, b);
        oStorer(out, res, i);
      }
    } else {
      for (uint64_t i = 0; i < nel; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = std::pow(b, a);
        oStorer(out, res, i);
      }
    }
  }
}

template <typename I>
void binaryarith_casted_parallel(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster, uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  constexpr uint64_t laneSize = simdSize<I>();
  typedef I SimdType __attribute__((vector_size(sizeof(I) * laneSize)));

  const auto &i1Loader = castedVectorStore<I, laneSize>(dtypes[i1TID]);
  const auto &i2Loader = castedVectorStore<I, laneSize>(dtypes[i2TID]);
  const auto &oStorer = castedVectorStore<I, laneSize>(dtypes[outTID]);

  using KernelType = std::function<void(uint64_t, uint64_t)>;
  KernelType kernel;
  if (op == BinaryOp::Plus) {
    kernel = [out, inp1, inp2, i1Loader, i2Loader,
              oStorer](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i += laneSize) {
        SimdType a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        SimdType res = a + b;
        oStorer(out, res, i);
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