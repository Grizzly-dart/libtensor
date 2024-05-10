#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <stdfloat>
#include <typeinfo>

#include "binaryarith.hpp"
#include "macro_unwind.hpp"
#include "reducer.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

const char *tcBinaryArith(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    uint8_t oTID, uint8_t i1TID, uint8_t i2TID, uint8_t cTID
) {
  const uint64_t nelPar = 1000;
  if (oTID == i1TID && oTID == i2TID) {
    if (oTID == f32.index) {
      if (nel > nelPar) {
        binaryarith_parallel<float>(
            static_cast<float *>(out), static_cast<float *>(inp1),
            static_cast<float *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<float>(
            static_cast<float *>(out), static_cast<float *>(inp1),
            static_cast<float *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == f64.index) {
      if (nel > nelPar) {
        binaryarith_parallel<double>(
            static_cast<double *>(out), static_cast<double *>(inp1),
            static_cast<double *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<double>(
            static_cast<double *>(out), static_cast<double *>(inp1),
            static_cast<double *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == i8.index) {
      if (nel > nelPar) {
        binaryarith_parallel<int8_t>(
            static_cast<int8_t *>(out), static_cast<int8_t *>(inp1),
            static_cast<int8_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<int8_t>(
            static_cast<int8_t *>(out), static_cast<int8_t *>(inp1),
            static_cast<int8_t *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == i16.index) {
      if (nel > nelPar) {
        binaryarith_parallel<int16_t>(
            static_cast<int16_t *>(out), static_cast<int16_t *>(inp1),
            static_cast<int16_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<int16_t>(
            static_cast<int16_t *>(out), static_cast<int16_t *>(inp1),
            static_cast<int16_t *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == i32.index) {
      if (nel > nelPar) {
        binaryarith_parallel<int32_t>(
            static_cast<int32_t *>(out), static_cast<int32_t *>(inp1),
            static_cast<int32_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<int32_t>(
            static_cast<int32_t *>(out), static_cast<int32_t *>(inp1),
            static_cast<int32_t *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == i64.index) {
      if (nel > nelPar) {
        binaryarith_parallel<int64_t>(
            static_cast<int64_t *>(out), static_cast<int64_t *>(inp1),
            static_cast<int64_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<int64_t>(
            static_cast<int64_t *>(out), static_cast<int64_t *>(inp1),
            static_cast<int64_t *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == u8.index) {
      if (nel > nelPar) {
        binaryarith_parallel<uint8_t>(
            static_cast<uint8_t *>(out), static_cast<uint8_t *>(inp1),
            static_cast<uint8_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<uint8_t>(
            static_cast<uint8_t *>(out), static_cast<uint8_t *>(inp1),
            static_cast<uint8_t *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == u16.index) {
      if (nel > nelPar) {
        binaryarith_parallel<uint16_t>(
            static_cast<uint16_t *>(out), static_cast<uint16_t *>(inp1),
            static_cast<uint16_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<uint16_t>(
            static_cast<uint16_t *>(out), static_cast<uint16_t *>(inp1),
            static_cast<uint16_t *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == u32.index) {
      if (nel > nelPar) {
        binaryarith_parallel<uint32_t>(
            static_cast<uint32_t *>(out), static_cast<uint32_t *>(inp1),
            static_cast<uint32_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<uint32_t>(
            static_cast<uint32_t *>(out), static_cast<uint32_t *>(inp1),
            static_cast<uint32_t *>(inp2), op, nel, flip
        );
      }
    } else if (oTID == u64.index) {
      if (nel > nelPar) {
        binaryarith_parallel<uint64_t>(
            static_cast<uint64_t *>(out), static_cast<uint64_t *>(inp1),
            static_cast<uint64_t *>(inp2), op, nel, flip
        );
      } else {
        binaryarith_1thread<uint64_t>(
            static_cast<uint64_t *>(out), static_cast<uint64_t *>(inp1),
            static_cast<uint64_t *>(inp2), op, nel, flip
        );
      }
    } else {
      return "Unsupported type";
    }
  } else {
    if (cTID == f32.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<float>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<float>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == f64.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<double>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<double>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == i8.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<int8_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<int8_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == i16.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<int16_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<int16_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == i32.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<int32_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<int32_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == i64.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<int64_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<int64_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == u8.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<uint8_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<uint8_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == u16.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<uint16_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<uint16_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == u32.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<uint32_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<uint32_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else if (cTID == u64.index) {
      if (nel > nelPar) {
        binaryarith_casted_parallel<uint64_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      } else {
        binaryarith_casted_1thread<uint64_t>(
            out, inp1, inp2, op, nel, flip, oTID, i1TID, i2TID
        );
      }
    } else {
      return "Unsupported type";
    }
  }

  return nullptr;
}

template <typename I>
void binaryarith_1thread(
    I *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip
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
      I * out, I * inp1, I * inp2, BinaryOp op, uint64_t nel, uint8_t flip     \
  );

UNWIND1_ALL_TYPES(BINARYARITH1THREAD)

template <typename I>
void binaryarith_parallel(
    I *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip
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
      out + offset, inp1 + offset, inp2 + offset, op, tail, flip
  );
}

#define BINARYARITHPARALLEL(I)                                                 \
  template void binaryarith_parallel<I>(                                       \
      I * out, I * inp1, I * inp2, BinaryOp op, uint64_t nel, uint8_t flip     \
  );

UNWIND1_ALL_TYPES(BINARYARITHPARALLEL)

template <typename I>
void binaryarith_casted_1thread(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  constexpr uint64_t laneSize = simdSize<I>();
  typedef I SimdType __attribute__((vector_size(sizeof(I) * laneSize)));

  Caster<I, laneSize> i1Caster, i2Caster, oCaster;
  dtypes[i1TID].template caster<I, laneSize>(i1Caster);
  dtypes[i2TID].template caster<I, laneSize>(i2Caster);
  dtypes[outTID].template caster<I, laneSize>(oCaster);

  uint64_t tail = nel % laneSize;
  uint64_t lanesEnd = nel - tail;

  if (op == BinaryOp::Plus) {
    for (uint64_t i = 0; i < lanesEnd; i+=laneSize) {
      SimdType a, b;
      i1Caster.loadSimd(a, inp1, i);
      i2Caster.loadSimd(b, inp2, i);
      SimdType res = a + b;
      oCaster.storeSimd(out, res, i);
    }
    for(uint64_t i = lanesEnd; i < nel; i++) {
      I a, b;
      i1Caster.load(a, inp1, i);
      i2Caster.load(b, inp2, i);
      I res = a + b;
      oCaster.store(out, res, i);
    }
  }

  // TODO handle tail
}

#define BINARYARITHCASTED1THREAD(I)                                            \
  template void binaryarith_casted_1thread<I>(                                 \
      void * out, void * inp1, void * inp2, BinaryOp op, uint64_t nel,         \
      uint8_t flip, uint8_t outTID, uint8_t i1TID, uint8_t i2TID               \
  );

UNWIND1_ALL_TYPES(BINARYARITHCASTED1THREAD)

template <typename I>
void binaryarith_casted_parallel(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
#if 0
  constexpr uint64_t laneSize = simdSize<I>();
  typedef I SimdType __attribute__((vector_size(sizeof(I) * laneSize)));

  const DType &oType = dtypes[outTID];
  const DType &i1Type = dtypes[i1TID];
  const DType &i2Type = dtypes[i2TID];
  const auto &i1SimdLoader = castedVectorStore<I, laneSize>(i1Type);
  const auto &i2SimdLoader = castedVectorStore<I, laneSize>(i2Type);
  const auto &oSimdStorer = castedVectorStore<I, laneSize>(oType);
  const auto &i1Loader = castedLoader<I>(i1Type);
  const auto &i2Loader = castedLoader<I>(i2Type);
  const auto &oStorer = castedStorer<I>(oType);

  using MainKernel = std::function<void(uint64_t, uint64_t)>;
  MainKernel kernel;
  MainKernel tailKernel;
  if (op == BinaryOp::Plus) {
    kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
              oSimdStorer](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i += laneSize) {
        SimdType a, b;
        i1SimdLoader(a, inp1, i);
        i2SimdLoader(b, inp2, i);
        SimdType res = a + b;
        oSimdStorer(out, res, i);
      }
    };
    tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                  oStorer](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = a + b;
        oSimdStorer(out, res, i);
      }
    };
  } else if (op == BinaryOp::Minus) {
    if (!flip) {
      kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
                oSimdStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          i1SimdLoader(a, inp1, i);
          i2SimdLoader(b, inp2, i);
          SimdType res = a - b;
          oSimdStorer(out, res, i);
        }
      };
      tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                    oStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
          I a, b;
          i1Loader(a, inp1, i);
          i2Loader(b, inp2, i);
          I res = a - b;
          oSimdStorer(out, res, i);
        }
      };
    } else {
      kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
                oSimdStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          i1SimdLoader(a, inp1, i);
          i2SimdLoader(b, inp2, i);
          SimdType res = b - a;
          oSimdStorer(out, res, i);
        }
      };
      tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                    oStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
          I a, b;
          i1Loader(a, inp1, i);
          i2Loader(b, inp2, i);
          I res = b - a;
          oSimdStorer(out, res, i);
        }
      };
    }
  } else if (op == BinaryOp::Mul) {
    kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
              oSimdStorer](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i += laneSize) {
        SimdType a, b;
        i1SimdLoader(a, inp1, i);
        i2SimdLoader(b, inp2, i);
        SimdType res = a * b;
        oSimdStorer(out, res, i);
      }
    };
    tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                  oStorer](uint64_t start, uint64_t end) {
      for (uint64_t i = start; i < end; i++) {
        I a, b;
        i1Loader(a, inp1, i);
        i2Loader(b, inp2, i);
        I res = a * b;
        oSimdStorer(out, res, i);
      }
    };
  } else if (op == BinaryOp::Div) {
    if (!flip) {
      kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
                oSimdStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          i1SimdLoader(a, inp1, i);
          i2SimdLoader(b, inp2, i);
          SimdType res = a / b;
          oSimdStorer(out, res, i);
        }
      };
      tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                    oStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
          I a, b;
          i1Loader(a, inp1, i);
          i2Loader(b, inp2, i);
          I res = a / b;
          oSimdStorer(out, res, i);
        }
      };
    } else {
      kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
                oSimdStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          i1SimdLoader(a, inp1, i);
          i2SimdLoader(b, inp2, i);
          SimdType res = b / a;
          oSimdStorer(out, res, i);
        }
      };
      tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                    oStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
          I a, b;
          i1Loader(a, inp1, i);
          i2Loader(b, inp2, i);
          I res = b / a;
          oSimdStorer(out, res, i);
        }
      };
    }
  } else if (op == BinaryOp::Pow) {
    if (!flip) {
      kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
                oSimdStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          i1SimdLoader(a, inp1, i);
          i2SimdLoader(b, inp2, i);
          SimdType res;
          for (uint64_t j = 0; j < laneSize; j++) {
            res[j] = std::pow(a[j], b[j]);
          }
          oSimdStorer(out, res, i);
        }
      };
      tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                    oStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
          I a, b;
          i1Loader(a, inp1, i);
          i2Loader(b, inp2, i);
          I res = std::pow(a, b);
          oSimdStorer(out, res, i);
        }
      };
    } else {
      kernel = [out, inp1, inp2, i1SimdLoader, i2SimdLoader,
                oSimdStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i += laneSize) {
          SimdType a, b;
          i1SimdLoader(a, inp1, i);
          i2SimdLoader(b, inp2, i);
          SimdType res;
          for (uint64_t j = 0; j < laneSize; j++) {
            res[j] = std::pow(b[j], a[j]);
          }
          oSimdStorer(out, res, i);
        }
      };
      tailKernel = [out, inp1, inp2, i1Loader, i2Loader,
                    oStorer](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
          I a, b;
          i1Loader(a, inp1, i);
          i2Loader(b, inp2, i);
          I res = std::pow(b, a);
          oSimdStorer(out, res, i);
        }
      };
    }
  }

  parallelSimdTransform(nel, laneSize, kernel);

  uint64_t tail = nel % laneSize;
  uint64_t offset = nel - tail;
  tailKernel(offset, nel);
#endif
}