//
// Created by tejag on 2024-05-04.
//

#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>

#include "macro_unwind.hpp"
#include "reducer.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename O, typename I>
void variance2d_1thread(
    O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction
) {
  using ISimd = VarianceSimd<O, I>::ISimdType;
  constexpr uint64_t laneSize = VarianceSimd<O, I>::sizeSimd;
  uint64_t tail = cols % laneSize;
  uint64_t endCol = cols - tail;

  for (uint64_t row = 0; row < rows; row++) {
    VarianceSimd<O, I> folder;
    for (uint64_t col = 0; col < endCol; col += laneSize) {
      ISimd a;
      memcpy(&a, inp, sizeof(ISimd));
      folder.consumeSimd(a);
      inp += laneSize;
    }

    Variance<O, I> reducer = folder.materialize();
    for (uint64_t col = 0; col < tail; col++) {
      reducer.consume(inp[col]);
    }
    inp += tail;
    out[row] = reducer.m2 / (cols - correction);
  }
}

#define TCVARIANCE2D1THREAD(O, I)                                              \
  template void variance2d_1thread(                                            \
      O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction        \
  );

UNWIND2_ALL_2ND(TCVARIANCE2D1THREAD, float)
UNWIND2_ALL_2ND(TCVARIANCE2D1THREAD, double)

template <typename O, typename I>
void variance2d_parallel(
    O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction
) {
  using ISimd = VarianceSimd<O, I>::ISimdType;
  constexpr uint64_t laneSize = VarianceSimd<O, I>::sizeSimd;
  uint64_t tail = cols % laneSize;
  uint64_t endCol = cols - tail;

  parallelFold2d(
      rows,
      [inp, cols, out, correction, tail,
       endCol](uint16_t threadId, uint64_t startRow, uint64_t endRow) {
        I *in = inp + startRow * cols;
        for (uint64_t row = startRow; row < endRow; row++) {
          VarianceSimd<O, I> folder;
          for (uint64_t col = 0; col < endCol; col += laneSize) {
            ISimd a;
            memcpy(&a, in, sizeof(ISimd));
            folder.consumeSimd(a);
            in += laneSize;
          }

          Variance<O, I> reducer = folder.materialize();
          for (uint64_t col = 0; col < tail; col++) {
            reducer.consume(in[col]);
          }
          in += tail;
          out[row] = reducer.m2 / (cols - correction);
        }
      }
  );
}

#define TCVARIANCE2DPARALLEL(O, I)                                             \
  template void variance2d_parallel(                                           \
      O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction        \
  );

UNWIND2_ALL_2ND(TCVARIANCE2DPARALLEL, float)
UNWIND2_ALL_2ND(TCVARIANCE2DPARALLEL, double)

template <typename O, typename I>
void tcVariance2d(
    O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction
) {
  if (cols * rows < 1000) {
    variance2d_1thread(out, inp, rows, cols, correction);
    return;
  } else {
    variance2d_parallel(out, inp, rows, cols, correction);
    return;
  }
}

#define TCVARIANCE2D(O, I)                                                     \
  template void tcVariance2d(                                                  \
      O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction        \
  );

UNWIND2_ALL_2ND(TCVARIANCE2D, float)
UNWIND2_ALL_2ND(TCVARIANCE2D, double)
