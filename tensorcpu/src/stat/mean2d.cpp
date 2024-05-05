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
void mean2d_1thread(O *out, I *inp, uint64_t rows, uint64_t cols) {
  using ISimd = MeanSimd<O, I>::ISimdType;
  constexpr uint64_t laneSize = MeanSimd<O, I>::sizeSimd;
  uint64_t tail = cols % laneSize;
  uint64_t endCol = cols - tail;

  for (uint64_t row = 0; row < rows; row++) {
    MeanSimd<O, I> folder;
    for (uint64_t col = 0; col < endCol; col += laneSize) {
      ISimd a;
      memcpy(&a, inp, sizeof(ISimd));
      folder.consumeSimd(a);
      inp += laneSize;
    }

    Mean<O, I> reducer = folder.materialize();
    for (uint64_t col = 0; col < tail; col++) {
      reducer.consume(inp[col]);
    }
    inp += tail;
    out[row] = reducer.mean;
  }
}

#define TCMEAN2D1THREAD(O, I)                                                  \
  template void mean2d_1thread(O *out, I *inp, uint64_t rows, uint64_t cols);

UNWIND2_ALL_2ND(TCMEAN2D1THREAD, float)
UNWIND2_ALL_2ND(TCMEAN2D1THREAD, double)

template <typename O, typename I>
void mean2d_parallel(O *out, I *inp, uint64_t rows, uint64_t cols) {
  constexpr uint64_t laneSize = MeanSimd<O, I>::sizeSimd;
  using ISimd = typename MeanSimd<O, I>::ISimdType;
  uint64_t tail = cols % laneSize;
  uint64_t endCol = cols - tail;

  parallelFold2d(
      rows,
      [inp, cols, out, tail, endCol](uint16_t threadId, uint64_t startRow, uint64_t endRow) {
        I *in = inp + startRow * cols;
        for (uint64_t row = startRow; row < endRow; row++) {
          MeanSimd<O, I> folder;
          for (uint64_t col = 0; col < endCol; col += laneSize) {
            ISimd a;
            memcpy(&a, in, sizeof(ISimd));
            folder.consumeSimd(a);
            in += laneSize;
          }

          Mean<O, I> reducer = folder.materialize();
          for (uint64_t col = 0; col < tail; col++) {
            reducer.consume(in[col]);
          }
          in += tail;
          out[row] = reducer.mean;
        }
      }
  );
}

#define TCMEANPARALLEL(O, I)                                                   \
  template void mean2d_parallel<O, I>(                                         \
      O * out, I * inp, uint64_t rows, uint64_t cols                           \
  );

UNWIND2_ALL_2ND(TCMEANPARALLEL, float)
UNWIND2_ALL_2ND(TCMEANPARALLEL, double)

template <typename O, typename I>
void tcMean2d(O *out, I *inp, uint64_t rows, uint64_t cols) {
  if (cols * rows < 1000 || rows == 1) {
    mean2d_1thread(out, inp, rows, cols);
  } else {
    mean2d_parallel(out, inp, rows, cols);
  }
}

#define TCMEAN2D(O, I)                                                         \
  template void tcMean2d(O *out, I *inp, uint64_t rows, uint64_t cols);

UNWIND2_ALL_2ND(TCMEAN2D, float)
UNWIND2_ALL_2ND(TCMEAN2D, double)
