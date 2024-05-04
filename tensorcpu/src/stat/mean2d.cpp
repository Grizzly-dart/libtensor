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
void mean2d_1thread(O *out, const I *inp, uint64_t rows, uint64_t cols) {
  using ISimd = MeanSimd<O, I>::ISimdType;
  constexpr uint64_t laneSize = MeanSimd<O, I>::laneSize;

  for (uint64_t row = 0; row < rows; row++) {
    MeanSimd<O, I> folder;
    for (uint64_t col = 0; col < cols; col += laneSize) {
      ISimd a;
      memcpy(&a, inp, sizeof(ISimd));
      folder.consumeSimd(a);
      inp += laneSize;
    }

    Mean<O, I> reducer = folder.materialize();
    uint64_t tail = cols % laneSize;
    for(uint64_t col = 0; col < tail; col++) {
      reducer.consume(inp[col]);
    }
    inp += tail;
    out[row] = reducer.mean;
  }
}

template <typename O, typename I>
const char *tcMean2d(O *out, const I *inp, uint64_t rows, uint64_t cols) {
  constexpr uint64_t laneSize = simdSize<O>();
  typedef I ISimdType __attribute__((vector_size(sizeof(I) * laneSize)));
  typedef O OSimdType __attribute__((vector_size(sizeof(O) * laneSize)));
  uint64_t laneEnd = (cols / laneSize) * laneSize;
  uint64_t tail = cols - laneEnd;

  parallelFold2d(
      rows,
      [laneEnd, inp, cols, tail, out](uint64_t startRow, uint64_t endRow) {
        const I *in = inp + startRow * cols;
        for (uint64_t row = startRow; row < endRow; row++) {
          MeanSimd<O, I> folder;
          for (uint64_t i = 0; i < laneEnd; i += laneSize) {
            ISimdType a = {0};
            memcpy(&a, in, sizeof(ISimdType));
            folder.consumeSimd(a);
            in += laneSize;
          }

          Mean<O, I> reducer = folder.materialize();
          for (uint64_t i = 0; i < tail; i++) {
            reducer.consume(in[i]);
          }
          out[row] = reducer.mean;
          in += tail;
        }
      }
  );
  return nullptr;
}

#define TCMEAN2D(O, I)                                                         \
  template const char *tcMean2d(                                               \
      O *out, const I *inp, uint64_t rows, uint64_t cols                       \
  );
