//
// Created by tejag on 2024-05-04.
//

#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>

#include "macro_unwind.hpp"
#include "thread_pool.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename O, typename I>
void sum2d_1thread(O *out, I *inp, uint64_t rows, uint64_t cols) {
  for (uint64_t row = 0; row < rows; row++) {
    O ret = O(0);
#pragma GCC ivdep
    for (uint64_t col = 0; col < cols; col++) {
      ret += inp[row * cols + col];
    }
    out[row] = ret;
  }
}

#define TCSUM2D1THREAD(O, I)                                                   \
  template void sum2d_1thread(O *out, I *inp, uint64_t rows, uint64_t cols);

UNWIND2_UP(TCSUM2D1THREAD)

template <typename O, typename I>
void sum2d_parallel(O *out, I *inp, uint64_t rows, uint64_t cols) {
  constexpr uint64_t laneSize = simdSize<O>();
  typedef I ISimdType __attribute__((vector_size(sizeof(I) * laneSize)));
  typedef O OSimdType __attribute__((vector_size(sizeof(O) * laneSize)));
  uint64_t tail = cols % laneSize;
  uint64_t endCol = cols - tail;

  parallelFold2d(
      rows,
      [inp, cols, out, tail, endCol](uint16_t threadId, uint64_t startRow, uint64_t endRow) {
        I *in = inp + startRow * cols;
        for (uint64_t row = startRow; row < endRow; row++) {
          OSimdType sum = {0};
          for (uint64_t i = 0; i < endCol; i += laneSize) {
            ISimdType a;
            memcpy(&a, in, sizeof(ISimdType));
            if constexpr (std::is_same<O, I>::value) {
              sum += a;
            } else {
              sum += __builtin_convertvector(a, OSimdType);
            }
            in += laneSize;
          }

          O res = 0;
          for (uint64_t i = 0; i < laneSize; i++) {
            res += O(sum[i]);
          }
          for (uint64_t i = 0; i < tail; i++) {
            res += in[i];
          }
          in += tail;
          out[row] = res;
        }
      }
  );
}

#define TCSUM2DPARALLEL(O, I)                                                  \
  template void sum2d_parallel(O *out, I *inp, uint64_t rows, uint64_t cols);

UNWIND2_UP(TCSUM2DPARALLEL)

template <typename O, typename I>
void tcSum2d(O *out, I *inp, uint64_t rows, uint64_t cols) {
  if (cols * rows < 1000 || rows == 1) {
    sum2d_1thread(out, inp, rows, cols);
    return;
  } else {
    sum2d_parallel(out, inp, rows, cols);
    return;
  }
}

#define TCSUM2D(O, I)                                                          \
  template void tcSum2d(O *out, I *inp, uint64_t rows, uint64_t cols);

UNWIND2_UP(TCSUM2D)