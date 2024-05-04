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
void sum2d_1thread(O *out, const I *inp, uint64_t rows, uint64_t cols) {
  for (uint64_t row = 0; row < rows; row++) {
    O ret = O(0);
    for (uint64_t col = 0; col < cols; col++) {
      ret += inp[row * cols + col];
    }
    out[row] = ret;
  }
}

#define TCSUM2D1THREAD(O, I)                                                 \
  template void sum2d_1thread(O *out, const I *inp, uint64_t rows, uint64_t cols);

UNWIND2_UP(TCSUM2D1THREAD)

template <typename O, typename I>
void sum2d_parsimd(O *out, I *inp, uint64_t rows, uint64_t cols) {
  constexpr uint64_t laneSize = simdSize<O>();
  typedef I ISimdType __attribute__((vector_size(sizeof(I) * laneSize)));
  typedef O OSimdType __attribute__((vector_size(sizeof(O) * laneSize)));

  // TODO
}

template <typename O, typename I>
const char *tcSum2d(O *out, const I *inp, uint64_t rows, uint64_t cols) {
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
          OSimdType sum = {0};
          ISimdType a = {0};
          for (uint64_t i = 0; i < laneEnd; i += laneSize) {
            memcpy(&a, in, sizeof(ISimdType));
            sum += __builtin_convertvector(a, OSimdType);
            in += laneSize;
          }

          O res = 0;
          for (uint64_t i = 0; i < laneSize; i++) {
            res += sum[i];
          }
          for (uint64_t i = 0; i < tail; i++) {
            res += in[i];
          }
          in += tail;
          out[row] = res;
        }
      }
  );

  return nullptr;
}

#define TCSUM2D(O, I)                                                          \
  template const char *tcSum2d(                                                \
      O *out, const I *inp, uint64_t rows, uint64_t cols                       \
  );

TCSUM2D(float, uint16_t)