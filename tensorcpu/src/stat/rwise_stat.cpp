//
// Created by tejag on 2024-04-29.
//

#include <cstdint>
#include <iostream>

#include "reducer.hpp"

template <typename O, typename I>
const char *tcSum2d(O *out, const I *inp, uint64_t rows, uint64_t cols) {
  constexpr uint64_t laneSize = simdSize<I>();
  typedef I ISimdType __attribute__((vector_size(sizeof(I) * laneSize)));
  typedef O OSimdType __attribute__((vector_size(sizeof(O) * laneSize)));
  uint64_t laneEnd = (cols / laneSize) * laneSize;
  uint64_t tail = cols - laneEnd;

  parallelFold2d(
      rows,
      [laneEnd, inp, cols, tail, out](uint64_t start, uint64_t last) {
        ISimdType a;
        OSimdType sum = {0};
        const I *in = inp + start * cols;
        for (uint64_t row = start; row < last; row++) {
          for (uint64_t i = 0; i < laneEnd; i += laneSize) {
            memcpy(&a, in, sizeof(ISimdType));
            sum += a;
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
          memset(&sum, 0, sizeof(OSimdType));
        }
      }
  );

  return nullptr;
}

#define TCSUM2D(O, I)                                                          \
  template const char *tcSum2d(                                                \
      O *out, const I *inp, uint64_t rows, uint64_t cols                       \
  );

TCSUM2D(float, float)