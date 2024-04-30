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
      [laneEnd, inp, cols, tail, out](uint64_t startRow, uint64_t endRow) {
        ISimdType a;
        OSimdType sum = {0};
        const I *in = inp + startRow * cols;
        for (uint64_t row = startRow; row < endRow; row++) {
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

template <typename O, typename I>
const char *tcMean2d(O *out, const I *inp, uint64_t rows, uint64_t cols) {
  constexpr uint64_t laneSize = simdSize<I>();
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
            ISimdType a;
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

TCMEAN2D(float, float)

template <typename O, typename I>
const char *tcVariance2d(
    O *out, const I *inp, uint64_t rows, uint64_t cols, uint64_t correction
) {
  constexpr uint64_t laneSize = simdSize<I>();
  typedef I ISimdType __attribute__((vector_size(sizeof(I) * laneSize)));
  typedef O OSimdType __attribute__((vector_size(sizeof(O) * laneSize)));
  uint64_t laneEnd = (cols / laneSize) * laneSize;
  uint64_t tail = cols - laneEnd;

  parallelFold2d(
      rows,
      [laneEnd, inp, cols, tail, out,
       correction](uint64_t startRow, uint64_t endRow) {
        const I *in = inp + startRow * cols;
        for (uint64_t row = startRow; row < endRow; row++) {
          VarianceSimd<O, I> folder;
          for (uint64_t i = 0; i < laneEnd; i += laneSize) {
            ISimdType a;
            memcpy(&a, in, sizeof(ISimdType));
            folder.consumeSimd(a);
            in += laneSize;
          }

          Variance<O, I> reducer = folder.materialize();
          for (uint64_t i = 0; i < tail; i++) {
            reducer.consume(in[i]);
          }
          out[row] = reducer.m2 / (cols - correction);
          in += tail;
        }
      }
  );
  return nullptr;
}

#define TCVARIANCE2D(O, I)                                                     \
  template const char *tcVariance2d(                                           \
      O *out, const I *inp, uint64_t rows, uint64_t cols, uint64_t correction  \
  );

TCVARIANCE2D(float, float)