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
void sum_noparnosimd(O *out, const I *inp, uint64_t nel) {
  O ret = O(0);
#pragma GCC ivdep
  for (uint64_t i = 0; i < nel; i++) {
    ret += inp[i];
  }
  *out = ret;
}

#define TCSUMNOPARNOSIMD(O, I)                                                 \
  template void sum_noparnosimd(O *out, const I *inp, uint64_t nel);

UNWIND2_ALL_TYPES(TCSUMNOPARNOSIMD)

template <typename O, typename I>
void sum_parsimd(O *out, I *inp, uint64_t nel) {
  constexpr uint64_t laneSize = simdSize<O>();
  typedef I ISimd __attribute__((vector_size(laneSize * sizeof(I))));
  typedef O OSimd __attribute__((vector_size(laneSize * sizeof(O))));

  std::vector<O> partialSums;
  parallelSimdFold<O>(
      nel, laneSize,
      [inp, &partialSums](uint16_t threadId, uint64_t start, uint64_t end) {
        OSimd ret = {0};
        ISimd a;
        for (uint64_t i = start; i < end; i += laneSize) {
          memcpy(&a, inp + i, laneSize * sizeof(I));
          if constexpr (std::is_same<O, I>::value) {
            ret += a;
          } else {
            ret += __builtin_convertvector(a, OSimd);
          }
        }

        O res = 0;
        for (uint64_t i = 0; i < laneSize; i++) {
          res += ret[i];
        }

        partialSums[threadId] = res;
      },
      partialSums
  );
  O ret = 0;
#pragma GCC ivdep
  for (O sum : partialSums) {
    ret += sum;
  }
  uint64_t tail = nel % laneSize;
  inp += nel - tail;
#pragma GCC ivdep
  for (uint64_t i = 0; i < tail; i++) {
    ret += inp[i];
  }
  *out = ret;
}

#define TCSUMPARSIMD(O, I)                                                     \
  template void sum_parsimd<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_SAME_ALL_TYPES(TCSUMPARSIMD)
TCSUMPARSIMD(double, float)

template <typename O, typename I> void tcSum(O *out, I *inp, uint64_t nel) {
  if (nel <= 204800) {
    sum_noparnosimd(out, inp, nel);
    return;
  }

  sum_parsimd<O, I>(out, inp, nel);
}

#define TCSUM(O, I) template void tcSum<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_SAME_ALL_TYPES(TCSUM)

template <typename O, typename I> void tcMean(O *out, I *inp, uint64_t nel) {
  uint64_t laneSize = MeanSimd<O, I>::sizeSimd;
  using ISimd = typename MeanSimd<O, I>::ISimdType;

  std::function<MeanSimd<O, I>(uint64_t, uint64_t)> kernel =
      [laneSize, inp](uint64_t start, uint64_t end) {
        MeanSimd<O, I> ret;
        ISimd i1;
        for (uint64_t lane = start; lane < end; lane += laneSize) {
          memcpy(&i1, inp + lane, laneSize * sizeof(I));
          ret.consumeSimd(i1);
        }
        return ret;
      };
  MeanSimd<O, I> foldedMean;
  /* TODO
  parallelSimdFold<MeanSimd<O, I>>(
      nel, laneSize, kernel, foldedMean,
      [](MeanSimd<O, I> &a, MeanSimd<O, I> b) { a.merge(b); }
  );
   */

  Mean<O, I> mean = foldedMean.materialize();
  uint64_t tail = nel % laneSize;
  for (uint64_t i = nel - tail; i < nel; i++) {
    mean.consume(inp[i]);
  }

  *out = mean.mean;
}

#define TCMEAN(O, I) template void tcMean<O, I>(O * out, I * inp, uint64_t nel);

TCMEAN(float, float)

template <typename O, typename I>
void tcVariance(O *out, I *inp, uint64_t nel, uint64_t correction) {
  uint64_t laneSize = VarianceSimd<O, I>::sizeSimd;
  using ISimd = typename VarianceSimd<O, I>::ISimdType;

  std::function<VarianceSimd<O, I>(uint64_t, uint64_t)> kernel =
      [laneSize, inp](uint64_t start, uint64_t end) {
        VarianceSimd<O, I> ret;
        ISimd i1;
        for (uint64_t lane = start; lane < end; lane += laneSize) {
          memcpy(&i1, inp + lane, laneSize * sizeof(I));
          ret.consumeSimd(i1);
        }
        return ret;
      };
  VarianceSimd<O, I> foldedMean;
  /*
  parallelSimdFold<VarianceSimd<O, I>>(
      nel, laneSize, kernel, foldedMean,
      [](VarianceSimd<O, I> &a, VarianceSimd<O, I> b) { a.merge(b); }
  );
   */

  Variance<O, I> reducer = foldedMean.materialize();
  uint64_t tail = nel % laneSize;
  for (uint64_t i = nel - tail; i < nel; i++) {
    reducer.consume(inp[i]);
  }

  *out = reducer.m2 / (reducer.n - correction);
}

#define TCVARIANCE(O, I)                                                       \
  template void tcVariance<O, I>(                                              \
      O * out, I * inp, uint64_t nel, uint64_t correction                      \
  );

TCVARIANCE(float, float)