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
void variance_1thread(O *out, I *inp, uint64_t nel, uint64_t correction) {
  uint64_t laneSize = VarianceSimd<O, I>::sizeSimd;
  using ISimd = typename VarianceSimd<O, I>::ISimdType;

  VarianceSimd<O, I> ret;
  for (uint64_t lane = 0; lane < nel; lane += laneSize) {
    ISimd i1;
    memcpy(&i1, inp + lane, sizeof(ISimd));
    ret.consumeSimd(i1);
  }

  Variance<O, I> reducer = ret.materialize();
  for (uint64_t i = nel - (nel % laneSize); i < nel; i++) {
    reducer.consume(inp[i]);
  }
  *out = reducer.m2 / (reducer.n - correction);
}

#define TCVARIANCE(O, I)                                                       \
  template void variance_1thread<O, I>(                                        \
      O * out, I * inp, uint64_t nel, uint64_t correction                      \
  );

UNWIND2_ALL_2ND(TCVARIANCE, float)
UNWIND2_ALL_2ND(TCVARIANCE, double)

template <typename O, typename I>
void variance_parallel(O *out, I *inp, uint64_t nel, uint64_t correction) {
  uint64_t laneSize = VarianceSimd<O, I>::sizeSimd;
  using ISimd = typename VarianceSimd<O, I>::ISimdType;

  uint16_t numThreads = 0;
  Variance<O, I> variances[std::thread::hardware_concurrency()];
  parallelSimdFold(
      nel, laneSize,
      [inp, &variances](uint16_t threadId, uint64_t start, uint64_t end) {
        VarianceSimd<O, I> ret;
        for (uint64_t lane = start; lane < end;
             lane += VarianceSimd<O, I>::sizeSimd) {
          ISimd i1;
          memcpy(&i1, inp + lane, sizeof(ISimd));
          ret.consumeSimd(i1);
        }
        variances[threadId] = ret.materialize();
      },
      numThreads
  );

  Variance<O, I> reducer = variances[0];
  for (uint16_t i = 1; i < numThreads; i++) {
    reducer.merge(variances[i]);
  }

  uint64_t tail = nel % laneSize;
  for (uint64_t i = nel - tail; i < nel; i++) {
    reducer.consume(inp[i]);
  }
  *out = reducer.m2 / (reducer.n - correction);
}

#define TCVARIANCE(O, I)                                                       \
  template void variance_parallel<O, I>(                                       \
      O * out, I * inp, uint64_t nel, uint64_t correction                      \
  );

UNWIND2_ALL_2ND(TCVARIANCE, float)
UNWIND2_ALL_2ND(TCVARIANCE, double)

template <typename O, typename I>
void tcVariance(O *out, I *inp, uint64_t nel, uint64_t correction) {
  if (nel < 1000) {
    variance_1thread(out, inp, nel, correction);
    return;
  } else {
    variance_parallel(out, inp, nel, correction);
    return;
  }
}

#define TCVARIANCE(O, I)                                                       \
  template void tcVariance<O, I>(                                              \
      O * out, I * inp, uint64_t nel, uint64_t correction                      \
  );

UNWIND2_ALL_2ND(TCVARIANCE, float)
UNWIND2_ALL_2ND(TCVARIANCE, double)