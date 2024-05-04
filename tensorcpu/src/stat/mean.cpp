//
// Created by tejag on 2024-05-03.
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
void mean_1thread(O *out, I *inp, uint64_t nel) {
  using ISimd = typename MeanSimd<O, I>::ISimdType;
  using OSimd = typename MeanSimd<O, I>::OSimdType;

  MeanSimd<O, I> mean;
  for (uint64_t i = 0; i < nel; i += MeanSimd<O, I>::sizeSimd) {
    ISimd i1;
    memcpy(&i1, inp + i, MeanSimd<O, I>::sizeSimd * sizeof(I));
    mean.consumeSimd(i1);
  }

  Mean<O, I> meanFinal = mean.materialize();
  auto tail = nel % MeanSimd<O, I>::sizeSimd;
  for (uint64_t i = nel - tail; i < nel; i++) {
    meanFinal.consume(inp[i]);
  }
  *out = meanFinal.mean;
}

#define TCMEAN1THREAD(O, I)                                                    \
  template void mean_1thread<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_ALL_2ND(TCMEAN1THREAD, float)
UNWIND2_ALL_2ND(TCMEAN1THREAD, double)

template <typename O, typename I>
void mean_parallel(O *out, I *inp, uint64_t nel) {
  uint64_t laneSize = MeanSimd<O, I>::sizeSimd;
  using ISimd = typename MeanSimd<O, I>::ISimdType;

  uint16_t numThreads = 0;
  Mean<O, I> means[std::thread::hardware_concurrency()];
  parallelSimdFold(
      nel, laneSize,
      [inp, &means](uint16_t threadId, uint64_t start, uint64_t end) {
        MeanSimd<O, I> ret;
        for (uint64_t lane = start; lane < end;
             lane += MeanSimd<O, I>::sizeSimd) {
          ISimd i1;
          memcpy(&i1, inp + lane, sizeof(ISimd));
          ret.consumeSimd(i1);
        }
        means[threadId] = ret.materialize();
      },
      numThreads
  );

  Mean<O, I> mean = means[0];
  for (uint16_t i = 1; i < numThreads; i++) {
    mean.merge(means[i]);
  }
  for (uint64_t i = nel - (nel % laneSize); i < nel; i++) {
    mean.consume(inp[i]);
  }
  *out = mean.mean;
}

#define TCMEANPARALLEL(O, I)                                                   \
  template void mean_parallel<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_ALL_2ND(TCMEANPARALLEL, float)
UNWIND2_ALL_2ND(TCMEANPARALLEL, double)

template <typename O, typename I> void tcMean(O *out, I *inp, uint64_t nel) {
  if (nel <= 20480) {
    mean_1thread(out, inp, nel);
    return;
  }
  mean_parallel(out, inp, nel);
}

#define TCMEAN(O, I) template void tcMean<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_ALL_2ND(TCMEAN, float)
UNWIND2_ALL_2ND(TCMEAN, double)