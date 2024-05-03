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

#define TCMEAN1THREAD(O, I) template void mean_1thread<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_ALL_2ND(TCMEAN1THREAD, float)
UNWIND2_ALL_2ND(TCMEAN1THREAD, double)

template <typename O, typename I>
void mean_parallel(O *out, I *inp, uint64_t nel) {
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