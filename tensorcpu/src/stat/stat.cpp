#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>

#include "macro_unwind.hpp"
#include "reducer.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename O, typename I> void tcSum(O *out, I *inp, uint64_t nel) {
  constexpr uint64_t laneSize = simdSize<O>();
  typedef I ISimd __attribute__((vector_size(laneSize * sizeof(I))));
  typedef O OSimd __attribute__((vector_size(laneSize * sizeof(O))));

  std::vector<OSimd> sums;
  parallelSimdFold(
      nel, laneSize,
      [inp, laneSize](uint64_t start, uint64_t end) -> OSimd {
        OSimd ret = {0};
        ISimd a;
        for (uint64_t i = start; i < end; i += laneSize) {
          memcpy(&a, inp + i, laneSize * sizeof(I));
          ret += a;
        }
        return ret;
      },
      sums
  );

  OSimd finalSum = {0};
  for (auto &sum : sums) {
    finalSum += sum;
  }

  O ret = 0;
  uint64_t tail = nel % laneSize;
  for (uint64_t i = 0; i < laneSize; i++) {
    ret += finalSum[i];
  }
  *out = ret;

  /**out = std::reduce(
      std::execution::par_unseq, inp, inp + nel, O(0),
      [](O a, I b) { return a + b; }
  );*/
}

#define TCSUM(O, I) template void tcSum<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_SAME_ALL_TYPES(TCSUM)

template <typename O, typename I> void tcMean(O *out, I *inp, uint64_t nel) {
  uint64_t laneSize = MeanSimd<O, I>::sizeSimd;
  using ISimd = typename MeanSimd<O, I>::ISimdType;

  std::vector<MeanSimd<O, I>> means;
  std::function<MeanSimd<O, I>(uint64_t, uint64_t)> kernel =
      [laneSize, inp](uint64_t start, uint64_t end) -> MeanSimd<O, I> {
    MeanSimd<O, I> mean;
    ISimd i1;
    for (uint64_t lane = start; lane < end; lane += laneSize) {
      memcpy(&i1, inp + lane, laneSize * sizeof(I));
      mean.consumeSimd(i1);
    }
    return mean;
  };
  parallelSimdFold(nel, laneSize, kernel, means);

  MeanSimd<O, I> finalMean;
  for (auto &mean : means) {
    finalMean.merge(mean);
  }

  Mean<O, I> mean = finalMean.mean();

  uint64_t tail = nel % laneSize;
  for (uint64_t i = nel - tail; i < nel; i++) {
    mean.consume(inp[i]);
  }

  *out = mean.mean;
}

#define TCMEAN(O, I) template void tcMean<O, I>(O * out, I * inp, uint64_t nel);

TCMEAN(float, float)