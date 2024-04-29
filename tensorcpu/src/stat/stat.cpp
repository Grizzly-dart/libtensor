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
  // TODO use parallelFold
  *out = std::reduce(
      std::execution::par_unseq, inp, inp + nel, O(0),
      [](O a, I b) { return a + b; }
  );
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