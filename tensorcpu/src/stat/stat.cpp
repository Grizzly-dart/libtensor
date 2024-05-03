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