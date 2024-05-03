//
// Created by Teja Gudapati on 2024-05-03.
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
void sum_1thread(O *out, const I *inp, uint64_t nel) {
  O ret = O(0);
#pragma GCC ivdep
  for (uint64_t i = 0; i < nel; i++) {
    ret += inp[i];
  }
  *out = ret;
}

#define TCSUM1THREAD(O, I)                                                 \
  template void sum_1thread(O *out, const I *inp, uint64_t nel);

UNWIND2_UP(TCSUM1THREAD)

template <typename O, typename I>
void sum_parsimd(O *out, I *inp, uint64_t nel) {
  constexpr uint64_t laneSize = simdSize<O>();
  typedef I ISimd __attribute__((vector_size(laneSize * sizeof(I))));
  typedef O OSimd __attribute__((vector_size(laneSize * sizeof(O))));

  OSimd simdSums[std::thread::hardware_concurrency()];
  uint16_t numThreads = 0;
  parallelSimdFold(
      nel, laneSize,
      [inp, &simdSums](uint16_t threadId, uint64_t start, uint64_t end) {
        OSimd ret = {0};
        ISimd a;
        for (uint64_t i = start; i < end; i += laneSize) {
          memcpy(&a, inp + i, sizeof(ISimd));
          if constexpr (std::is_same<O, I>::value) {
            ret += a;
          } else {
            ret += __builtin_convertvector(a, OSimd);
          }
        }

        simdSums[threadId] = ret;
      },
      numThreads
  );
  for (uint16_t i = 1; i < numThreads; i++) {
    simdSums[0] += simdSums[i];
  }
  O ret = 0;
  for (uint64_t i = 0; i < laneSize; i++) {
    ret += simdSums[0][i];
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

UNWIND2_UP(TCSUMPARSIMD)

template <typename O, typename I> void tcSum(O *out, I *inp, uint64_t nel) {
  if (nel <= 20480) {
    sum_1thread(out, inp, nel);
    return;
  }

  sum_parsimd<O, I>(out, inp, nel);
}

#define TCSUM(O, I) template void tcSum<O, I>(O * out, I * inp, uint64_t nel);

UNWIND2_UP(TCSUM)