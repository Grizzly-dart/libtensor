//
// Created by tejag on 2024-04-26.
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <future>
#include <iostream>
#include <limits>
#include <stdfloat>

#include "tensorcpu.hpp"
#include "typed_array.hpp"
#include "test_common.hpp"
#include "stat.hpp"
#include "thread_pool.hpp"

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename O, typename I>
const char *tcVarianceNaive(
    O *out, const I *inp, uint64_t nel, uint64_t correction
) {
  O mean = 0;
  for (uint64_t i = 0; i < nel; i++) {
    mean += inp[i];
  }
  mean /= nel;

  O ret = 0;
  for (uint64_t i = 0; i < nel; i++) {
    ret += (inp[i] - mean) * (inp[i] - mean);
  }
  *out = ret / (nel - correction);
  return nullptr;
}

template <typename O, typename I>
void check(
    O out, const I *inp, uint64_t nel, uint64_t correction, const char *name,
    uint64_t iteration
) {
  O res;
  tcVarianceNaive(&res, inp, nel, correction);
  O diff = std::abs(res - out);
  if (diff > nel * 1e-6) {
    std::cerr << "In " << name << "; size = " << nel
              << "; Iteration: " << iteration << "; Mismatch => " << res
              << " != " << out << "; " << diff << std::endl;
    exit(1);
  }
}

int main() {
  using I = float;
  using O = float;
  uint64_t correction = 1;

  std::vector<uint64_t> sizes;
  makeSizes1d(sizes, std::min(simdSize<O>(), simdSize<I>()));

  const int64_t iterations = 100;
  for (uint64_t size : sizes) {
    I *inp = new (std::align_val_t(128)) I[size];
    fillRand(inp, size);
    for (uint64_t i = 0; i < iterations; i++) {
      O out = 0;
      variance_parallel<O, I>(&out, inp, size, correction);
      check(out, inp, size, correction, "sum_parallel", i);
    }
    for (uint64_t i = 0; i < iterations; i++) {
      O out = 0;
      variance_1thread<O, I>(&out, inp, size, correction);
      check(out, inp, size, correction, "sum_1thread", i);
    }
    for (uint64_t i = 0; i < iterations; i++) {
      O out = 0;
      tcVariance<O, I>(&out, inp, size, correction);
      check(out, inp, size, correction, "tcSum", i);
    }
    delete[] inp;
  }

  pool.kill();

  return 0;
}