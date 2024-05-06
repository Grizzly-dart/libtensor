//
// Created by tejag on 2024-04-26.
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>

#include "reducer.hpp"
#include "stat.hpp"
#include "tensorcpu.hpp"
#include "test_common.hpp"
#include "typed_array.hpp"
#include "nn_activation.hpp"

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename T> T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

template <typename I>
void check(
    I *out, const I *inp, uint64_t nel, const char *name, uint64_t iteration
) {
  for (uint64_t i = 0; i < nel; i++) {
    I res = sigmoid(inp[i]);
    I diff = std::abs(res - out[i]);
    if (diff > res * 1e-3) {
      std::cerr << "In " << name << "; size = " << nel
                << "; Iteration: " << iteration << "; Mismatch => " << res
                << " != " << out[i] << "; " << diff << std::endl;
      exit(1);
    }
  }
}

int main() {
  using I = float;

  std::vector<uint64_t> sizes;
  make1dTestSizes(sizes, simdSize<I>());

  const int64_t iterations = 1;
  for (uint64_t size : sizes) {
    std::unique_ptr<I> inp(new I[size]);
    fillRand(inp.get(), size);
    for (uint64_t i = 0; i < iterations; i++) {
      std::unique_ptr<I> out(new I[size]);
      sigmoid_parallel<I>(out.get(), inp.get(), size);
      check(out.get(), inp.get(), size, "sigmoid_parallel", i);
    }
  }

  pool.kill();

  return 0;
}