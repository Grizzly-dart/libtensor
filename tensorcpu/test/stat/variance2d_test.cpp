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

#include "stat.hpp"
#include "tensorcpu.hpp"
#include "test_common.hpp"
#include "thread_pool.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename O, typename I>
const char *tcVariance2dNaive(
    O *out, const I *inp, uint64_t rows, uint64_t cols, uint64_t correction
) {
  for (uint64_t row = 0; row < rows; row++) {
    O mean = 0;
    for (uint64_t col = 0; col < cols; col++) {
      mean += inp[col];
    }
    mean /= cols;

    O sum = 0;
    for (uint64_t col = 0; col < cols; col++) {
      sum += (inp[col] - mean) * (inp[col] - mean);
    }
    out[row] = sum / (cols - correction);
    inp += cols;
  }
  return nullptr;
}

template <typename O, typename I>
void check(
    O *out, const I *inp, uint64_t rows, uint64_t cols, uint64_t correction,
    const char *name, uint64_t iteration
) {
  for (uint64_t row = 0; row < rows; row++) {
    O res = 0;
    O mean = 0;
    for (uint64_t col = 0; col < cols; col++) {
      mean += inp[col];
    }
    mean /= cols;

    for (uint64_t col = 0; col < cols; col++) {
      res += (inp[col] - mean) * (inp[col] - mean);
    }
    res /= (cols - correction);

    O diff = std::abs(res - out[row]);
    if (diff > cols * 1e-5) {
      std::cerr << "In " << name << "; size = " << rows << ":" << cols
                << "; Iteration: " << iteration << "; Mismatch @" << row
                << " => " << res << " != " << out[row] << "; " << diff
                << std::endl;
      exit(1);
    }
    inp += cols;
  }
}

int main() {
  using I = float;
  using O = float;

  uint64_t correction = 1;

  std::vector<Dim2> sizes;
  make2dTestSizes(sizes, std::min(simdSize<O>(), simdSize<I>()));

  const int64_t iterations = 100;
  for (Dim2 &size : sizes) {
    I *inp = new (std::align_val_t(128)) I[size.nel()];
    fillRand(inp, size.nel());
    for (uint64_t i = 0; i < iterations; i++) {
      O *out = new (std::align_val_t(128)) O[size.r];
      variance2d_parallel<O, I>(out, inp, size.r, size.c, correction);
      check(out, inp, size.r, size.c, correction, "sum_parallel", i);
    }
    for (uint64_t i = 0; i < iterations; i++) {
      O *out = new (std::align_val_t(128)) O[size.r];
      variance2d_1thread<O, I>(out, inp, size.r, size.c, correction);
      check(out, inp, size.r, size.c, correction, "sum_1thread", i);
    }
    for (uint64_t i = 0; i < iterations; i++) {
      O *out = new (std::align_val_t(128)) O[size.r];
      tcVariance2d<O, I>(out, inp, size.r, size.c, correction);
      check(out, inp, size.r, size.c, correction, "tcSum", i);
    }
    delete[] inp;
  }

  pool.kill();

  return 0;
}