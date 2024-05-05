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

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename O, typename I>
void check(O out, const I *inp, uint64_t nel, const char *name) {
  O res = 0;
  for (uint64_t i = 0; i < nel; i++) {
    res += inp[i];
  }
  O diff = std::abs(res - out);
  if (diff > res * 1e-3) {
    std::cout << "In " << name << "; size = " << nel << "; Mismatch => " << res << " != " << out
              << "; " << diff << std::endl;
  }
}

int main() {
  using I = float;
  using O = float;

  uint64_t sizes[] = {
      1,    2,         simdSize<O>() - 1, simdSize<O>(), simdSize<O>() + 1,
      2048, 2048 * 10, 2048 * 100,        2048 * 1000
  };

  const int64_t iterations = 1;
  for (uint64_t size : sizes) {
    I *inp = new (std::align_val_t(128)) I[size];
    fillRand(inp, size);
    for (uint64_t i = 0; i < iterations; i++) {
      O out = 0;
      sum_parsimd<O, I>(&out, inp, size);
      check(out, inp, size, "sum_parsimd");
    }
    for (uint64_t i = 0; i < iterations; i++) {
      O out = 0;
      sum_1thread<O, I>(&out, inp, size);
      check(out, inp, size, "sum_1thread");
    }
    for (uint64_t i = 0; i < iterations; i++) {
      O out = 0;
      tcSum<O, I>(&out, inp, size);
      check(out, inp, size, "tcSum");
    }
    delete[] inp;
  }

  pool.kill();

  return 0;
}