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

#include "reducer.hpp"
#include "stat.hpp"
#include "tensorcpu.hpp"
#include "test_common.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename I>
const char *sum_stdalgo(I *out, const I *inp, uint64_t nel) {
  *out = std::reduce(
      std::execution::par_unseq, inp, inp + nel, I(0),
      [](I a, I b) { return a + b; }
  );
  return nullptr;
}

template <typename O, typename I>
void check(
    O out, const I *inp, uint64_t nel, const char *name, uint64_t iteration
) {
  O res;
  for (uint64_t i = 0; i < nel; i++) {
    res += inp[i];
  }
  O diff = std::abs(res - out);
  if (diff > res * 1e-3) {
    std::cerr << "In " << name << "; size = " << nel
              << "; Iteration: " << iteration << "; Mismatch => " << res
              << " != " << out << "; " << diff << std::endl;
    exit(1);
  }
}

int main() {
  using I = float;
  using O = float;

  std::vector<uint64_t> sizes;
  make1dBenchSizes(sizes, std::min(simdSize<O>(), simdSize<I>()));

  const int64_t iterations = 100;

  steady_clock::time_point begin, end;
  for (bool sleep : {false, true}) {
    for (uint64_t size : sizes) {
      I *inp = new (std::align_val_t(128)) I[size];
      fillRand(inp, size);
      Mean<double, int64_t> average;
      for (uint64_t i = 0; i < iterations; i++) {
        O out = 0;
        begin = steady_clock::now();
        sum_parallel<O, I>(&out, inp, size);
        end = steady_clock::now();
        auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
        // TODO print single line result
        check(out, inp, size, "sum_parallel", i);
        average.consume(dur.count());
        if (sleep) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
      delete[] inp;
      // TODO print average
    }
  }
  for (bool sleep : {false, true}) {
    for (uint64_t size : sizes) {
      I *inp = new (std::align_val_t(128)) I[size];
      fillRand(inp, size);
      Mean<double, int64_t> average;
      for (uint64_t i = 0; i < iterations; i++) {
        O out = 0;
        begin = steady_clock::now();
        sum_1thread<O, I>(&out, inp, size);
        end = steady_clock::now();
        auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
        // TODO print single line result
        check(out, inp, size, "sum_1thread", i);
        average.consume(dur.count());
        if (sleep) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
      delete[] inp;
      // TODO print average
    }
  }
  for (bool sleep : {false, true}) {
    for (uint64_t size : sizes) {
      I *inp = new (std::align_val_t(128)) I[size];
      fillRand(inp, size);
      Mean<double, int64_t> average;
      for (uint64_t i = 0; i < iterations; i++) {
        O out = 0;
        begin = steady_clock::now();
        sum_stdalgo<I>(&out, inp, size);
        end = steady_clock::now();
        auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
        // TODO print single line result
        check(out, inp, size, "sum_atdalgo", i);
        average.consume(dur.count());
        if (sleep) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
      delete[] inp;
      // TODO print average
    }
  }

  pool.kill();

  return 0;
}