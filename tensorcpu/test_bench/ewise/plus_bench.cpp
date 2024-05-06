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
#include "typed_array.hpp"
#include "test_common.hpp"
#include "binaryarith.hpp"

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
    O out, const I *inp1, const I *inp2, uint64_t nel, const char *name, uint64_t iteration
) {
  for (uint64_t i = 0; i < nel; i++) {
    O res = inp1[i] + inp2[i];
    O diff = std::abs(res - out[i]);
    if (diff > res * 1e-3) {
      std::cerr << "In " << name << "; size = " << nel
                << "; Iteration: " << iteration << "; Mismatch @" << i << " " << res
                << " != " << out[i] << "; " << diff << std::endl;
      exit(1);
    }
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
      std::unique_ptr<I> inp1(new (std::align_val_t(128)) I[size]);
      std::unique_ptr<I> inp2(new (std::align_val_t(128)) I[size]);
      fillRand(inp1.get(), size);
      fillRand(inp2.get(), size);
      std::unique_ptr<O> out(new (std::align_val_t(128)) O[size]);
      Mean<double, int64_t> average;
      {
        average = Mean<double, int64_t>();
        for (uint64_t i = 0; i < iterations; i++) {
          begin = steady_clock::now();
          tcBinaryArith<O, I>(out.get(), inp1.get(), inp2.get(), Plus, size, false, Dim2{.r = 0, .c = 0});
          end = steady_clock::now();
          auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
          // TODO print single line result
          average.consume(dur.count());
          if (sleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
        }
        std::cerr << "tcBinaryArith size = " << size << "; took: " << average.mean << " us" << std::endl;
      }
      {
        average = Mean<double, int64_t>();
        for (uint64_t i = 0; i < iterations; i++) {
          begin = steady_clock::now();
          binaryarith_parallel<I>(out.get(), inp1.get(), inp2.get(), Plus, size, 0, Dim2{.r = 0, .c = 0});
          end = steady_clock::now();
          auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
          // TODO print single line result
          average.consume(dur.count());
          if (sleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
        }
        std::cerr << "parallel => size = " << size << "; took: " << average.mean << " us" << std::endl;
      }
      /*{
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
      }
      {
        Mean<double, int64_t> average;
        for (uint64_t i = 0; i < iterations; i++) {
          O out = 0;
          begin = steady_clock::now();
          sum_stdalgo<I>(&out, inp, size);
          end = steady_clock::now();
          auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
          // TODO print single line result
          check(out, inp, size, "sum_stdalgo", i);
          average.consume(dur.count());
          if (sleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
        }
      }*/
      // TODO print average
    }
  }

  pool.kill();

  return 0;
}