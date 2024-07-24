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

#include "binaryarith.hpp"
#include "reducer.hpp"
#include "stat.hpp"
#include "tensorcpu.hpp"
#include "test_common.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

int main() {
  using I = float;
  using O = float;

  std::vector<uint64_t> sizes;
  make1dBenchSizes(sizes, std::min(simdSize<O>(), simdSize<I>()));

  const int64_t iterations = 100;

  steady_clock::time_point begin, end;

  for (uint64_t i = 0; i < 5; i++) {
    uint64_t size = 1;
    std::unique_ptr<I> inp1(new (std::align_val_t(128)) I[size]);
    std::unique_ptr<I> inp2(new (std::align_val_t(128)) I[size]);
    std::unique_ptr<O> out(new (std::align_val_t(128)) O[size]);
    begin = steady_clock::now();
    binaryarith_parallel<I>(out.get(), inp1.get(), inp2.get(), Plus, size, 0);
    end = steady_clock::now();
    std::cout
        << "Duration:"
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    begin = steady_clock::now();
  }

  Mean<double, int64_t> average;
  for (bool sleep : {false, true}) {
    for (uint64_t size : sizes) {
      std::unique_ptr<I> inp1(new (std::align_val_t(128)) I[size]);
      std::unique_ptr<I> inp2(new (std::align_val_t(128)) I[size]);
      std::unique_ptr<O> out(new (std::align_val_t(128)) O[size]);
      fillRand(inp1.get(), size);
      fillRand(inp2.get(), size);
      for (BinaryOp op : {Plus, Minus, Mul, Div, Pow}) {
        {
          average = Mean<double, int64_t>();
          average = Mean<double, int64_t>();
          for (uint64_t i = 0; i < iterations; i++) {
            begin = steady_clock::now();
            binaryarith_parallel<I>(
                out.get(), inp1.get(), inp2.get(), op, size, 0
            );
            end = steady_clock::now();
            auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
            average.consume(dur.count());
            if (sleep) {
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
          }
          std::cerr << "parallel; size:" << size << "; sleep:" << sleep
                    << "; took:" << average.mean << " us" << std::endl;
        }
        {
          average = Mean<double, int64_t>();
          for (uint64_t i = 0; i < iterations; i++) {
            begin = steady_clock::now();
            binaryarith_1thread<I>(
                out.get(), inp1.get(), inp2.get(), op, size, 0
            );
            end = steady_clock::now();
            auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
            average.consume(dur.count());
            if (sleep) {
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
          }
          std::cerr << "1thread; size:" << size << "; took:" << average.mean
                    << " us" << std::endl;
        }
      }
    }
  }

  pool.kill();

  return 0;
}