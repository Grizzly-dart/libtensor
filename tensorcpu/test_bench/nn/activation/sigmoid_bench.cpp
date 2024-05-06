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

#include "nn_activation.hpp"
#include "reducer.hpp"
#include "stat.hpp"
#include "tensorcpu.hpp"
#include "test_common.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename T> T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

template <typename T> void sigmoid_naive(T *out, T *inp, uint64_t nel) {
  for (uint64_t i = 0; i < nel; i++) {
    out[i] = sigmoid(inp[i]);
  }
}

template <typename T>
void check(
    T *out, const T *inp, uint64_t nel, const char *name, uint64_t iteration
) {
  for (uint64_t i = 0; i < nel; i++) {
    T res = sigmoid(inp[i]);
    T diff = std::abs(res - out[i]);
    if (diff > res * 1e-3) {
      std::cerr << "In " << name << "; size = " << nel
                << "; Iteration: " << iteration << "; Mismatch => " << res
                << " != " << out[i] << "; " << diff << std::endl;
      exit(1);
    }
  }
}

int main() {
  using T = float;

  std::vector<uint64_t> sizes;
  make1dBenchSizes(sizes, simdSize<T>());

  const int64_t iterations = 100;

  steady_clock::time_point begin, end;
  for (bool sleep : {false, true}) {
    for (uint64_t size : sizes) {
      std::unique_ptr<T> inp(new T[size]);
      fillRand(inp.get(), size);
      {
        Mean<double, int64_t> average;
        for (uint64_t i = 0; i < iterations; i++) {
          std::unique_ptr<T> out(new T[size]);
          begin = steady_clock::now();
          sigmoid_parallel<T>(out.get(), inp.get(), size);
          end = steady_clock::now();
          auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
          // TODO print single line result
          check(out.get(), inp.get(), size, "sigmoid_parallel", i);
          average.consume(dur.count());
          if (sleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
        }
        std::cout << "sigmoid_parallel: " << size << ": " << size
                  << "; average: " << average.mean << std::endl;
      }
      {
        Mean<double, int64_t> average;
        for (uint64_t i = 0; i < iterations; i++) {
          std::unique_ptr<T> out(new T[size]);
          begin = steady_clock::now();
          sigmoid_naive<T>(out.get(), inp.get(), size);
          end = steady_clock::now();
          auto dur = chrono::duration_cast<chrono::microseconds>(end - begin);
          // TODO print single line result
          check(out.get(), inp.get(), size, "sigmoid_naive", i);
          average.consume(dur.count());
          if (sleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
        }
        std::cout << "sigmoid_naive: " << size << ": " << size
                  << "; average: " << average.mean << std::endl;
      }
    }
  }

  pool.kill();

  return 0;
}