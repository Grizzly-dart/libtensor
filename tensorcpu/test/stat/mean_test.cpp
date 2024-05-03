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
#include "tensorcpu.hpp"
#include "typed_array.hpp"
#include "stat.hpp"

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename O, typename I>
const char *mean_naive(O *out, const I *inp, uint64_t nel) {
  O ret = 0;
  for (uint64_t i = 0; i < nel; i++) {
    ret += inp[i];
  }
  *out = ret / nel;
  return nullptr;
}

template <typename O, typename I>
void check(O out, const I *inp, uint64_t nel) {
  O res;
  mean_naive(&res, inp, nel);
  O diff = std::abs(res - out);
  if (diff > nel * 1e-5) {
    std::cout << "Mismatch => " << res << " != " << out << "; " << diff
              << std::endl;
  }
}

int main() {
  using I = float;
  using O = float;
  const uint64_t size = 2048 * 100;
  I *inp = new (std::align_val_t(128)) I[size];
  O out;

  for (uint64_t i = 0; i < size; i++) {
    if constexpr (std::is_floating_point<I>::value)
      inp[i] = drand48();
    else
      inp[i] = static_cast<I>(i);
  }

  steady_clock::time_point begin, end;
  Mean<double, int64_t> averageNaive, averageAutoVec, averageOptim;
  int64_t dur;
  int64_t timeSum = 0;
  const int64_t iterations = 10;
  for (uint8_t i = 0; i < iterations; i++) {
    {
      out = 0;
      begin = steady_clock::now();
      mean_naive(&out, inp, size);
      end = steady_clock::now();
      dur = chrono::duration_cast<chrono::microseconds>(end - begin).count();
      std::cout << "Naive:     " << dur << "us" << std::endl;
      check(out, inp, size);
    }

    {
      out = 0;
      begin = steady_clock::now();
      mean_1thread(&out, inp, size);
      end = steady_clock::now();
      dur = chrono::duration_cast<chrono::microseconds>(end - begin).count();
      std::cout << "1thread:   " << dur << "us" << std::endl;
      check(out, inp, size);
    }
  }

  delete[] inp;

  return 0;
}