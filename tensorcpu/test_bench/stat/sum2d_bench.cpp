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

namespace stdx = std::experimental;

namespace chrono = std::chrono;
using std::chrono::steady_clock;

template <typename O, typename I>
const char *tcSum2dNaive(O *out, const I *inp, uint64_t rows, uint64_t cols) {
  for (uint64_t row = 0; row < rows; row++) {
    O sum = 0;
    for (uint64_t col = 0; col < cols; col++) {
      sum += inp[col];
    }
    out[row] = sum;
    inp += cols;
  }
  return nullptr;
}

template <typename O, typename I>
void check(O *out, const I *inp, uint64_t rows, uint64_t cols) {
  for (uint64_t row = 0; row < rows; row++) {
    O res = 0;
    for (uint64_t col = 0; col < cols; col++) {
      res += inp[col];
    }
    O diff = std::abs(res - out[row]);
    if (diff > cols * 1e-5) {
      std::cout << "Mismatch @"
                << row << " => "
                << res << " != " << out[row] << "; " << diff
                << std::endl;
      break;
    }
    inp += cols;
  }
}

int main() {
  using I = float;
  using O = float;
  const uint64_t rows = 1024;
  const uint64_t cols = 1024 * 2;
  const uint64_t size = rows * cols;
  I *inp = new (std::align_val_t(128)) I[size];
  O *out = new (std::align_val_t(128)) O[rows];

  for (uint64_t i = 0; i < size; i++) {
    if constexpr (std::is_floating_point<I>::value)
      inp[i] = drand48();
    else
      inp[i] = static_cast<I>(i);
  }

  int64_t timeSum = 0;
  const int64_t iterations = 1;
  for (uint8_t i = 0; i < iterations; i++) {
    memset(out, 0, rows * sizeof(O));
    steady_clock::time_point begin = steady_clock::now();
    tcSum2dNaive<O, I>(out, inp, rows, cols);
    steady_clock::time_point end = steady_clock::now();
    std::cout
        << "Naive:   "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check(out, inp, rows, cols);
    auto timeA =
        chrono::duration_cast<chrono::microseconds>(end - begin).count();

    memset(out, 0, rows * sizeof(O));
    begin = steady_clock::now();
    tcSum2d<O, I>(out, inp, rows, cols);
    end = steady_clock::now();
    std::cout
        << "AutoVec: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    auto timeB =
        chrono::duration_cast<chrono::microseconds>(end - begin).count();
    check(out, inp, rows, cols);
    timeSum += timeA - timeB;
    std::cout << "---------" << std::endl;
  }
  std::cout << "Time diff: " << timeSum / iterations << "us" << std::endl;

  delete[] inp;

  return 0;
}