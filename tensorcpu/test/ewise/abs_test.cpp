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

template <typename I>
const char *tcAbsNaive(I *out, const I *inp, uint64_t nel) {
  for (uint64_t i = 0; i < nel; i++) {
    out[i] = inp[i] >= 0 ? inp[i] : -inp[i];
  }
  return nullptr;
}

template <typename I>
const char *tcAbsSlow(I *out, const I *inp, uint64_t nel) {
  constexpr size_t laneSize = simdSize<I>();
  uint16_t concurrency = std::thread::hardware_concurrency();
  uint64_t totalLanes = (nel + laneSize - 1) / laneSize;
  uint64_t lanesPerThread = std::max(
      uint64_t((totalLanes + concurrency - 1) / concurrency), uint64_t(1)
  );
  std::vector<std::future<void>> futures(concurrency);

  for (uint16_t threadNum = 0; threadNum < concurrency; threadNum++) {
    futures[threadNum] = std::async(
        std::launch::async,
        [threadNum, lanesPerThread, out, inp, laneSize]() {
          uint64_t start = threadNum * lanesPerThread * laneSize;
          uint64_t last = (threadNum + 1) * lanesPerThread * laneSize;
          typedef I V __attribute__((vector_size(laneSize * sizeof(I))));
          V v;
          for (uint64_t lane = start; lane < last; lane += laneSize) {
            memcpy(&v, inp + lane, laneSize * sizeof(I));
            v = v >= 0 ? v : -v;
            memcpy(out + lane, &v, laneSize * sizeof(I));
          }
        }
    );
  }

  for (uint16_t i = 0; i < concurrency; i++) {
    futures[i].get();
  }
  return nullptr;
}

template <typename O, typename I>
void check(const O *out, const I *inp, uint64_t nel) {
  for (uint64_t i = 0; i < nel; i++) {
    O res = static_cast<O>(inp[i]);
    O diff = std::abs(res - out[i]);
    if (diff > 1e-3) {
      std::cout << "Mismatch at " << i << " => " << res << " != " << out[i]
                << "; " << diff << std::endl;
      break;
    }
  }
}

int main() {
  using I = float;
  const uint64_t size = 2048 * 10000;
  I *inp = new (std::align_val_t(128)) I[size];
  I *out = new (std::align_val_t(128)) I[size];

  for (uint64_t i = 0; i < size; i++) {
    inp[i] = static_cast<I>(-i);
  }

  int64_t timeSum = 0;
  const int64_t iterations = 10;
  for (uint8_t i = 0; i < iterations; i++) {
    memset(out, 0, size * sizeof(I));
    steady_clock::time_point begin = steady_clock::now();
    tcAbsSlow<I>(out, inp, size);
    steady_clock::time_point end = steady_clock::now();
    auto timeA =
        chrono::duration_cast<chrono::microseconds>(end - begin).count();
    std::cout
        << "Naive: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check(out, inp, size);

    memset(out, 0, size * sizeof(I));
    begin = steady_clock::now();
    tcAbs<I>(out, inp, size);
    end = steady_clock::now();
    auto timeB =
        chrono::duration_cast<chrono::microseconds>(end - begin).count();
    std::cout
        << "AutoVec: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check(out, inp, size);
    timeSum += timeA - timeB;
    std::cout << "---------" << std::endl;
  }
  std::cout << "Time diff: " << timeSum / iterations << "us" << std::endl;

  delete[] inp;
  delete[] out;

  return 0;
}