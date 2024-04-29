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
const char *tcSumNaive(O *out, const I *inp, uint64_t nel) {
  O ret = 0;
  for (uint64_t i = 0; i < nel; i++) {
    ret += inp[i];
  }
  *out = ret;
  return nullptr;
}

/*
template <typename I>
const char *tcNegSlow(I *out, const I *inp, uint64_t nel) {
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
            v = -v;
            memcpy(out + lane, &v, laneSize * sizeof(I));
          }
        }
    );
  }

  for (uint16_t i = 0; i < concurrency; i++) {
    futures[i].wait();
  }

  return nullptr;
}
 */

template <typename O, typename I>
void check(O out, const I *inp, uint64_t nel) {
  O res;
  tcSumNaive(&res, inp, nel);
  O diff = std::abs(res - out);
  if (diff > nel * 1e-5) {
    std::cout << "Mismatch => " << res << " != " << out << "; " << diff
              << std::endl;
  }
}

int main() {
  using I = float;
  using O = float;
  const uint64_t size = 2048 * 10000;
  I *inp = new (std::align_val_t(128)) I[size];
  O out;

  for (uint64_t i = 0; i < size; i++) {
    if constexpr (std::is_floating_point<I>::value)
      inp[i] = drand48();
    else
      inp[i] = static_cast<I>(i);
  }

  int64_t timeSum = 0;
  const int64_t iterations = 10;
  for (uint8_t i = 0; i < iterations; i++) {
    out = 0;
    steady_clock::time_point begin = steady_clock::now();
    tcSumNaive<O, I>(&out, inp, size);
    steady_clock::time_point end = steady_clock::now();
    std::cout
        << "Naive:   "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    auto timeA =
        chrono::duration_cast<chrono::microseconds>(end - begin).count();
    check(out, inp, size);

    out = 0;
    begin = steady_clock::now();
    tcSum<O, I>(&out, inp, size);
    end = steady_clock::now();
    std::cout
        << "AutoVec: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    auto timeB =
        chrono::duration_cast<chrono::microseconds>(end - begin).count();
    check(out, inp, size);
    timeSum += timeA - timeB;
    std::cout << "---------" << std::endl;
  }
  std::cout << "Time diff: " << timeSum / iterations << "us" << std::endl;

  delete[] inp;

  return 0;
}