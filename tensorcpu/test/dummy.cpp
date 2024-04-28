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
const char *tcCast(O *out, I *inp, uint64_t nel) {
  constexpr size_t laneSize = simdSize<I>();
  uint16_t concurrency = std::thread::hardware_concurrency();
  auto iAccessor = Accessor<I>(inp, laneSize, nel);
  auto oAccessor = Accessor<O>(out, laneSize, nel);
  uint64_t totalLanes = (nel + laneSize - 1) / laneSize;
  uint64_t lanesPerThread = std::max(
      uint64_t((totalLanes + concurrency - 1) / concurrency), uint64_t(1)
  );
  std::vector<std::future<void>> futures(concurrency);

  for (uint16_t threadNum = 0; threadNum < concurrency; threadNum++) {
    futures[threadNum] = std::async(
        std::launch::async,
        [threadNum, lanesPerThread, iAccessor, oAccessor]() {
          stdx::fixed_size_simd<I, laneSize> a;
          uint64_t last = (threadNum + 1) * lanesPerThread;
          for (uint64_t lane = threadNum * lanesPerThread; lane < last;
               lane++) {
            oAccessor.store(lane, iAccessor.load(lane, a));
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
const char *tcCastPlain(O *out, const I *inp, uint64_t nel) {
  /*for(uint64_t i = 0; i < nel; i++) {
    out[i] = static_cast<O>(inp[i]);
  }*/
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return static_cast<O>(a);
  });
  return nullptr;
}

template <typename O, typename I>
void check(const O *out, const I *inp, uint64_t nel) {
  for (uint64_t i = 0; i < nel; i++) {
    O res = static_cast<O>(inp[i]);
    O diff = std::abs(res - out[i]);
    if (diff > res * 1e-3) {
      std::cout << "Mismatch at " << i << " => " << res << " != " << out[i]
                << "; " << diff << std::endl;
      break;
    }
  }
}

int main() {
  using I = float;
  using O = uint8_t;
  const uint64_t size = 2048 * 10000;
  I *inp = new(std::align_val_t(128)) I[size];
  O *out = new(std::align_val_t(128)) O[size];

  for (uint64_t i = 0; i < size; i++) {
    inp[i] = static_cast<I>(i);
  }

  for (uint8_t i = 0; i < 10; i++) {
    memset(out, 0, size * sizeof(O));
    steady_clock::time_point begin = steady_clock::now();
    tcCast<O, I>(out, inp, size);
    steady_clock::time_point end = steady_clock::now();
    std::cout
        << "SIMDed: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check(out, inp, size);

    memset(out, 0, size * sizeof(O));
    begin = steady_clock::now();
    tcCastPlain<O, I>(out, inp, size);
    end = steady_clock::now();
    std::cout
        << "Plain: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check(out, inp, size);
  }

  delete[] inp;
  delete[] out;

  return 0;
}