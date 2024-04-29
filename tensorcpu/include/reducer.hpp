//
// Created by Teja Gudapati on 2024-04-11.
//

#ifndef TENSORC_REDUCER_HPP
#define TENSORC_REDUCER_HPP

#include <cstdint>
#include <future>
#include "typed_array.hpp"

template <typename O, typename I> class Mean {
public:
  O mean = 0;
  uint32_t n = 0;

  Mean() = default;

  Mean(I mean, uint32_t n) : mean(mean), n(n) {}

  void consume(I sample) {
    n++;
    auto delta = sample - mean;
    mean += delta / n;
  }

  void merge(const Mean<O, I> &other) {
    if (other.n == 0) {
      return;
    }
    if (n == 0) {
      mean = other.mean;
      n = other.n;
      return;
    }

    n = n + other.n;
    auto delta = other.mean - mean;
    mean += delta * other.n / n;
  }
};

template <typename O, typename I> class MeanSimd {
public:
  static constexpr uint32_t sizeSimd = std::min(simdSize<I>(), simdSize<O>());
  typedef I ISimdType __attribute__((vector_size(sizeof(I) * sizeSimd)));
  typedef O OSimdType __attribute__((vector_size(sizeof(I) * sizeSimd)));

  OSimdType runningMean = {0};
  uint32_t n = 0;

  void consumeSimd(ISimdType &input) {
    n++;
    auto delta = __builtin_convertvector(input, OSimdType) - runningMean;
    runningMean += delta / O(n);
  }

  void merge(const MeanSimd<O, I> &other) {
    if (other.n == 0) {
      return;
    }

    if (n == 0) {
      runningMean = other.runningMean;
      n = other.n;
      return;
    }

    n = n + other.n;
    auto delta = other.runningMean - runningMean;
    runningMean += delta * O(other.n) / O(n);
  }

  O meanScalar() {
    O mean = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < sizeSimd; i++) {
      count += n;
      auto delta = O(runningMean[i]) - mean;
      mean += delta / count;
    }
    return mean;
  }

  Mean<O, I> mean() {
    O mean = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < sizeSimd; i++) {
      count += n;
      auto delta = O(runningMean[i]) - mean;
      mean += delta / count;
    }
    return {mean, count};
  }
};

template<typename F>
void parallelSimdFold(
    uint64_t nel, uint64_t laneSize,
    const std::function<void(uint64_t, uint64_t, F&)> &kernel, std::vector<F> &results
) {
  uint64_t totalLanes = nel / laneSize;
  uint64_t concurrency = std::thread::hardware_concurrency();
  uint64_t lanesPerThread;
  if (concurrency > totalLanes) {
    concurrency = totalLanes;
    lanesPerThread = 1;
  } else {
    lanesPerThread = (totalLanes + concurrency - 1) / concurrency;
  }
  std::vector<std::future<F>> futures(concurrency);

  for (uint64_t threadNum = 0; threadNum < concurrency; threadNum++) {
    uint64_t start = threadNum * lanesPerThread * laneSize;
    uint64_t last = (threadNum + 1) * lanesPerThread;
    if (last > totalLanes) {
      last = totalLanes;
    }
    last *= laneSize;

    futures[threadNum] = std::async(
        std::launch::async,
        [start, last, kernel]() { return kernel(start, last); }
    );
  }

  results.resize(concurrency);
  for (uint64_t i = 0; i < concurrency; i++) {
    results[i] = futures[i].get();
  }
}

#endif // TENSORC_REDUCER_HPP
