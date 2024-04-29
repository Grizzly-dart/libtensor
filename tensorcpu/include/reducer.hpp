//
// Created by Teja Gudapati on 2024-04-11.
//

#ifndef TENSORC_REDUCER_HPP
#define TENSORC_REDUCER_HPP

#include "typed_array.hpp"
#include <cstdint>
#include <future>

template <typename O, typename I> class Mean {
public:
  O mean = 0;
  uint64_t n = 0;

  Mean() = default;

  Mean(I mean, uint64_t n) : mean(mean), n(n) {}

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
  uint64_t n = 0;

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
    Mean<O, I> ret;
    for (uint32_t i = 0; i < sizeSimd; i++) {
      ret.merge({O(runningMean[i]), n});
    }
    return ret;
  }

  Mean<O, I> materialize() {
    O mean = 0;
    uint64_t count = 0;
    for (uint32_t i = 0; i < sizeSimd; i++) {
      count += n;
      auto delta = O(runningMean[i]) - mean;
      mean += delta / count;
    }
    return {mean, count};
  }
};

template <typename O, typename I> class Variance {
public:
  O mean = 0;
  uint64_t n = 0;
  O m2 = 0;

  Variance() = default;

  Variance(O mean, uint64_t n, O m2) : mean(mean), n(n), m2(m2) {}

  void consume(I sample) {
    n++;
    auto delta = sample - mean;
    mean += delta / n;
    m2 += delta * (sample - mean);
  }

  void merge(const Variance<O, I> &other) {
    if (other.n == 0) {
      return;
    }
    if (n == 0) {
      mean = other.mean;
      m2 = other.m2;
      n = other.n;
      return;
    }

    auto newN = n + other.n;
    auto delta = other.mean - mean;
    mean += delta * other.n / newN;
    m2 += other.m2 + delta * delta * n * other.n / newN;
    n = newN;
  }
};

template <typename O, typename I> class VarianceSimd {
public:
  static constexpr uint32_t sizeSimd = std::min(simdSize<I>(), simdSize<O>());
  typedef I ISimdType __attribute__((vector_size(sizeof(I) * sizeSimd)));
  typedef O OSimdType __attribute__((vector_size(sizeof(I) * sizeSimd)));

  OSimdType runningMean = {0};
  uint64_t n = 0;
  OSimdType m2 = {0};

  void consumeSimd(ISimdType &input) {
    n++;
    OSimdType convInput = __builtin_convertvector(input, OSimdType);
    auto delta = convInput - runningMean;
    runningMean += delta / O(n);
    m2 += delta * (convInput - runningMean);
  }

  void merge(const VarianceSimd<O, I> &other) {
    if (other.n == 0) {
      return;
    }
    if (n == 0) {
      runningMean = other.runningMean;
      m2 = other.m2;
      n = other.n;
      return;
    }

    auto newN = n + other.n;
    auto delta = other.runningMean - runningMean;
    runningMean += delta * O(other.n) / O(newN);
    m2 += other.m2 + delta * delta * O(n * other.n) / O(newN);
    n = newN;
  }

  Variance<O, I> materialize() {
    Variance<O, I> ret;
    for (uint32_t i = 0; i < sizeSimd; i++) {
      ret.merge({O(runningMean[i]), n, O(m2[i])});
    }
    return ret;
  }
};

template <typename O>
void parallelSimdFold(
    uint64_t nel, uint64_t laneSize,
    const std::function<O(uint64_t, uint64_t)> &kernel, O &result,
    std::function<void(O &, O)> merge
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
  std::vector<std::future<O>> futures(concurrency);

  for (uint64_t threadNum = 0; threadNum < concurrency; threadNum++) {
    uint64_t start = threadNum * lanesPerThread * laneSize;
    uint64_t last = (threadNum + 1) * lanesPerThread;
    if (last > totalLanes) {
      last = totalLanes;
    }
    last *= laneSize;

    futures[threadNum] =
        std::async(std::launch::async, [start, last, kernel]() {
          return kernel(start, last);
        });
  }

  for (uint64_t i = 0; i < concurrency; i++) {
    merge(result, futures[i].get());
  }
}

#endif // TENSORC_REDUCER_HPP
