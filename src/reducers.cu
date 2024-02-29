#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "reducers.hpp"

template <typename T>
__device__ void Mean<T>::consume(T sample) {
  n++;
  auto delta = sample - mean;
  mean += delta / n;
}

template <typename T>
__device__ void Mean<T>::merge(const Mean& other) {
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

template <typename T>
__device__ Mean<T> Mean<T>::shfl_down(int offset) {
  Mean<T> other;
  other.mean = __shfl_down_sync(0xffffffff, mean, offset);
  other.n = __shfl_down_sync(0xffffffff, n, offset);
  return other;
}

template <typename T>
__device__ void Variance<T>::consume(T sample) {
  n++;
  auto delta = sample - mean;
  mean += delta / n;
  m2 += delta * (sample - mean);
}

template <typename T>
__device__ void Variance<T>::merge(const Variance<T>& other) {
  if (other.n == 0) {
    return;
  }
  if (n == 0) {
    mean = other.mean;
    n = other.n;
    m2 = other.m2;
    return;
  }

  n = n + other.n;
  auto delta = other.mean - mean;
  mean += delta * other.n / n;
  m2 += other.m2 + delta * delta * n * other.n / (n + other.n);
  printf("n: %d, mean: %f, m2: %f\n", n, mean, m2);
}

template <typename T>
__device__ Variance<T> Variance<T>::shfl_down(int offset) {
  Variance<T> other;
  other.mean = __shfl_down_sync(0xffffffff, mean, offset);
  other.n = __shfl_down_sync(0xffffffff, n, offset);
  other.m2 = __shfl_down_sync(0xffffffff, m2, offset);
  return other;
}
