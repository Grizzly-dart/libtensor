//
// Created by Teja Gudapati on 2024-05-06.
//

#include <algorithm>
#include <cstdint>
#include <execution>
#include <cmath>

template <typename T> void sigmoid_stdalgo(T *out, const T *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](T x) {
    return 1 / (1 + std::exp(-x));
  });
}

#define TCSIGMOIDSTDALGO(T) template void sigmoid_stdalgo(T *out, const T *inp, uint64_t nel);

TCSIGMOIDSTDALGO(float)
TCSIGMOIDSTDALGO(double)

template <typename T> void sigmoid_parallel(T *out, const T *inp, uint64_t nel) {
  // TODO
}