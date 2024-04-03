#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>
#include <limits>

template <typename O, typename I>
const char *tcCast(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return static_cast<O> a;
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcNeg(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return -a;
  });
  return nullptr;
}

template <typename T>
const char *tcAbs(T *out, const T *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](T a) {
    if (inp[i] == std::numeric_limits<T>::min()) {
      out[i] = std::numeric_limits<T>::max();
    } else {
      out[i] = inp[i] >= 0 ? inp[i] : -inp[i];
    }
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcExp(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::exp(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcLog(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::log(a);
  });
  return nullptr;
}