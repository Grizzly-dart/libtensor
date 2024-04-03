#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>
#include <limits>

template <typename O, typename I>
const char *tcSin(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::sin(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcCos(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::cos(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcTan(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::tan(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcSinh(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::sinh(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcCosh(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::cosh(a);
  });
  return nullptr;
}

template <typename O, typename I>
const char *tcTanh(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return std::tanh(a);
  });
  return nullptr;
}