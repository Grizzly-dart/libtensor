#include <algorithm>
#include <cmath>
#include <execution>
#include <future>
#include <limits>
#include <stdfloat>

#include "macro_unwind.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename O, typename I>
const char *tcCast(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return static_cast<O>(a);
  });
  return nullptr;
}

#define TCCAST(O, I)                                                           \
  template const char *tcCast(O *out, const I *inp, uint64_t nel);

// TODO use bigger intermediate type to decrease library size
UNWIND2_ALL_TYPES(TCCAST)

template <typename I> const char *tcAbs(I *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return a >= 0 ? a : -a;
  });
  return nullptr;
}

#define TCABS(I) template const char *tcAbs(I *out, const I *inp, uint64_t nel);

UNWIND1_SIGNED(TCABS)

template <typename O, typename I>
const char *tcNeg(O *out, const I *inp, uint64_t nel) {
  std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
    return -a;
  });
  return nullptr;
}

#define TCNEG(O, I)                                                            \
  template const char *tcNeg(O *out, const I *inp, uint64_t nel);

UNWIND2_SIGNED(TCNEG)