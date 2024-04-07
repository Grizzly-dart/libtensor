#include <algorithm>
#include <cmath>
#include <execution>
#include <stdint.h>
#include <stdlib.h>

enum ArithMode : uint8_t {
  ewise,
  rwise,
  scalar,
};

/*
template <typename O, typename I1, typename I2>
const char *tcPlus(
    O *out, const I1 *inp1, const I2 *inp2, uint64_t nel, ArithMode mode,
    uint8_t flip
) {


  if (inp2 != nullptr) {
    std::transform(
        std::execution::par_unseq, inp1, inp1 + nel, inp2, out,
        [](I1 a, I2 b) { return a + b; }
    );
  } else {
    std::transform(
        std::execution::par_unseq, inp1, inp1 + nel, out,
        [scalar](I1 a) { return a + *scalar; }
    );
  }
  return nullptr;
}
*/