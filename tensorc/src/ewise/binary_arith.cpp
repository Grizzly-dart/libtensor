#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <execution>

template<typename O, typename I1, typename I2>
const char* tcPlus(O* out, const I1* inp1, const I2* inp2, const I2* scalar, uint64_t nel, uint8_t flip) {
  if((inp2 == nullptr) == (scalar == nullptr)) {
    return "Both inp2 and scalar cannot be null or non-null at the same time";
  }
  if(inp2 != nullptr) {
    std::transform(std::execution::par_unseq, inp1, inp1 + nel, inp2, out, [](I1 a, I2 b) { return a + b; });
  } else {
    if(!flip) {
      std::transform(std::execution::par_unseq, inp1, inp1 + nel, out, [scalar](I1 a) { return a + *scalar; });
    } else {
      std::transform(std::execution::par_unseq, inp1, inp1 + nel, out, [scalar](I1 a) { return *scalar + a; });
    }
  }
  return nullptr;
}

#include "binary_arith_gen.inc"