#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <stdfloat>

#include "binaryarith.hpp"
#include "tensorcpu.hpp"
#include "test_common.hpp"
#include "thread_pool.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

std::map<uint8_t, std::string> opNames = {
    {0, "Plus"}, {1, "Minus"}, {2, "Mul"}, {3, "Div"}, {4, "Pow"}
};

template <typename O, typename I1, typename I2>
void check(
    O *out, I1 *inp1, I2 *inp2, BinaryOp op, uint64_t nel, bool flip,
    const char *name, uint64_t iteration
) {
  for (uint64_t i = 0; i < nel; i++) {
    O v;
    if (op == BinaryOp::Plus) {
      v = inp1[i] + inp2[i];
    } else if (op == BinaryOp::Minus) {
      if (!flip) {
        v = inp1[i] - inp2[i];
      } else {
        v = inp2[i] - inp1[i];
      }
    } else if (op == BinaryOp::Mul) {
      v = inp1[i] * inp2[i];
    } else if (op == BinaryOp::Div) {
      if (!flip) {
        v = inp1[i] / inp2[i];
      } else {
        v = inp2[i] / inp1[i];
      }
    } else if (op == BinaryOp::Pow) {
      if (!flip) {
        O tmp = std::pow(inp1[i], inp2[i]);
        v = static_cast<O>(tmp);
      } else {
        O tmp = std::pow(inp2[i], inp1[i]);
        v = static_cast<O>(tmp);
      }
    }
    O diff = std::abs(v - out[i]);
    if (diff > 1e-3) {
      std::cerr << "In " << name << "; size = " << nel
                << "; op = " << opNames[op] << "; Iteration: " << iteration
                << "; Mismatch @" << i << "; "
                << "a: " << +inp1[i] << " b: " << +inp2[i] << "; " << +out[i]
                << " != " << +v << "; diff = " << diff << std::endl;
      exit(1);
    }
  }
  std::cerr << "In " << name << "; size = " << nel << "; op = " << opNames[op]
            << "; Iteration: " << iteration << "; OK" << std::endl;
}

int main() {
  using I = float;
  using O = float;

  std::vector<uint64_t> sizes;
  make1dBenchSizes(sizes, std::min(simdSize<O>(), simdSize<I>()));

  uint8_t flip = 0;
  for (uint64_t size : sizes) {
    std::unique_ptr<I> inp1(new I[size]);
    std::unique_ptr<I> inp2(new I[size]);
    std::unique_ptr<O> out(new O[size]);
    fillRand(inp1.get(), size);
    fillRand(inp2.get(), size);
    for (BinaryOp op : {
             BinaryOp::Plus, /*BinaryOp::Minus, BinaryOp::Mul, BinaryOp::Div,
             BinaryOp::Pow*/
         }) {
      {
        memset(out.get(), 0, size * sizeof(O));
        binaryarith_parallel<I>(
            out.get(), inp1.get(), inp2.get(), op, size, flip
        );
        check(out.get(), inp1.get(), inp2.get(), op, size, flip, "parallel", 0);
      }
      {
        memset(out.get(), 0, size * sizeof(O));
        binaryarith_1thread<I>(
            out.get(), inp1.get(), inp2.get(), op, size, flip
        );
        check(out.get(), inp1.get(), inp2.get(), op, size, flip, "1thread", 0);
      }
      {
        memset(out.get(), 0, size * sizeof(O));
        binaryarith_casted_1thread<double>(
            out.get(), inp1.get(), inp2.get(), op, size, flip, f32Id, f32Id,
            f32Id
        );
        check(
            out.get(), inp1.get(), inp2.get(), op, size, flip, "1thread_casted",
            0
        );
      }
    }
  }

  pool.kill();

  return 0;
}
