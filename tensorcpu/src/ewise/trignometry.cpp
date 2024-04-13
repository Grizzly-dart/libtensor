#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <stdfloat>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename O, typename I, UnaryOp op>
void tcTrignometry(O *out, I *inp1, uint64_t nel) {
  size_t width = stdx::native_simd<I>::size();
  printf("width: %zu\n", width);
  auto i1 = Simd<I>(inp1, width, nel);
  auto o = Simd<O>(out, width, nel);

  std::for_each(
      std::execution::par, i1.countBegin(), i1.countEnd(),
      [&i1, &o](uint64_t i) {
        using std::pow;
        auto elements = i1.calcRemainingElements(i);
        std::vector<I> a, b;
        i1.load(i, a);
        std::vector<O> res;
        res.resize(elements);

#pragma GCC ivdep
        for (int j = 0; j < elements; j++) {
          if constexpr (op == UnaryOp::Log) {
            res[j] = std::log(a[j]);
          } else if constexpr (op == UnaryOp::Exp) {
            res[j] = std::exp(a[j]);
          } else if constexpr (op == UnaryOp::Sin) {
            res[j] = std::sin(a[j]);
          } else if constexpr (op == UnaryOp::Cos) {
            res[j] = std::cos(a[j]);
          } else if constexpr (op == UnaryOp::Tan) {
            res[j] = std::tan(a[j]);
          } else if constexpr (op == UnaryOp::Sinh) {
            res[j] = std::sinh(a[j]);
          } else if constexpr (op == UnaryOp::Cosh) {
            res[j] = std::cosh(a[j]);
          } else if constexpr (op == UnaryOp::Tanh) {
            res[j] = std::tanh(a[j]);
          }
        }
        o.store(i, res);
      }
  );
}

// #define TRIGNOMETRY(O, I, op)
