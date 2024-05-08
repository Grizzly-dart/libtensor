#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <stdfloat>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

#if 0
template <typename O, typename I>
static Kernel makeKernel(
    O *out, I *inp, Accessor<I> &i1, uint16_t width, std::function<O(I)> op
) {
  return [&out, &inp, &width, &i1, &op](uint64_t simdI) {
    auto elements = i1.calcRemainingElements(simdI);
    auto start = simdI * width;
#pragma GCC ivdep
    for (uint64_t i = 0; i < elements; i++) {
      out[start] = op(inp[start]);
    }
  };
}

template <typename O, typename I>
const char *tcFUnary(O *out, I *inp, FUnaryOp op, uint64_t nel) {
  static_assert(std::is_floating_point<O>::value, "O must be floating point");

  size_t width = stdx::native_simd<I>::size();
  printf("width: %zu\n", width);
  auto i1 = Accessor<I>(inp, width, nel);

  std::function<O(I)> func;
  switch (op) {
  case FUnaryOp::Log:
    func = [](I a) { return std::log(a); };
    break;
  case FUnaryOp::Exp:
    func = [](I a) { return std::exp(a); };
    break;
  case FUnaryOp::Expm1:
    func = [](I a) { return std::expm1(a); };
    break;
  case FUnaryOp::Sqrt:
    func = [](I a) { return std::sqrt(a); };
    break;
  case FUnaryOp::Sin:
    func = [](I a) { return std::sin(a); };
    break;
  case FUnaryOp::Cos:
    func = [](I a) { return std::cos(a); };
    break;
  case FUnaryOp::Tan:
    func = [](I a) { return std::tan(a); };
    break;
  case FUnaryOp::Sinh:
    func = [](I a) { return std::sinh(a); };
    break;
  case FUnaryOp::Cosh:
    func = [](I a) { return std::cosh(a); };
    break;
  case FUnaryOp::Tanh:
    func = [](I a) { return std::tanh(a); };
    break;
  case FUnaryOp::ASin:
    func = [](I a) { return std::asin(a); };
    break;
  case FUnaryOp::ACos:
    func = [](I a) { return std::acos(a); };
    break;
  case FUnaryOp::ATan:
    func = [](I a) { return std::atan(a); };
    break;
  case FUnaryOp::ASinh:
    func = [](I a) { return std::asinh(a); };
    break;
  case FUnaryOp::ACosh:
    func = [](I a) { return std::acosh(a); };
    break;
  case FUnaryOp::ATanh:
    func = [](I a) { return std::atanh(a); };
    break;
  default:
    return "Invalid unary operation";
  }

  std::for_each(
      std::execution::par, i1.countBegin(), i1.countEnd(),
      makeKernel<O, I>(out, inp, i1, width, func)
  );
  return nullptr;
}

template <typename O, typename I>
const char *tcFUnaryPlain(O *out, I *inp, FUnaryOp op, uint64_t nel) {
  std::function<O(I)> func;
  switch (op) {
  case FUnaryOp::Log:
    func = [](I a) { return std::log(a); };
    break;
  case FUnaryOp::Exp:
    func = [](I a) { return std::exp(a); };
    break;
  case FUnaryOp::Expm1:
    func = [](I a) { return std::expm1(a); };
    break;
  case FUnaryOp::Sqrt:
    func = [](I a) { return std::sqrt(a); };
    break;
  case FUnaryOp::Sin:
    func = [](I a) { return std::sin(a); };
    break;
  case FUnaryOp::Cos:
    func = [](I a) { return std::cos(a); };
    break;
  case FUnaryOp::Tan:
    func = [](I a) { return std::tan(a); };
    break;
  case FUnaryOp::Sinh:
    func = [](I a) { return std::sinh(a); };
    break;
  case FUnaryOp::Cosh:
    func = [](I a) { return std::cosh(a); };
    break;
  case FUnaryOp::Tanh:
    func = [](I a) { return std::tanh(a); };
    break;
  case FUnaryOp::ASin:
    func = [](I a) { return std::asin(a); };
    break;
  case FUnaryOp::ACos:
    func = [](I a) { return std::acos(a); };
    break;
  case FUnaryOp::ATan:
    func = [](I a) { return std::atan(a); };
    break;
  case FUnaryOp::ASinh:
    func = [](I a) { return std::asinh(a); };
    break;
  case FUnaryOp::ACosh:
    func = [](I a) { return std::acosh(a); };
    break;
  case FUnaryOp::ATanh:
    func = [](I a) { return std::atanh(a); };
    break;
  default:
    return "Invalid unary operation";
  }

  std::transform(std::execution::par_unseq, inp, inp + nel, out, func);

  return nullptr;
}

#define FUNARY(O, I)                                                           \
  template const char *tcFUnary<O, I>(                                         \
      O * out, I * inp, FUnaryOp op, uint64_t nel                              \
  );

#define FUNARY_PLAIN(O, I)                                                     \
  template const char *tcFUnaryPlain<O, I>(                                    \
      O * out, I * inp, FUnaryOp op, uint64_t nel                              \
  );

#define UNWIND2_2ND(T, OP)                                                     \
  OP(T, int8_t)                                                                \
  OP(T, int16_t)                                                               \
  OP(T, int32_t)                                                               \
  OP(T, int64_t)                                                               \
  OP(T, uint8_t)                                                               \
  OP(T, uint16_t)                                                              \
  OP(T, uint32_t)                                                              \
  OP(T, uint64_t)                                                              \
  OP(T, float)                                                                 \
  OP(T, double)

UNWIND2_2ND(float, FUNARY)
UNWIND2_2ND(double, FUNARY)

UNWIND2_2ND(float, FUNARY_PLAIN)
UNWIND2_2ND(double, FUNARY_PLAIN)
#endif