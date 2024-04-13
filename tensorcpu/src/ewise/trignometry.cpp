#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>

#include "tensorcpu.hpp"
#include "typed_array.hpp"



template <typename O, typename I, UnaryOp op>
void tcBinaryArith(
        O *out, I *inp1, uint64_t nel
) {
    size_t width = std::min(
            std::min(stdx::native_simd<O>::size(), stdx::native_simd<I>::size()),
            stdx::native_simd<I>::size()
    );
    printf("width: %zu\n", width);
    auto i1 = Simd<I>(inp1, width, nel);
    std::unique_ptr<ISimd<I>> i2;
    uint64_t snel = i2broadcaster.nel();
    if (snel == 0) {
        i2 = std::make_unique<Simd<I>>(inp2, width, nel);
    } else if (snel == 1) {
        i2 = std::make_unique<SameSimd<I>>(SameSimd<I>(*inp2, width, nel));
    } else {
        i2 = std::make_unique<RwiseSimd<I>>(inp2, width, nel, i2broadcaster);
    }
    auto o = Simd<O>(out, width, nel);

    std::for_each(
            std::execution::par, i1.countBegin(), i1.countEnd(),
            [&i1, &i2, &o, flip](uint64_t i) {
                stdx::native_simd<I> a, b;
                if constexpr (op == BinaryOp::Plus) {
                    o.store(i, i1.load(i, a) + i2->load(i, b));
                } else if constexpr (op == BinaryOp::Minus) {
                    if (!flip) {
                        o.store(i, i1.load(i, a) - i2->load(i, b));
                    } else {
                        o.store(i, i2->load(i, b) - i1.load(i, a));
                    }
                } else if constexpr (op == BinaryOp::Mul) {
                    o.store(i, i1.load(i, a) * i2->load(i, b));
                } else if constexpr (op == BinaryOp::Div) {
                    if (!flip) {
                        o.store(i, i1.load(i, a) / i2->load(i, b));
                    } else {
                        o.store(i, i2->load(i, b) / i1.load(i, a));
                    }
                } else if constexpr (op == BinaryOp::Pow) {
                    auto elements = i1.calcRemainingElements(i);
                    using std::pow;
                    if (!flip) {
                        for (int j = 0; j < elements; j++) {
                            uint64_t ind = i * i1.width + j;
                            o.set(ind, pow(i1.get(ind), i2->get(ind)));
                        }
                    } else {
                        for (int j = 0; j < elements; j++) {
                            uint64_t ind = i * i1.width + j;
                            o.set(ind, pow(i2->get(ind), i1.get(ind)));
                        }
                    }
                }
            }
    );
}

/*
template<typename O, typename I>
const char *tcSin(O *__restrict__ out, const I *__restrict__ inp, uint64_t nel) {
    std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
        return std::sin(a);
    });

#pragma GCC ivdep
    for (uint64_t i = 0; i < nel; i++) {
        out[i] = std::sin(inp[i]);
    }
    return nullptr;
}

template const char *tcSin<double, double>(double *__restrict__ out, const double *__restrict__ inp, uint64_t nel);
*/

template<typename O, typename I>
const char *tcPow(O *__restrict__ out, const I *__restrict__ inp, uint64_t nel) {
    /*std::transform(std::execution::par_unseq, inp, inp + nel, out, [](I a) {
        return std::sin(a);
    });*/

#pragma GCC ivdep
    for (uint64_t i = 0; i < nel; i++) {
        out[i] = std::pow(inp[i], 150);
    }
    return nullptr;
}

template const char *tcPow<double, double>(double *__restrict__ out, const double *__restrict__ inp, uint64_t nel);

/*
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
*/