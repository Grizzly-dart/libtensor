#include <algorithm>
#include <execution>
#include <experimental/simd>
#include <iostream>
#include <memory>

#include "tensorc.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

template <typename O, typename I1, typename I2>
void tcPlus(
    O *out, I1 *inp1, I2 *inp2, uint64_t nel, uint8_t flip, Dim3 size,
    ArithMode mode
) {
  size_t width = std::min(
      std::min(stdx::native_simd<O>::size(), stdx::native_simd<I1>::size()),
      stdx::native_simd<I2>::size()
  );
  printf("width: %zu\n", width);
  std::unique_ptr<SimdIter<I1>> in1It =
      std::make_unique<SimdIterator<I1>>(inp1, width, nel);
  std::unique_ptr<SimdIter<I2>> in2It;
  if (mode == ArithMode::ewise) {
    in2It = std::make_unique<SimdIterator<I2>>(inp2, width, nel);
  } else if (mode == ArithMode::rwise) {
    in2It = std::make_unique<RwiseSimdIterator<I2>>(inp2, width, size);
  } else if (mode == ArithMode::scalar) {
    in2It = std::make_unique<ScalarSimdInpIter<I2>>(
        ScalarSimdInpIter<I2>(5, width, nel)
    );
  } else {
    throw std::invalid_argument("Invalid mode");
  }
  auto outIt = SimdIterator<O>(out, width, nel);

  std::for_each(
      std::execution::par, (*in1It).countBegin(), (*in1It).countEnd(),
      [&in1It, &in2It, &outIt, flip](uint64_t a) {
        if (flip == 0) {
          outIt[a] = in1It->at(a) + in2It->at(a);
        } else {
          outIt[a] = in2It->at(a) - in1It->at(a);
        }
      }
  );
}

template <typename T> void testTcPlus(Dim3 size) {
  auto inp1 = std::unique_ptr<T>(new T[size.nel()]);
  auto inp2 = std::unique_ptr<T>(new T[size.nel()]);
  auto out = std::unique_ptr<T>(new T[size.nel()]);

  for (uint64_t i = 0; i < size.nel(); i++) {
    inp1.get()[i] = 10 + i;
    inp2.get()[i] = i;
  }

  tcPlus<T, T, T>(
      out.get(), inp1.get(), inp2.get(), size.nel(), 0, size, ArithMode::ewise
  );

  for (uint64_t i = 0; i < size.nel(); i++) {
    // printf("%d %d %d\n", inp1.get()[i], inp2.get()[i], out.get()[i]);
    if ((T)(inp1.get()[i] + inp2.get()[i]) != out.get()[i]) {
      printf(
          "Error at %lu; %u + %u != %u\n", i, inp1.get()[i], inp2.get()[i],
          out.get()[i]
      );
      return;
    }
  }
}

template <typename O, typename I1, typename I2>
void tcPlusGeneric(
    O *out, I1 *inp1, I2 *inp2, uint64_t nel, uint8_t flip, Dim3 size,
    ArithMode mode
) {
  size_t width = std::min(
      std::min(stdx::native_simd<O>::size(), stdx::native_simd<I1>::size()),
      stdx::native_simd<I2>::size()
  );
  printf("width: %zu\n", width);
  std::unique_ptr<SimdIter<I1>> in1It =
      std::make_unique<SimdIterator<I1>>(inp1, width, nel);
  std::unique_ptr<SimdIter<I2>> in2It;
  if (mode == ArithMode::ewise) {
    in2It = std::make_unique<SimdIterator<I2>>(inp2, width, nel);
  } else if (mode == ArithMode::rwise) {
    in2It = std::make_unique<RwiseSimdIterator<I2>>(inp2, width, size);
  } else if (mode == ArithMode::scalar) {
    in2It = std::make_unique<ScalarSimdInpIter<I2>>(
        ScalarSimdInpIter<I2>(5, width, nel)
    );
  } else {
    throw std::invalid_argument("Invalid mode");
  }
  auto outIt = SimdIterator<uint8_t>(out, width, nel);

  std::for_each(
      std::execution::par, (*in1It).countBegin(), (*in1It).countEnd(),
      [&in1It, &in2It, &outIt, flip](uint64_t a) {
        if (flip == 0) {
          outIt[a] = in1It->at(a) + in2It->at(a);
        } else {
          outIt[a] = in2It->at(a) - in1It->at(a);
        }
      }
  );
}

int main() {
  auto size = Dim3{10, 5, 3};

  testTcPlus<uint8_t>(size);

  return 0;
}