#include <algorithm>
#include <execution>
#include <experimental/simd>
#include <iostream>
#include <memory>

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

  auto *inp1 = new uint8_t[size.nel()];
  auto *inp2 = new uint8_t[size.nel()];
  auto *out = new uint8_t[size.nel()];

  for (uint64_t i = 0; i < size.nel(); i++) {
    inp1[i] = (10 + i) % 256;
    inp2[i] = i % 256;
  }

  tcPlus<uint8_t, uint8_t, uint8_t>(out, inp1, inp2, size.nel(), 0, size, ArithMode::ewise);

  /*uint8_t simdSize = 16;

  auto in1It = SimdIterator<uint8_t>(inp1, simdSize, size.nel());
  // auto in2It = SimdIterator<uint8_t>(inp2, simdSize, size.nel());
  // auto in2It = ScalarSimdInpIter<uint8_t>(5, simdSize, size.nel(), 0);
  auto in2It = RwiseSimdIterator<uint8_t>(inp2, simdSize, size, 0);
  auto outIt = SimdIterator<uint8_t>(out, simdSize, size.nel());*/

  /*std::for_each(
      std::execution::par, in1It.begin(), in1It.end(),
      [&in1It, &in2It, &outIt](uint64_t a) {
        outIt[a] = *in1It[a] + *in2It[a];
      }
  );*/

  /*std::for_each(
      std::execution::par, in1It.countBegin(), in1It.countEnd(),
      [&in1It, &in2It, &outIt, simdSize](uint64_t a) {
        auto inp1 = *in1It[a];
        auto inp2 = *in2It[a];
        std::cout << "a: " << a << " ";
        for (uint8_t i = 0; i < simdSize; i++) {
          std::cout << "(" << (uint64_t)(uint8_t)inp1[i] << " "
                    << (uint64_t)(uint8_t)inp2[i] << ") ";
        }
        std::cout << std::endl;
        outIt[a] = *in1It[a] + *in2It[a];
      }
  );*/

  /*std::for_each(
      std::execution::par, in1It.countBegin(), in1It.countEnd(),
      [&in1It, &in2It, &outIt](uint64_t a) { outIt[a] = *in1It[a] + *in2It[a]; }
  );*/

  /*std::for_each(
      std::execution::par, in1It.countBegin(), in1It.countEnd(),
      [&in1It, &in2It, &outIt](uint64_t a) {
        std::cout << "a: " << a << ", thread: " << std::this_thread::get_id()
                  << std::endl;
        outIt[a] = *in1It[a] + *in2It[a];
      }
  );*/

  for (uint64_t i = 0; i < size.nel(); i++) {
    printf("%d %d %d\n", inp1[i], inp2[i], out[i]);
    if ((uint8_t)(inp1[i] + inp2[i]) != out[i]) {
      printf("Error at %lu; %u + %u != %u\n", i, inp1[i], inp2[i], out[i]);
      return 1;
    }
  }

  delete[] inp1;
  delete[] inp2;
  delete[] out;

  return 0;
}