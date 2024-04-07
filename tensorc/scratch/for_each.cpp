#include <algorithm>
#include <execution>
#include <experimental/simd>
#include <iostream>

#include "typed_array.hpp"

namespace stdx = std::experimental;

template <typename T> void tcPlus(T *out, const T *inp1, const T *inp2) {

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

  uint8_t simdSize = 16;

  auto in1It = SimdInpIterator<uint8_t>(inp1, simdSize, size.nel());
  // auto in2It = SimdInpIterator<uint8_t>(inp2, simdSize, size.nel());
  // auto in2It = ScalarSimdInpIter<uint8_t>(5, simdSize, size.nel(), 0);
  auto in2It = RwiseSimdIterator<uint8_t>(inp2, simdSize, size, 0);
  auto outIt = SimdInpIterator<uint8_t>(out, simdSize, size.nel());

  /*std::for_each(
      std::execution::par, in1It.begin(), in1It.end(),
      [&in1It, &in2It, &outIt](uint64_t a) {
        outIt[a] = *in1It[a] + *in2It[a];
      }
  );*/

  std::for_each(
      std::execution::par, in1It.countBegin(), in1It.countEnd(),
      [&in1It, &in2It, &outIt, simdSize](uint64_t a) {
        auto inp1 = *in1It[a];
        auto inp2 = *in2It[a];
        std::cout << "a: " << a << " ";
        for (uint8_t i = 0; i < simdSize; i++) {
          std::cout << "(" << (uint64_t)(uint8_t)inp1[i] << " " << (uint64_t)(uint8_t)inp2[i] << ") ";
        }
        std::cout << std::endl;
        outIt[a] = *in1It[a] + *in2It[a];
      }
  );

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
  }

  delete[] inp1;
  delete[] inp2;

  return 0;
}