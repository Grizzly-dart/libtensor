#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

template <typename T> auto testTcPlus(Dim3 size) {
  auto inp1 = std::unique_ptr<T>(new T[size.nel()]);
  auto inp2 = std::unique_ptr<T>(new T[size.nel()]);
  auto out = std::unique_ptr<T>(new T[size.nel()]);

  for (uint64_t i = 0; i < size.nel(); i++) {
    inp1.get()[i] = 10 + i;
    inp2.get()[i] = i;
  }

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  tcBinaryArith<T, T, BinaryOp::Plus>(
      out.get(), inp1.get(), inp2.get(), size.nel(), 0, Dim2{0, 0}
  );
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  for (uint64_t i = 0; i < size.nel(); i++) {
    // printf("%d %d %d\n", inp1.get()[i], inp2.get()[i], out.get()[i]);
    if ((T)(inp1.get()[i] + inp2.get()[i]) != out.get()[i]) {
      printf(
          "Error load %lu; %u + %u != %u\n", i, inp1.get()[i], inp2.get()[i],
          out.get()[i]
      );
      break;
    }
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
}

template <typename C, typename T> auto testTcPlusSlow(Dim3 size) {
  auto inp1 = std::unique_ptr<T>(new T[size.nel()]);
  auto inp2 = std::unique_ptr<T>(new T[size.nel()]);
  auto out = std::unique_ptr<T>(new T[size.nel()]);

  for (uint64_t i = 0; i < size.nel(); i++) {
    inp1.get()[i] = 10 + i;
    inp2.get()[i] = i;
  }

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  tcPlusSlow<C, C, BinaryOp::Plus>(
      (C*)out.get(), (C *)inp1.get(), (C *)inp2.get(), size.nel(), 0,
      Dim2{0, 0}, dtypeOf<T>().index, dtypeOf<T>().index, dtypeOf<T>().index
  );
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  for (uint64_t i = 0; i < size.nel(); i++) {
    // printf("%d %d %d\n", inp1.get()[i], inp2.get()[i], out.get()[i]);
    if ((T)(inp1.get()[i] + inp2.get()[i]) != out.get()[i]) {
      printf(
          "Error load %lu; %u + %u != %u\n", i, inp1.get()[i], inp2.get()[i],
          out.get()[i]
      );
      break;
    }
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
}

int main() {
  auto size = Dim3{10, 500, 500};

  std::chrono::microseconds dur(0);

  for(int i = 0; i < 5; i++) {
    dur = testTcPlusSlow<int64_t, uint8_t>(size);
    std::cout << "SlowDuration: " << dur.count() << "ns" << std::endl;

    dur = testTcPlusSlow<int16_t, uint8_t>(size);
    std::cout << "MediumDuration: " << dur.count() << "ns" << std::endl;

    dur = testTcPlus<uint8_t>(size);
    std::cout << "FastDuration: " << dur.count() << "ns" << std::endl;
  }

  fflush(stdout);

  return 0;
}
