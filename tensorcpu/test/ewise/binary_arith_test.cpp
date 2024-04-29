#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdfloat>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

namespace stdx = std::experimental;

template <typename C, typename O, typename I1, typename I2>
void checkBinaryArith(
    O *out, I1 *inp1, I2 *inp2, BinaryOp op, bool flip, uint64_t nel
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
        C tmp = std::pow(inp1[i], inp2[i]);
        v = static_cast<O>(tmp);
      } else {
        C tmp = std::pow(inp2[i], inp1[i]);
        v = static_cast<O>(tmp);
      }
    }
    if (out[i] != v) {
      std::cout << "CheckFail @" << i << "; "
                << "a: " << +inp1[i] << " b: " << +inp2[i] << "; " << +out[i]
                << " != " << +v << std::endl;
      break;
    }
  }
}

template <typename T> std::unique_ptr<T> allocate(uint64_t size) {
  return std::unique_ptr<T>(new T[size]);
}

template <typename T> void fill1(T *arr, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = i;
    if (arr[i] == 0) {
      arr[i] = 1;
    }
  }
}

template <typename T> void fill2(T *arr, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = i + 11;
    if (arr[i] == 0) {
      arr[i] = 1;
    }
  }
}

template <typename T>
auto testBinaryArith(
    T *out, T *inp1, T *inp2, Dim3 size, BinaryOp op, bool flip
) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  tcBinaryArith<T, T>(out, inp1, inp2, op, size.nel(), flip, Dim2{0, 0});
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  checkBinaryArith<T, T, T, T>(out, inp1, inp2, op, flip, size.nel());
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
}

template <typename CO, typename CI, typename O, typename I>
auto testBinaryArithSlow(
    void *out, void *inp1, void *inp2, Dim3 size, BinaryOp op, bool flip
) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  tcBinaryArithCasted<CO, CI>(
      out, inp1, inp2, op, size.nel(), flip, Dim2{0, 0}, dtypeOf<O>().index,
      dtypeOf<I>().index, dtypeOf<I>().index
  );
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  checkBinaryArith<CO, O, I, I>(
      (O *)out, (I *)inp1, (I *)inp2, op, flip, size.nel()
  );
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
}

template <typename C, typename T>
auto testBinaryArithCastedPlain(
    void *out, void *inp1, void *inp2, Dim3 size, BinaryOp op, bool flip
) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  tcBinaryArithCastedPlain<C, C, C>(
      out, inp1, inp2, op, size.nel(), flip, Dim2{0, 0}, dtypeOf<T>().index,
      dtypeOf<T>().index, dtypeOf<T>().index
  );
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  checkBinaryArith<C, T, T, T>(
      (T *)out, (T *)inp1, (T *)inp2, op, flip, size.nel()
  );
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
}

int main() {
  auto size = Dim3{10, 500, 500};

  std::chrono::microseconds dur(0);

  const BinaryOp op = BinaryOp::Div;
  const bool flip = false;

  auto inp1 = allocate<uint8_t>(size.nel());
  auto inp2 = allocate<uint8_t>(size.nel());
  auto out = allocate<uint8_t>(size.nel());
  fill1(inp1.get(), size.nel());
  fill2(inp2.get(), size.nel());

  for (int i = 0; i < 1; i++) {
    dur = testBinaryArithSlow<int64_t, int64_t, uint8_t, uint8_t>(
        out.get(), inp1.get(), inp2.get(), size, op, flip
    );
    std::cout << "SlowDuration: " << dur.count() << "ns" << std::endl;

    dur = testBinaryArithSlow<int16_t, int16_t, uint8_t, uint8_t>(
        out.get(), inp1.get(), inp2.get(), size, op, flip
    );
    std::cout << "MediumDuration: " << dur.count() << "ns" << std::endl;

    dur = testBinaryArith<uint8_t>(
        out.get(), inp1.get(), inp2.get(), size, op, flip
    );
    std::cout << "FastDuration: " << dur.count() << "ns" << std::endl;

    /*
    dur = testBinaryArithCastedPlain<int16_t, uint8_t>(
        out.get(), inp1.get(), inp2.get(), size, op, flip
    );
    std::cout << "NovecDurationFast: " << dur.count() << "ns" << std::endl;

    dur = testBinaryArithCastedPlain<int32_t, uint8_t>(
        out.get(), inp1.get(), inp2.get(), size, op, flip
    );
    std::cout << "NovecDurationFast: " << dur.count() << "ns" << std::endl;

    dur = testBinaryArithCastedPlain<int64_t, uint8_t>(
        out.get(), inp1.get(), inp2.get(), size, op, flip
    );
    std::cout << "NovecDurationSlow: " << dur.count() << "ns" << std::endl;
     */
  }

  fflush(stdout);

  return 0;
}
