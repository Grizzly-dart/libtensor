#include <algorithm>
#include <cblas.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <experimental/simd>
#include <future>
#include <iostream>
#include <memory>
#include <thread>

#include "tensorcpu.hpp"
#include "typed_array.hpp"

#include "native.hpp"

namespace stdx = std::experimental;

void mm_openBlas(
    float *out, const float *inp1, const float *inp2, Dim2 inp1S, Dim2 inp2S,
    uint32_t batchSize
) {
  for (uint32_t b = 0; b < batchSize; b++) {
    float *o = out + b * inp1S.r * inp2S.c;
    const float *i1 = inp1 + b * inp1S.r * inp1S.c;
    const float *i2 = inp2 + b * inp2S.r * inp2S.c;
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, inp1S.r, inp2S.c, inp1S.c,
        1.0f, i1, inp1S.c, i2, inp2S.r, 0.0f, o, inp2S.c
    );
  }
}

template <typename T> std::unique_ptr<T> allocate(uint64_t size) {
  return std::unique_ptr<T>(new T[size]);
}

template <typename T> void zero(T *arr, uint64_t size) {
  std::fill(arr, arr + size, 0);
}

template <typename T> void fill1(T *arr, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = i + 1;
    if (isRealNum<T>()) {
      arr[i] = arr[i] / 100000;
    }
  }
}

template <typename T> void fill1(T *arr, Dim2 size) { fill1(arr, size.nel()); }

template <typename T> void check(T *expected, T *produced, Dim3 size) {
  for (uint32_t bat = 0; bat < size.ch; bat++) {
    T *e = expected + bat * size.r * size.c;
    T *p = produced + bat * size.r * size.c;
    for (uint32_t m = 0; m < size.r; m++) {
      for (uint32_t n = 0; n < size.c; n++) {
        T a = e[m * size.c + n];
        T b = p[m * size.c + n];
        T diff = std::abs(a - b);
        if (diff > a * 1e-3) {
          std::cout << "Mismatch at " << bat << ":" << m << ":" << n << " => "
                    << a << " != " << b << "; " << diff << std::endl;
          return;
        }
      }
    }
  }
}

namespace chrono = std::chrono;
using std::chrono::steady_clock;

int main() {
  uint32_t b = 1;
  uint32_t m = 2048;
  uint32_t n = 2048;
  uint32_t k = 2048;

  Dim2 inp1S = {m, k};
  Dim2 inp2S = {k, n};
  Dim2 outS = {m, n};
  auto out = allocate<float>(b * m * n);

  auto inp1 = allocate<float>(b * inp1S.nel());
  auto inp2 = allocate<float>(b * inp2S.nel());
  fill1(inp1.get(), inp1S);
  fill1(inp2.get(), inp2S);

  std::cout << "Concurrent: " << std::thread::hardware_concurrency()
            << std::endl;

  for (int j = 0; j < 10; j++) {
    zero(out.get(), m * n);
    steady_clock::time_point begin = steady_clock::now();
    mm_openBlas(out.get(), inp1.get(), inp2.get(), inp1S, inp2S, b);
    steady_clock::time_point end = steady_clock::now();
    std::cout
        << "OpenBlas Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;

    auto out1 = allocate<float>(b * m * n);

    zero(out1.get(), m * n);
    begin = steady_clock::now();
    mmBt(out1.get(), inp1.get(), inp2.get(), outS, k, b, 128);
    end = steady_clock::now();
    std::cout
        << "Tiled128 Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check<float>(out.get(), out1.get(), {b, m, n});

    /*
    zero(out1.get(), m * n);
    begin = steady_clock::now();
    mm_naive(out1.get(), inp1.get(), inp2.get(), inp1S, inp2S, b);
    end = steady_clock::now();
    std::cout
        << "Naive Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check<float>(out.get(), out1.get(), {m, n});

    zero(out1.get(), m * n);
    begin = steady_clock::now();
    mm_naive_loopReordered(out1.get(), inp1.get(), inp2.get(), inp1S, inp2S, b);
    end = steady_clock::now();
    std::cout
        << "LoopOrdered Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check<float>(out.get(), out1.get(), {m, n});
  */

    /*
    for (uint64_t i = 0; i < m * n; i++) {
      std::cout << "@" << i << " " << out.get()[i] << std::endl;
    }
     */

    std::cout << "====================" << std::endl;
  }

  return 0;
}
