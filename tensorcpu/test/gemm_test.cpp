#include <cblas.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>

#include "tensorcpu.hpp"

void mm_naive(
    float *__restrict__ out, const float *__restrict__ inp1,
    const float *__restrict__ inp2, Dim2 inp1S, Dim2 inp2S
) {
  for (uint32_t i = 0; i < inp1S.r; i++) {
    for (uint32_t j = 0; j < inp2S.c; j++) {
      float sum = 0;
#pragma GCC ivdep
      for (uint32_t k = 0; k < inp1S.c; k++) {
        sum += inp1[i * inp1S.c + k] * inp2[k * inp2S.c + j];
      }
      out[i * inp2S.c + j] = sum;
    }
  }
}

void mm_tiled(
    float *__restrict__ out, const float *__restrict__ inp1,
    const float *__restrict__ inp2, Dim2 inp1S, Dim2 inp2S
) {
  for (uint32_t i = 0; i < inp1S.r; i++) {
    for (uint32_t k = 0; k < inp1S.c; k++) {
      float a = inp1[i * inp1S.c + k];
#pragma GCC ivdep
      for (uint32_t j = 0; j < inp2S.c; j++) {
        out[i * inp2S.c + j] += a * inp2[k * inp2S.c + j];
      }
    }
  }
}

void mm_naive_loopReordered(
    float *__restrict__ out, const float *__restrict__ inp1,
    const float *__restrict__ inp2, Dim2 inp1S, Dim2 inp2S
) {
  for (uint32_t i = 0; i < inp1S.r; i++) {
    for (uint32_t k = 0; k < inp1S.c; k++) {
      float a = inp1[i * inp1S.c + k];
#pragma GCC ivdep
      for (uint32_t j = 0; j < inp2S.c; j++) {
        out[i * inp2S.c + j] += a * inp2[k * inp2S.c + j];
      }
    }
  }
}

void mm_openBlas(float *out, float *inp1, float *inp2, Dim2 inp1S, Dim2 inp2S) {
  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans, inp1S.r, inp2S.c, inp1S.c,
      1.0f, inp1, inp1S.c, inp2, inp2S.c, 0.0f, out, inp2S.c
  );
}

template <typename T> std::unique_ptr<T> allocate(uint64_t size) {
  return std::unique_ptr<T>(new T[size]);
}

template <typename T> void fill1(T *arr, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = i + 1;
    if (arr[i] == 0) {
      arr[i] = 1;
    }
  }
}

template <typename T> void fill1(T *arr, Dim2 size) { fill1(arr, size.nel()); }

namespace chrono = std::chrono;
using std::chrono::steady_clock;

int main() {
  uint32_t m = 512;
  uint32_t k = 256;
  uint32_t n = 512;

  Dim2 inp1S = {m, k};
  Dim2 inp2S = {k, n};
  auto out = allocate<float>(m * n);

  auto inp1 = allocate<float>(inp1S.nel());
  auto inp2 = allocate<float>(inp2S.nel());
  fill1(inp1.get(), inp1S);
  fill1(inp2.get(), inp2S);

  steady_clock::time_point begin = steady_clock::now();
  mm_openBlas(out.get(), inp1.get(), inp2.get(), inp1S, inp2S);
  steady_clock::time_point end = steady_clock::now();

  std::cout << "Time: "
            << chrono::duration_cast<chrono::microseconds>(end - begin).count()
            << "us" << std::endl;

  begin = steady_clock::now();
  mm_naive(out.get(), inp1.get(), inp2.get(), inp1S, inp2S);
  end = steady_clock::now();

  std::cout << "Time: "
            << chrono::duration_cast<chrono::microseconds>(end - begin).count()
            << "us" << std::endl;

  begin = steady_clock::now();
  mm_naive_loopReordered(out.get(), inp1.get(), inp2.get(), inp1S, inp2S);
  end = steady_clock::now();

  std::cout << "Time: "
            << chrono::duration_cast<chrono::microseconds>(end - begin).count()
            << "us" << std::endl;

  /*
  for (uint64_t i = 0; i < m * n; i++) {
    std::cout << "@" << i << " " << out.get()[i] << std::endl;
  }
   */

  return 0;
}
