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

void mm_naive(
    float *__restrict__ out, const float *__restrict__ inp1,
    const float *__restrict__ inp2, Dim2 inp1S, Dim2 inp2S, uint32_t batchSize
) {
  for (uint32_t b = 0; b < batchSize; b++) {
    float *o = out + b * inp1S.r * inp2S.c;
    const float *i1 = inp1 + b * inp1S.r * inp1S.c;
    const float *i2 = inp2 + b * inp2S.r * inp2S.c;
    for (uint32_t m = 0; m < inp1S.r; m++) {
      for (uint32_t n = 0; n < inp2S.c; n++) {
        float sum = 0;
#pragma GCC ivdep
        for (uint32_t k = 0; k < inp1S.c; k++) {
          sum += i1[m * inp1S.c + k] * i2[k * inp2S.c + n];
        }
        o[m * inp2S.c + n] = sum;
      }
    }
  }
}

void mm_naive_loopReordered(
    float *__restrict__ out, const float *__restrict__ inp1,
    const float *__restrict__ inp2, Dim2 inp1S, Dim2 inp2S, uint32_t batchSize
) {
  for (uint32_t b = 0; b < batchSize; b++) {
    float *o = out + b * inp1S.r * inp2S.c;
    const float *i1 = inp1 + b * inp1S.r * inp1S.c;
    const float *i2 = inp2 + b * inp2S.r * inp2S.c;
    for (uint32_t m = 0; m < inp1S.r; m++) {
      for (uint32_t k = 0; k < inp1S.c; k++) {
        float a = i1[m * inp1S.c + k];
#pragma GCC ivdep
        for (uint32_t n = 0; n < inp2S.c; n++) {
          o[m * inp2S.c + n] += a * i2[k * inp2S.c + n];
        }
      }
    }
  }
}

void mm_multithreaded_colwise_nobatch(
    float *out, const float *inp1, const float *inp2, Dim2 inp1S, Dim2 inp2S,
    uint32_t batchSize, uint16_t tileSize
) {
  if (tileSize == 0) {
    tileSize = 64 / sizeof(float);
  }
  uint16_t numThreads = std::thread::hardware_concurrency();
  const uint32_t i1NTilesR = (inp1S.r + tileSize - 1) / tileSize;
  const uint32_t i1NTilesC = (inp1S.c + tileSize - 1) / tileSize;
  const uint32_t i1NTiles = i1NTilesR * i1NTilesC;
  const uint32_t i2NTilesC = (inp2S.c + tileSize - 1) / tileSize;
  uint16_t tilesPerThread;
  if (i1NTiles < numThreads) {
    tilesPerThread = 1;
    numThreads = i1NTiles;
  } else {
    tilesPerThread = (i1NTiles + numThreads - 1) / numThreads;
  }

  std::vector<std::future<void>> futures;
  for (uint16_t i = 0; i < numThreads; i++) {
    futures.push_back(std::async(
        std::launch::async,
        [i, tilesPerThread, &out, &inp1, &inp2, inp1S, inp2S, i1NTilesR,
         i2NTilesC, tileSize]() {
          for (uint16_t j = 0; j < tilesPerThread; j++) {
            uint16_t tile = i * tilesPerThread + j;
            uint32_t aTileR = tile % i1NTilesR;
            uint32_t aTileC = tile / i1NTilesR;

            uint16_t kMax =
                std::min(uint16_t(inp1S.c - aTileC * tileSize), tileSize);

            for (uint32_t m = aTileR * tileSize;
                 m < std::min((aTileR * tileSize) + tileSize, inp1S.r); m++) {
              for (uint32_t i2Tc = 0; i2Tc < i2NTilesC; i2Tc++) {
                for (uint32_t k = 0; k < kMax; k++) {
                  uint32_t curK = aTileC * tileSize + k;
                  float i1 = inp1[m * inp1S.c + curK];
#pragma GCC ivdep
                  for (uint32_t n = i2Tc * tileSize;
                       n < std::min((i2Tc * tileSize) + tileSize, inp2S.c);
                       n++) {
                    // out[m * inp2S.c + n] += i1 * inp2[curK * inp2S.c + n];
                    float v = i1 * inp2[curK * inp2S.c + n];
                    float *ptr = &out[m * inp2S.c + n];
                    atomicAdd(ptr, v);
                  }
                }
              }
            }
          }
        }
    ));
  }

  for (auto &f : futures) {
    f.get();
  }
}

void mm_openBlas(
    float *out, const float *inp1, const float *inp2, Dim2 inp1S, Dim2 inp2S,
    uint32_t batchSize
) {
  for (uint32_t b = 0; b < batchSize; b++) {
    float *o = out + b * inp1S.r * inp2S.c;
    const float *i1 = inp1 + b * inp1S.r * inp1S.c;
    const float *i2 = inp2 + b * inp2S.r * inp2S.c;
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, inp1S.r, inp2S.c, inp1S.c,
        1.0f, i1, inp1S.c, i2, inp2S.c, 0.0f, o, inp2S.c
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
        if (diff > 1e-3) {
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
  uint32_t b = 20;
  uint32_t m = 256;
  uint32_t k = 256;
  uint32_t n = 256;

  Dim2 inp1S = {m, k};
  Dim2 inp2S = {k, n};
  auto out = allocate<float>(b * m * n);

  auto inp1 = allocate<float>(b * inp1S.nel());
  auto inp2 = allocate<float>(b * inp2S.nel());
  fill1(inp1.get(), inp1S);
  fill1(inp2.get(), inp2S);

  for (int j = 0; j < 10; j++) {
    steady_clock::time_point begin = steady_clock::now();
    mm_openBlas(out.get(), inp1.get(), inp2.get(), inp1S, inp2S, b);
    steady_clock::time_point end = steady_clock::now();
    std::cout
        << "Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;

    auto out1 = allocate<float>(b * m * n);
    /*
    begin = steady_clock::now();
    mm_naive(out1.get(), inp1.get(), inp2.get(), inp1S, inp2S, b);
    end = steady_clock::now();
    std::cout
        << "Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check<float>(out.get(), out1.get(), {m, n});

    zero(out1.get(), m * n);
    begin = steady_clock::now();
    mm_naive_loopReordered(out1.get(), inp1.get(), inp2.get(), inp1S, inp2S, b);
    end = steady_clock::now();
    std::cout
        << "Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check<float>(out.get(), out1.get(), {m, n});
     */

    zero(out1.get(), m * n);
    begin = steady_clock::now();
    mm_multithreaded_colwise(
        out1.get(), inp1.get(), inp2.get(), inp1S, inp2S, b, 0
    );
    end = steady_clock::now();
    std::cout
        << "Time: "
        << chrono::duration_cast<chrono::microseconds>(end - begin).count()
        << "us" << std::endl;
    check<float>(out.get(), out1.get(), {b, m, n});

    /*
    for (uint64_t i = 0; i < m * n; i++) {
      std::cout << "@" << i << " " << out.get()[i] << std::endl;
    }
     */

    std::cout << "====================" << std::endl;
  }

  return 0;
}
