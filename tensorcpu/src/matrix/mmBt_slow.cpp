#include <algorithm>
#include <cmath>
#include <execution>
#include <experimental/simd>
#include <future>
#include <iostream>
#include <stdfloat>
#include <thread>
#include <typeinfo>

#include "matrix.hpp"
#include "tensorcpu.hpp"

namespace stdx = std::experimental;

#if 0
template <typename T>
void mmBtTile(
    T *out, const T *inp1, const T *inp2, Dim2 tileSize, uint16_t kTileSize,
    uint16_t origTileSize, bool first
) {
  constexpr const uint16_t laneSize = std::min(
      uint16_t(64 / sizeof(T)), uint16_t(stdx::native_simd<T>::size())
  );

  for (uint32_t m = 0; m < tileSize.r; m++) {
    const T* i2 = inp2;
    for (uint32_t n = 0; n < tileSize.c; n++) {
      stdx::fixed_size_simd<T, laneSize> c(0);
      for (uint32_t k = 0; k < kTileSize; k += laneSize) {
        stdx::fixed_size_simd<T, laneSize> a(inp1 + k, stdx::vector_aligned);
        stdx::fixed_size_simd<T, laneSize> b(i2 + k, stdx::vector_aligned);
        c += a * b;
      }
      T sum = 0;
      // TODO vectorize horizontal sum
      for (uint16_t i = 0; i < laneSize; i++) {
        T v = static_cast<T>(c[i]);
        sum += v;
      }
      if (!first) {
        out[m * origTileSize + n] += sum;
      } else {
        out[m * origTileSize + n] = sum;
      }
      i2 += origTileSize;
    }
    inp1 += origTileSize;
  }
}

#define MMBTTILE(T)                                                           \
  template void mmBtTile<T>(                                                       \
      T * out, const T *inp1, const T *inp2, Dim2 tileSize, uint16_t kTileSize, \
      uint16_t origTileSize, bool first                                        \
  );

UNWIND1_ALL_TYPES(MMBTTILE)

template <typename T>
static void gemmRows(
    T *out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,
    uint16_t tileSize, uint32_t start, uint32_t end, T *c, T *a, T *b
) {

  for (uint32_t mOffset = start; mOffset < end; mOffset += tileSize) {
    uint32_t mTileSize = std::min(uint32_t(tileSize), end - mOffset);
    for (uint32_t nOffset = 0; nOffset < size.c; nOffset += tileSize) {
      uint32_t nTileSize = std::min(uint32_t(tileSize), size.c - nOffset);
      for (uint32_t kOffset = 0; kOffset < k; kOffset += tileSize) {
        uint16_t kTileSize = std::min(uint32_t(tileSize), k - kOffset);
        loadTile(
            a, inp1, {.r = size.r, .c = k}, {.r = mOffset, .c = kOffset},
            {mTileSize, kTileSize}, tileSize
        );
        loadTile(
            b, inp2, {.r = size.c, .c = k}, {.r = nOffset, .c = kOffset},
            {nTileSize, kTileSize}, tileSize
        );
        mmBtTile(
            c, a, b, {mTileSize, nTileSize}, kTileSize, tileSize, kOffset == 0
        );
      }
      storeTile(
          out, c, size, {.r = mOffset, .c = nOffset}, {mTileSize, nTileSize},
          tileSize
      );
    }
  }
}

template <typename T>
void mmBt_same_slow(
    T *out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize
) {
  constexpr const uint16_t tileSize = 128; // TODO deduce tileSize
  uint16_t numThreads = std::thread::hardware_concurrency();
  uint32_t numBlocksPerMatrix = (size.r + tileSize - 1) / tileSize;
  uint32_t numBlocks = numBlocksPerMatrix * batchSize;
  if (numBlocks < numThreads) {
    numThreads = numBlocks;
  }
  uint32_t numBlocksPerThread = (numBlocks + numThreads - 1) / numThreads;

  std::vector<std::future<void>> futures;
  for (uint16_t i = 0; i < numThreads; i++) {
    futures.push_back(std::async(
        std::launch::async,
        [i, out, inp1, inp2, size, k, tileSize, numBlocks, numBlocksPerThread,
         numBlocksPerMatrix]() {
          auto a = std::unique_ptr<T>(new T[tileSize * tileSize]);
          auto b = std::unique_ptr<T>(new T[tileSize * tileSize]);
          auto c = std::unique_ptr<T>(new T[tileSize * tileSize]);

          for (uint32_t block = i * numBlocksPerThread;
               block <
               std::min(i * numBlocksPerThread + numBlocksPerThread, numBlocks);
               block++) {
            uint32_t batch = block / numBlocksPerMatrix;
            gemmRows(
                out + batch * size.r * size.c, inp1 + batch * size.r * k,
                inp2 + batch * k * size.c, size, k, tileSize, block * tileSize,
                (block + 1) * tileSize, c.get(), a.get(), b.get()
            );
          }
        }
    ));
  }

  for (auto &f : futures) {
    f.wait();
  }
}

#define MMBT_SLOW(T)                                                           \
  template void mmBt_same_slow<T>(                                                       \
      T * out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,            \
      uint32_t batchSize                                                       \
  );

UNWIND1_ALL_TYPES(MMBT_SLOW)
#endif