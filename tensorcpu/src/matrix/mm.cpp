#include <algorithm>
#include <cassert>
#include <cmath>
#include <execution>
#include <experimental/simd>
#include <future>
#include <iostream>
#include <stdfloat>
#include <thread>
#include <typeinfo>

#include "tensorcpu.hpp"

namespace stdx = std::experimental;

template <typename T>
static void loadTile(
    T *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,
    uint16_t origTileSize
) {
  for (uint32_t i = 0; i < tileSize.r; i++) {
    uint32_t row = tileOffset.r + i;
    if (row >= size.r) {
#pragma GCC ivdep
      for (uint32_t j = 0; j < tileSize.c; j++) {
        out[i * origTileSize + j] = 0;
      }
      continue;
    }
#pragma GCC ivdep
    for (uint32_t j = 0; j < tileSize.c; j++) {
      uint32_t col = tileOffset.c + j;
      if (col < size.c) {
        uint32_t idx = row * size.c + col;
        out[i * origTileSize + j] = inp[idx];
      } else {
        out[i * origTileSize + j] = 0;
      }
    }
  }
}

template <typename T>
static void storeTile(
    T *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,
    uint16_t origTileSize
) {
  for (uint32_t i = 0; i < tileSize.r; i++) {
#pragma GCC ivdep
    for (uint32_t j = 0; j < tileSize.c; j++) {
      uint32_t idx = (tileOffset.r + i) * size.c + tileOffset.c + j;
      if (idx < size.r * size.c) {
        out[idx] = inp[i * origTileSize + j];
      }
    }
  }
}

template <typename T>
static void mmTile(
    T *out, const T *inp1, const T *inp2, Dim2 tileSize, uint16_t kTileSize,
    uint16_t origTileSize, bool first
) {
  constexpr const uint16_t laneSize = 64 / sizeof(T);

  for (uint16_t m = 0; m < tileSize.r; m++) {
    for (uint16_t n = 0; n < tileSize.c; n += laneSize) {
      stdx::fixed_size_simd<T, laneSize> c = 0;
      if (!first) {
        c.copy_from(out + (m * origTileSize) + n, stdx::vector_aligned);
      }
      for (uint16_t k = 0; k < kTileSize; k++) {
        stdx::fixed_size_simd<T, laneSize> b(
            inp2 + (k * origTileSize) + n, stdx::vector_aligned
        );
        stdx::fixed_size_simd<T, laneSize> a(*(inp1 + (m * origTileSize) + k));
        c += a * b;
      }
      auto rem = tileSize.c - n;
      if (rem >= laneSize) {
        c.copy_to(out + m * origTileSize + n, stdx::vector_aligned);
      } else {
#pragma GCC ivdep
        for (uint16_t i = 0; i < rem; i++) {
          out[m * origTileSize + n + i] = static_cast<T>(c[i]);
        }
      }
    }
  }
}

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
            b, inp2, {.r = k, .c = size.c}, {.r = kOffset, .c = nOffset},
            {kTileSize, nTileSize}, tileSize
        );
        mmTile(
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
void mm(
    T *out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize, uint16_t tileSize
) {
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
          // TODO adaptive tile size for last
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

template void mm<float>(
    float *out, const float *inp1, const float *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize, uint16_t tileSize
);