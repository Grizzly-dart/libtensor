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
#include "typed_array.hpp"

namespace stdx = std::experimental;

template <typename T>
static void mmTile(
    T *out, const T *inp1, const T *inp2, Dim2 tileSize, uint16_t kTileSize,
    uint16_t origTileSize, bool first
) {
  constexpr const uint16_t laneSize = std::min(
      uint16_t(64 / sizeof(T)), uint16_t(stdx::native_simd<T>::size())
  );

  for (uint32_t m = 0; m < tileSize.r; m++) {
    for (uint32_t n = 0; n < tileSize.c; n += laneSize) {
      stdx::fixed_size_simd<T, laneSize> c(0);
      if (!first) {
        c.copy_from(out + (m * origTileSize) + n, stdx::vector_aligned);
      }
      for (uint32_t k = 0; k < kTileSize; k++) {
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
        for (uint32_t i = 0; i < rem; i++) {
          out[m * origTileSize + n + i] = static_cast<T>(c[i]);
        }
      }
    }
  }
}

template <typename T>
static void gemmRows(
    void *out, const void *inp1, const void *inp2, Dim2 size, uint32_t k,
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

template <typename O, typename I1, typename I2>
void mm_casted_slow(
    void *out, void *inp1, void *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize, uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  constexpr const uint16_t tileSize = 128; // TODO deduce tileSize
  uint16_t numThreads = std::thread::hardware_concurrency();
  uint32_t numBlocksPerMatrix = (size.r + tileSize - 1) / tileSize;
  uint32_t numBlocks = numBlocksPerMatrix * batchSize;
  if (numBlocks < numThreads) {
    numThreads = numBlocks;
  }
  uint32_t numBlocksPerThread = (numBlocks + numThreads - 1) / numThreads;

  DType outType = dtypes[outTID];
  DType inp1Type = dtypes[i1TID];
  DType inp2Type = dtypes[i2TID];
  const Caster<O>& oCaster = Caster<O>::lookup(outType);
  const Caster<I1>& i1Caster = Caster<I1>::lookup(inp1Type);
  const Caster<I2>& i2Caster = Caster<I2>::lookup(inp2Type);

  std::vector<std::future<void>> futures;
  for (uint16_t i = 0; i < numThreads; i++) {
    futures.push_back(std::async(
        std::launch::async,
        [i, out, inp1, inp2, size, k, tileSize, numBlocks, numBlocksPerThread,
         numBlocksPerMatrix, oCaster, i1Caster, i2Caster]() {
          auto a = std::unique_ptr<I1>(new I1[tileSize * tileSize]);
          auto b = std::unique_ptr<I2>(new I2[tileSize * tileSize]);
          auto c = std::unique_ptr<O>(new O[tileSize * tileSize]);

          for (uint32_t block = i * numBlocksPerThread;
               block <
               std::min(i * numBlocksPerThread + numBlocksPerThread, numBlocks);
               block++) {
            uint32_t batch = block / numBlocksPerMatrix;
            gemmRows(
                oCaster.indexer(out, batch * size.r * size.c),
                i1Caster.indexer(inp1, batch * size.r * k),
                i2Caster.indexer(inp2, batch * k * size.c), size, k, tileSize,
                block * tileSize, (block + 1) * tileSize, c.get(), a.get(),
                b.get()
            );
          }
        }
    ));
  }

  for (auto &f : futures) {
    f.wait();
  }
}

#define MM_CASTED_SLOW(O, I1, I2)                                              \
  template void mm_casted_slow<O, I1, I2>(                                     \
      void *out, void *inp1, void *inp2, Dim2 size, uint32_t k,    \
      uint32_t batchSize, uint8_t outTID, uint8_t i1TID, uint8_t i2TID         \
  );

MM_CASTED_SLOW(double, double, double)