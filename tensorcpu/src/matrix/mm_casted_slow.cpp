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

#if 0
template <typename T>
static void gemmRows(
    void *out, void *inp1, void *inp2, Dim2 size, uint32_t k, uint16_t tileSize,
    uint32_t start, uint32_t end, T *c, T *a, T *b, const Caster<T> &oCaster,
    const Caster<T> &i1Caster, const Caster<T> &i2Caster
) {

  for (uint32_t mOffset = start; mOffset < end; mOffset += tileSize) {
    uint32_t mTileSize = std::min(uint32_t(tileSize), end - mOffset);
    for (uint32_t nOffset = 0; nOffset < size.c; nOffset += tileSize) {
      uint32_t nTileSize = std::min(uint32_t(tileSize), size.c - nOffset);
      for (uint32_t kOffset = 0; kOffset < k; kOffset += tileSize) {
        uint16_t kTileSize = std::min(uint32_t(tileSize), k - kOffset);
        loadTileCasted(
            a, inp1, {.r = size.r, .c = k}, {.r = mOffset, .c = kOffset},
            {mTileSize, kTileSize}, tileSize, i1Caster
        );
        loadTileCasted(
            b, inp2, {.r = k, .c = size.c}, {.r = kOffset, .c = nOffset},
            {kTileSize, nTileSize}, tileSize, i2Caster
        );
        mmTile(
            c, a, b, {mTileSize, nTileSize}, kTileSize, tileSize, kOffset == 0
        );
      }
      storeTileCasted(
          out, c, size, {.r = mOffset, .c = nOffset}, {mTileSize, nTileSize},
          tileSize, oCaster
      );
    }
  }
}

template <typename T>
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
  const Caster<T> &oCaster = Caster<T>::lookup(outType);
  const Caster<T> &i1Caster = Caster<T>::lookup(inp1Type);
  const Caster<T> &i2Caster = Caster<T>::lookup(inp2Type);

  std::vector<std::future<void>> futures;
  for (uint16_t i = 0; i < numThreads; i++) {
    futures.push_back(std::async(
        std::launch::async,
        [i, out, inp1, inp2, size, k, tileSize, numBlocks, numBlocksPerThread,
         numBlocksPerMatrix, &oCaster, &i1Caster, &i2Caster]() {
          auto a = std::unique_ptr<T>(new T[tileSize * tileSize]);
          auto b = std::unique_ptr<T>(new T[tileSize * tileSize]);
          auto c = std::unique_ptr<T>(new T[tileSize * tileSize]);

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
                b.get(), oCaster, i1Caster, i2Caster
            );
          }
        }
    ));
  }

  for (auto &f : futures) {
    f.wait();
  }
}

#define MM_CASTED_SLOW(O)                                                      \
  template void mm_casted_slow<O>(                                             \
      void *out, void *inp1, void *inp2, Dim2 size, uint32_t k,                \
      uint32_t batchSize, uint8_t outTID, uint8_t i1TID, uint8_t i2TID         \
  );

UNWIND1_ALL_TYPES(MM_CASTED_SLOW)
#endif