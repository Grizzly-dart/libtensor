#include <algorithm>
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
static void loadATile(
    T *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize
) {
  // TODO vectorize loads
  for (uint32_t i = 0; i < tileSize.r; i++) {
    for (uint32_t j = 0; j < tileSize.c; j++) {
      uint32_t idx = (tileOffset.r + i) * size.c + tileOffset.c + j;
      if (idx < size.r * size.c) {
        out[i * tileSize.c + j] = inp[idx];
      } else {
        out[i * tileSize.c + j] = 0;
      }
    }
  }
}

template <typename T>
static void loadBTileT(
    T *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize
) {
  // TODO vectorize loads
  for (uint32_t i = 0; i < tileSize.r; i++) {
    for (uint32_t j = 0; j < tileSize.c; j++) {
      uint32_t idx = (tileOffset.r + i) * size.c + tileOffset.c + j;
      if (idx < size.r * size.c) {
        out[j * tileSize.r + i] = inp[idx];
      } else {
        out[j * tileSize.r + i] = 0;
      }
    }
  }
}

template <typename T>
static void mmKernel(T *out, const T *inp1, const T *inp2, uint16_t tileSize) {
  constexpr const uint16_t laneSize = 64 / sizeof(T);

  for(uint16_t m = 0; m < tileSize; m++) {
    for (uint16_t k = 0; k < tileSize; k += laneSize) {
      stdx::fixed_size_simd<T, laneSize> c;
      c.copy_from(out + (m * tileSize) + k);
      for (uint16_t n = 0; n < tileSize; n++) {
        stdx::fixed_size_simd<T, laneSize> b;
        b.copy_from(inp2 + (n * tileSize) + k);
        stdx::fixed_size_simd<T, laneSize> a = *(inp1 + (m * tileSize) + n);
        c += a * b;
      }
      c.copy_to(out + (m * tileSize) + k);
    }
  }
}

template <typename T>
static void mmTile(
    T *out, const T *inp1, const T *inp2, Dim2 size, Dim2 tileOffset,
    Dim2 tileSize
) {
  loadATile(out, inp1, size, tileOffset, tileSize);
  loadBTileT(out, inp2, size, tileOffset, tileSize);

  // TODO
}

void mm(
    float *out, const float *inp1, const float *inp2, Dim2 inp1S, Dim2 inp2S,
    uint32_t batchSize
) {
  constexpr const uint16_t tileSize = 64 / sizeof(float);
  constexpr uint8_t simdSize = stdx::native_simd<float>::size();
  constexpr uint16_t numSimdLoops = tileSize / simdSize;
  constexpr uint16_t lastSimdSize = tileSize % simdSize;
  uint16_t numThreads = std::thread::hardware_concurrency();
  const uint32_t i1NTilesR = (inp1S.r + tileSize - 1) / tileSize;
  const uint32_t i1NTilesC = (inp1S.c + tileSize - 1) / tileSize;
  const uint32_t i1NTiles = i1NTilesR * i1NTilesC;
  const uint32_t i2NTilesC = (inp2S.c + tileSize - 1) / tileSize;
  uint32_t batchesPerThread;
  if (batchSize < numThreads) {
    batchesPerThread = 1;
    numThreads = batchSize;
  } else {
    batchesPerThread = (batchSize + numThreads - 1) / numThreads;
  }

  std::vector<std::future<void>> futures;
  for (uint16_t i = 0; i < numThreads; i++) {
    futures.push_back(std::async(
        std::launch::async,
        [i, batchesPerThread, &out, &inp1, &inp2, inp1S, inp2S, i1NTilesR,
         i2NTilesC, tileSize, batchSize, i1NTiles]() {
          for (uint32_t b = i * batchesPerThread;
               b < std::min(i * batchesPerThread + batchesPerThread, batchSize);
               b++) {
            float *o = out + b * inp1S.r * inp2S.c;
            const float *i1 = inp1 + b * inp1S.r * inp1S.c;
            const float *i2 = inp2 + b * inp2S.r * inp2S.c;
            for (uint16_t tile = 0; tile < i1NTiles; tile++) {
              uint32_t i1TileR = tile % i1NTilesR;
              uint32_t i1TileC = tile / i1NTilesR;

              uint16_t kMax =
                  std::min(uint16_t(inp1S.c - i1TileC * tileSize), tileSize);

              for (uint32_t m = i1TileR * tileSize;
                   m < std::min((i1TileR * tileSize) + tileSize, inp1S.r);
                   m++) {
                for (uint32_t i2Tc = 0; i2Tc < i2NTilesC; i2Tc++) {
                  for (uint32_t k = 0; k < kMax; k++) {
                    uint32_t curK = i1TileC * tileSize + k;
                    float aSc = i1[m * inp1S.c + curK];
                    stdx::native_simd<float> a(aSc);
                    for (uint32_t n = i2Tc * tileSize;
                         n < std::min((i2Tc * tileSize) + tileSize, inp2S.c);
                         n++) {

                      o[m * inp2S.c + n] += a * i2[curK * inp2S.c + n];
                    }
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