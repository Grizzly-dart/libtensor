//
// Created by tejag on 2024-04-27.
//

#include <cstdint>

#include "matrix.hpp"
#include "tensorcpu.hpp"
#include "typed_array.hpp"

template <typename T>
void loadTile(
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

#define LOADTILE(T)                                                            \
  template void loadTile<T>(                                                   \
      T * out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,        \
      uint16_t origTileSize                                                    \
  );

UNWIND1_ALL_TYPES(LOADTILE)

template <typename T>
void storeTile(
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

#define STORETILE(T)                                                           \
  template void storeTile<T>(                                                  \
      T * out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,        \
      uint16_t origTileSize                                                    \
  );

UNWIND1_ALL_TYPES(STORETILE)

template <typename T>
void loadTileCasted(
    T *out, void *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,
    uint16_t origTileSize, const Caster<T> &caster
) {
  stdx::native_simd<T> tmp;
  constexpr uint16_t laneSize = stdx::native_simd<T>::size();
  uint32_t rowStart = tileOffset.r;
  for (uint32_t i = 0; i < tileSize.r; i++) {
    if (tileOffset.r + i >= size.r) {
#pragma GCC ivdep
      for (uint32_t j = 0; j < tileSize.c; j++)
        out[i * origTileSize + j] = 0;
      continue;
    }
    for (uint32_t j = 0; j < tileSize.c; j += laneSize) {
      if (j + laneSize < tileSize.c && tileOffset.c + j + laneSize < size.c) {
        caster.simdLoader((void *)inp, rowStart + tileOffset.c + j, tmp);
        tmp.copy_to(&out[i * origTileSize + j], stdx::vector_aligned);
      } else {
        for (uint32_t e = 0; e < laneSize; e++) {
          uint32_t col = tileOffset.c + j + e;
          if (col < size.c) {
            out[i * origTileSize + j + e] = caster.loader(inp, rowStart + col);
          } else {
            out[i * origTileSize + j + e] = 0;
          }
        }
      }
    }
    rowStart += size.c;
  }
}

#define LOADTILECASTED(T)                                                      \
  template void loadTileCasted<T>(                                             \
      T * out, void *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,           \
      uint16_t origTileSize, const Caster<T> &caster                                 \
  );

UNWIND1_ALL_TYPES(LOADTILECASTED)

template <typename T>
void storeTileCasted(
    void *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,
    uint16_t origTileSize, const Caster<T> &caster
) {
  constexpr uint16_t laneSize = stdx::native_simd<T>::size();
  stdx::native_simd<T> tmp;
  const T *i1 = inp;
  uint64_t oOffset = tileOffset.r * size.c + tileOffset.c;
  uint32_t maxR = std::min(tileSize.r, size.r - tileOffset.r);
  uint32_t maxC = std::min(tileSize.c, size.c - tileOffset.c);
  for (uint32_t i = 0; i < maxR; i++) {
    for (uint32_t j = 0; j < maxC; j += laneSize) {
      if (j + laneSize < maxC && tileOffset.c + j + laneSize < size.c) {
        tmp.copy_from(&i1[j], stdx::vector_aligned);
        caster.simdStorer(out, oOffset + j, tmp);
      } else {
        for (uint32_t e = 0; e < laneSize; e++) {
          if (tileOffset.c + j + e >= size.c) {
            break;
          }
          caster.storer(out, oOffset + j + e, inp[j + e]);
        }
      }
    }
    i1 += origTileSize;
    oOffset += size.c;
  }
}

#define STORETILECASTED(T)                                                     \
  template void storeTileCasted<T>(                                            \
      void *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,      \
      uint16_t origTileSize, const Caster<T> &caster                                 \
  );

UNWIND1_ALL_TYPES(STORETILECASTED)