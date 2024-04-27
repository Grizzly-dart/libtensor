//
// Created by tejag on 2024-04-27.
//

#include <cstdint>

#include "tensorcpu.hpp"
#include "matrix.hpp"

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