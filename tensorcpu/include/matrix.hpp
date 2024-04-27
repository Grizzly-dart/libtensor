//
// Created by tejag on 2024-04-26.
//

#ifndef TENSORCPU_MATRIX_HPP
#define TENSORCPU_MATRIX_HPP

#include <cstdint>
#include "tensorcpu.hpp"

template <typename T>
extern void loadTile(
    T *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,
    uint16_t origTileSize
);

template <typename T>
extern void storeTile(
    T *out, const T *inp, Dim2 size, Dim2 tileOffset, Dim2 tileSize,
    uint16_t origTileSize
);

template <typename T>
extern void mm_same_slow(
    T *out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize
);

template <typename T>
void mmBt_same_slow(
    T *out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize
);

#define UNWIND1_ALL_TYPES(MACRO) \
  MACRO(double) \
  MACRO(float) \
  MACRO(uint64_t) \
  MACRO(uint32_t) \
  MACRO(uint16_t) \
  MACRO(uint8_t) \
  MACRO(int64_t) \
  MACRO(int32_t) \
  MACRO(int16_t) \
  MACRO(int8_t)

#endif // TENSORCPU_MATRIX_HPP
