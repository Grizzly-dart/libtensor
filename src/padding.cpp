#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "libgpuc_cuda.hpp"

template <typename T>
T constant2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x || y < padding.y || x >= (size.x + padding.x) || y >= (size.y + padding.y)) {
    return constant;
  } else {
    return data[(y - padding.y) * size.x + (x - padding.x)];
  }
}

template <typename T>
T circular2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x) {
    x += size.x;
  } else if (x >= (size.x + padding.x)) {
    x -= size.x;
  }
  if (y < padding.y) {
    y += size.y;
  } else if (y >= (size.y + padding.y)) {
    y -= size.y;
  }
  return data[(y - padding.y) * size.x + (x - padding.x)];
}

template <typename T>
T reflect2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x) {
    x = padding.x - x;
  } else if (x >= (size.x + padding.x)) {
    x = (size.x + padding.x) - (x - size.x - padding.x) - 1;
  }
  if (y < padding.y) {
    y = padding.y - y;
  } else if (y >= (size.y + padding.y)) {
    y = (size.y + padding.y) - (y - size.y - padding.y) - 1;
  }
  return data[(y - padding.y) * size.x + (x - padding.x)];
}

template <typename T>
T replicate2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x) {
    x = padding.x;
  } else if (x >= (size.x + padding.x)) {
    x = size.x + padding.x - 1;
  }
  if (y < padding.y) {
    y = padding.y;
  } else if (y >= (size.y + padding.y)) {
    y = size.y + padding.y - 1;
  }
  return data[(y - padding.y) * size.x + (x - padding.x)];
}