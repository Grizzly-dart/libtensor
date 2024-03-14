#ifndef PADDING_HPP
#define PADDING_HPP

#include <cstdint>

#include "libgpuc_cuda.hpp"

// Function declarations
template <typename T>
__device__ T constant2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r);

template <typename T>
__device__ T circular2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r);

template <typename T>
__device__ T reflect2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r);

template <typename T>
__device__ T replicate2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r);

template <typename T>
__device__ T padder(T* data, Dim2 size, Dim2 padding, PadMode mode, T constant, uint64_t c, uint64_t r);


template <typename T>
__device__ T constant2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r) {
  if (c < padding.c || r < padding.r || c >= (size.c + padding.c) || r >= (size.r + padding.r)) {
    return constant;
  } else {
    return data[(r - padding.r) * size.c + (c - padding.c)];
  }
}

template <typename T>
__device__ T circular2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r) {
  if (c < padding.c) {
    c += size.c;
  } else if (c >= (size.c + padding.c)) {
    c -= size.c;
  }
  if (r < padding.r) {
    r += size.r;
  } else if (r >= (size.r + padding.r)) {
    r -= size.r;
  }
  return data[(r - padding.r) * size.c + (c - padding.c)];
}

template <typename T>
__device__ T reflect2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r) {
  if (c < padding.c) {
    c = padding.c - c;
  } else if (c >= (size.c + padding.c)) {
    c = (size.c + padding.c) - (c - size.c - padding.c) - 1;
  }
  if (r < padding.r) {
    r = padding.r - r;
  } else if (r >= (size.r + padding.r)) {
    r = (size.r + padding.r) - (r - size.r - padding.r) - 1;
  }
  return data[(r - padding.r) * size.c + (c - padding.c)];
}

template <typename T>
__device__ T replicate2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t c, uint64_t r) {
  if (c < padding.c) {
    c = padding.c;
  } else if (c >= (size.c + padding.c)) {
    c = size.c + padding.c - 1;
  }
  if (r < padding.r) {
    r = padding.r;
  } else if (r >= (size.r + padding.r)) {
    r = size.r + padding.r - 1;
  }
  return data[(r - padding.r) * size.c + (c - padding.c)];
}

template <typename T>
__device__ T padder(T* data, Dim2 size, Dim2 padding, PadMode mode, T constant, uint64_t c, uint64_t r) { 
  if (mode == CONSTANT) {
    return constant2DPadding<T>(data, size, padding, constant, c, r);
  } else if (mode == CIRCULAR) {
    return circular2DPadding<T>(data, size, padding, constant, c, r);
  } else if (mode == REFLECT) {
    return reflect2DPadding<T>(data, size, padding, constant, c, r);
  } else if (mode == REPLICATION) {
    return replicate2DPadding<T>(data, size, padding, constant, c, r);
  } else {
    return constant2DPadding<T>(data, size, padding, constant, c, r);
  }
}

#endif // PADDING_HPP
