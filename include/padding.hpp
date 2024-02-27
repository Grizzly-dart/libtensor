#ifndef PADDING_HPP
#define PADDING_HPP

#include <cstdint>

#include "libgpuc_cuda.hpp"

template <typename T>
T constant2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y);

template <typename T>
T circular2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y);

template <typename T>
T reflect2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y);

template <typename T>
T replicate2DPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y);

#endif // PADDING_HPP
