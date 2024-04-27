// Created by Teja Gudapati on 2024-04-16.

#ifndef TENSORCPU_NATIVE_HPP
#define TENSORCPU_NATIVE_HPP

#include <cstddef>
#include <type_traits>
#include "typed_array.hpp"

#if __GNUC__
#if defined(__x86_64__) || defined(__i386__)
#define TC_ARCH_X86
#elif defined(__aarch64__)
#define TC_ARCH_ARM
#endif
#endif

template <typename T>
#if defined(__GNUC__)
#if defined(__clang__)
[[clang::always_inline]]
#else
[[gnu::always_inline]]
#endif
#endif
inline void atomicAdd(T *ptr, T val);

extern size_t cacheLineSize();

extern size_t cacheSizeL1d();

extern size_t cacheSizeL2d();

extern size_t cacheSizeL3d();

#endif // TENSORCPU_NATIVE_HPP
