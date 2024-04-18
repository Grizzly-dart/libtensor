//
// Created by Teja Gudapati on 2024-04-16.
//

#ifndef TENSORCPU_NATIVE_HPP
#define TENSORCPU_NATIVE_HPP

#if __GNUC__
#if defined(__x86_64__) || defined(__i386__)
#define TC_ARCH_X86
#elif defined(__aarch64__)
#define TC_ARCH_ARM
#endif
#endif

template <typename T>
void atomicAdd(T *ptr, T val);

#endif // TENSORCPU_NATIVE_HPP
