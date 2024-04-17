#if __GNUC__
#if defined(__x86_64__) || defined(__i386__)
#define ARCH_X86
#elif defined(__aarch64__)
#define ARCH_ARM
#endif
#endif

#include "native.hpp"

#if defined(ARCH_X86)

/*
template <typename T>
[[gnu::always_inline]]
__attribute__((always_inline))
inline void atomicAdd(T *ptr, T val) {
  asm volatile("lock xadd %0, %1" : "+r"(val), "+m"(*ptr) : : "memory");
}
 */

#elif defined(ARCH_ARM)

template <typename T>
[[gnu::always_inline]]
inline void atomicAdd(T *ptr, T val) {
  asm volatile( //"STR %q1, [%0] ; hello\n"
      "1: LDXR w0, [%0]\n"
      "FADD s0, s0, %s1\n"
      "STXR w1, w0, [%0]\n"
      "CBNZ w1, 1b\n"
      :
      : "r"(ptr), "w"(v)
      : "d0", "w0", "s0", "w1", "memory", "cc"
  );
}

#endif
// ATOMICADD(double)

/*
#ifdef __amd64__
                    asm volatile("lock addss %1, %0"
                                 : "+m"(out[m * inp2S.c + n])
                                 : "x"(v)
                                 : "memory");
#endif
#ifdef __arm__
                    asm volatile("ldrex r0, [%0]\n"
                                 "add r0, r0, %1\n"
                                 "strex r1, r0, [%0]\n"
                                 "cmp r1, #0\n"
                                 "bne 1b"
                                 : "+r"(out[m * inp2S.c + n])
                                 : "r"(v)
                                 : "r0", "r1", "memory");
#endif
 */

/*
asm volatile("ldrex r0, [%0]\n"
             "add r0, r0, %1\n"
             "strex r1, r0, [%0]\n"
             "cmp r1, #0\n"
             "bne 1b"
             : "+r"(out[m * inp2S.c + n])
             : "r"(v)
             : "r0", "r1", "memory");
 */