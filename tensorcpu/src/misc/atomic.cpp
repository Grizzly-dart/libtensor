#include <type_traits>

#include "native.hpp"
#include "typed_array.hpp"

template <typename T>
[[gnu::always_inline]]
[[clang::always_inline]]
inline void atomicAdd(T *ptr, T val) {
#if defined(TC_ARCH_X86)
  if constexpr (std::is_same<T, float>::value) {
    volatile float a = 0;
    asm volatile("tjloop: MOVSS (%[ptr]), %[a]\n"
                 "MOVQ %[a], %%rax\n"
                 "ADDSS %[val], %[a]\n"
                 "MOVQ %[a], %%rdx\n"
                 "lock cmpxchg %%edx, (%[ptr])\n"
                 "jnz tjloop\n"
                 : [ptr] "+r"(ptr), [a] "=&x"(a)
                 : [val] "x"(val)
                 : "rax", "rdx", "memory", "cc");
  } else if constexpr (isAnyInt<T>()) {
    asm volatile("lock xadd %0, %1" : "+m"(*ptr) : "x"(val) : "memory");
  } else {
    static_assert(false, "Unsupported type");
  }
  // TODO implement double
#elif defined(TC_ARCH_ARM)
  if constexpr (std::is_same<T, float>::value) {
    volatile float a = 0;
    asm volatile(
        "1: LDXR w0, [%[ptr]]; tjatomic\n"
        "FMOV %s[a], w0\n"
        "FADD %s[a], %s[a], %s[val]\n"
        "FMOV w0, %s[a]\n"
        "STXR w1, w0, [%[ptr]]\n"
        "CBNZ w1, 1b\n"
        : [ptr] "+r"(ptr), [a] "=&x"(a)
        : [val] "x"(val)
        : "w0", "w1", "memory", "cc"
    );
  } else if constexpr (isAnyInt<T>()) {
    // TODO
    static_assert(false, "Unimplemented type");
  } else {
    static_assert(false, "Unsupported type");
  }
  // TODO implement double
#else
  static_assert(false, "Unsupported architecture");
#endif
}

template void atomicAdd<float>(float *ptr, float val);