#ifndef TENSORC_H
#define TENSORC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum : uint8_t {
  i8 = 1,
  i16 = 2,
  i32 = 4,
  i64 = 8,

  u8 = 17,
  u16 = 18,
  u32 = 20,
  u64 = 24,

  f16 = 34,
  f32 = 36,
  f64 = 40,
  bf16 = 66,
} dtype;

typedef struct {
  uint32_t r;
  uint32_t c;
} Dim2;

typedef struct {
  uint32_t ch;
  uint32_t r;
  uint32_t c;

  /*__device__ __host__*/ Dim2 toDim2() { return {r, c}; };

  [[nodiscard]] uint64_t nel() const { return ch * r * c; }
} Dim3;

typedef enum PadMode : uint8_t {
  CONSTANT,
  CIRCULAR,
  REFLECT,
  REPLICATION
} PadMode;

extern void tcFree(void* ptr);
extern void* tcRealloc(void* ptr, uint64_t size);
extern void tcMemcpy(void* dst, void* src, uint64_t size);

#ifdef __cplusplus
}
#endif

template<typename O, typename I1, typename I2>
const char* tcPlus(O* out, const I1* inp1, const I2* inp2, const I2* scalar, uint64_t nel, uint8_t flip);

#endif // TENSORC_H
