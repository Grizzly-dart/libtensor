#ifndef TENSORC_H
#define TENSORC_H

#include <cstdint>

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
} Dim3;

typedef enum PadMode : uint8_t {
  CONSTANT,
  CIRCULAR,
  REFLECT,
  REPLICATION
} PadMode;

extern void libtcFree(void* ptr);
extern void* libtcRealloc(void* ptr, uint64_t size);
extern void libtcMemcpy(void* dst, void* src, uint64_t size);

#ifdef __cplusplus
}
#endif

#endif // TENSORC_H
