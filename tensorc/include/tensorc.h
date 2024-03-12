#ifndef TENSORC_H
#define TENSORC_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t r;
  uint32_t c;
} Size2;

extern void libtcFree(void* ptr);
extern void* libtcRealloc(void* ptr, uint64_t size);
extern void libtcMemcpy(void* dst, void* src, uint64_t size);

#ifdef __cplusplus
}
#endif

#endif // TENSORC_H
