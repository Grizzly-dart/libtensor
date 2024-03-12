#include <stdint.h>
#include <stdlib.h>
#include <memory.h>

void libtcFree(void* ptr) {
  free(ptr);
}

void* libtcRealloc(void* ptr, uint64_t size) {
  return realloc(ptr, size);
}

void libtcMemcpy(void* dst, void* src, uint64_t size) {
  memcpy(dst, src, size);
}