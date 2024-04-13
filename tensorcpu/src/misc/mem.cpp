#include <stdint.h>
#include <stdlib.h>
#include <memory.h>

#include "tensorcpu.hpp"

void tcFree(void* ptr) {
  free(ptr);
}

void* tcRealloc(void* ptr, uint64_t size) {
  return realloc(ptr, size);
}

void tcMemcpy(void* dst, void* src, uint64_t size) {
  memcpy(dst, src, size);
}