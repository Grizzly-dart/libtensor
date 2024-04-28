#include <stddef.h>
#include <cstring>

#include "native.hpp"

#if defined(linux)
#include <stdio.h>

size_t cacheLineSize() {
  FILE *p = nullptr;
  for (int i = 0; i < 10; i++) {
    int err = 0;
    char path[100];
    // Check if it's a L1 cache
    {
      snprintf(
          path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/level", i
      );
      p = fopen(path, "r");
      if (p == nullptr)
        return 0;
      int level = 0;
      err = fscanf(p, "%d", &level);
      fclose(p);

      if (level != 1)
        continue;
    }

    // Check if it's a data cache
    {
      snprintf(path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
      p = fopen(path, "r");
      if (p == NULL)
        continue;
      char type[100];
      err = fscanf(p, "%s", type);
      fclose(p);

      if (strcmp(type, "Data") != 0)
        continue;
    }

    snprintf(
        path, 100,
        "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size", i
    );
    p = fopen(path, "r");
    unsigned int size = 0;
    if (p) {
      err = fscanf(p, "%d", &size);
      fclose(p);
    }
    return size;
  }
  return 0;
}
#pragma clang diagnostic pop

size_t cacheSizeL1d() {
  FILE *p = 0;
  for (int i = 0; i < 10; i++) {
    int err = 0;
    char path[100];
    // Check if it's a L1 cache
    {
      snprintf(
          path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/level", i
      );
      p = fopen(path, "r");
      if (p == nullptr)
        return 0;
      int level = 0;
      err = fscanf(p, "%d", &level);
      fclose(p);

      if (level != 1)
        continue;
    }

    // Check if it's a data cache
    {
      snprintf(path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
      p = fopen(path, "r");
      if (p == NULL)
        continue;
      char type[100];
      err = fscanf(p, "%s", type);
      fclose(p);

      if (strcmp(type, "Data") != 0)
        continue;
    }

    snprintf(
        path, 100,
        "/sys/devices/system/cpu/cpu0/cache/index%d/size", i
    );
    p = fopen(path, "r");
    unsigned int size = 0;
    if (p) {
      err = fscanf(p, "%d", &size);
      fclose(p);
    }
    return size;
  }
  return 0;
}

size_t cacheSizeL2d() {
  FILE *p = nullptr;
  for (int i = 0; i < 10; i++) {
    int err = 0;
    char path[100];
    // Check if it's a L1 cache
    {
      snprintf(
          path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/level", i
      );
      p = fopen(path, "r");
      if (p == NULL)
        return 0;
      int level = 0;
      err = fscanf(p, "%d", &level);
      fclose(p);

      if (level != 2)
        continue;
    }

    // Check if it's a data cache
    {
      snprintf(path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
      p = fopen(path, "r");
      if (p == NULL)
        continue;
      char type[100];
      err = fscanf(p, "%s", type);
      fclose(p);

      if (strcmp(type, "Data") != 0)
        continue;
    }

    snprintf(
        path, 100,
        "/sys/devices/system/cpu/cpu0/cache/index%d/size", i
    );
    p = fopen(path, "r");
    unsigned int size = 0;
    if (p) {
      err = fscanf(p, "%d", &size);
      fclose(p);
    }
    return size;
  }
  return 0;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
size_t cacheSizeL3d() {
  FILE *p = nullptr;
  for (int i = 0; i < 10; i++) {
    int err = 0;
    char path[100];
    // Check if it's a L1 cache
    {
      snprintf(
          path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/level", i
      );
      p = fopen(path, "r");
      if (p == NULL)
        return 0;
      int level = 0;
      err = fscanf(p, "%d", &level);
      fclose(p);

      if (level != 3)
        continue;
    }

    // Check if it's a data cache
    {
      snprintf(path, 100, "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
      p = fopen(path, "r");
      if (p == NULL)
        continue;
      char type[100];
      err = fscanf(p, "%s", type);
      fclose(p);

      if (strcmp(type, "Data") != 0)
        continue;
    }

    snprintf(
        path, 100,
        "/sys/devices/system/cpu/cpu0/cache/index%d/size", i
    );
    p = fopen(path, "r");
    unsigned int size = 0;
    if (p) {
      err = fscanf(p, "%d", &size);
      fclose(p);
    }
    return size;
  }
  return 0;
}
#pragma clang diagnostic pop

#elif defined(__APPLE__)

#include <sys/sysctl.h>

size_t cacheLineSize() {
  size_t line_size = 0;
  size_t sizeof_line_size = sizeof(line_size);
  sysctlbyname("hw.cachelinesize", &line_size, &sizeof_line_size, 0, 0);
  return line_size;
}

size_t cacheSizeL1d() {
  size_t line_size = 0;
  size_t sizeof_line_size = sizeof(line_size);
  sysctlbyname("hw.l1dcachesize", &line_size, &sizeof_line_size, 0, 0);
  return line_size;
}

size_t cacheSizeL2d() {
  size_t line_size = 0;
  size_t sizeof_line_size = sizeof(line_size);
  sysctlbyname("hw.l2cachesize", &line_size, &sizeof_line_size, 0, 0);
  return line_size;
}

size_t cacheSizeL3d() {
  size_t line_size = 0;
  size_t sizeof_line_size = sizeof(line_size);
  sysctlbyname("hw.l3cachesize", &line_size, &sizeof_line_size, 0, 0);
  return line_size;
}
#else
#error Unrecognized platform
#endif
