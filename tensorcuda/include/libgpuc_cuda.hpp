#ifndef LIBGPUC_CUDA_HPP
#define LIBGPUC_CUDA_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024U

typedef enum : uint8_t {
  i8 = 0,
  i16 = 1,
  i32 = 2,
  i64 = 3,

  u8 = 10,
  u16 = 11,
  u32 = 12,
  u64 = 13,

  f16 = 20,
  bf16 = 21,
  f32 = 22,
  f64 = 23
} dtype;

typedef struct {
  uint64_t totalGlobalMem;
  uint64_t totalConstMem;
  uint64_t sharedMemPerBlock;
  uint64_t reservedSharedMemPerBlock;
  uint64_t sharedMemPerMultiprocessor;
  uint32_t warpSize;
  uint32_t multiProcessorCount;
  uint32_t maxThreadsPerMultiProcessor;
  uint32_t maxThreadsPerBlock;
  uint32_t maxBlocksPerMultiProcessor;
  uint32_t l2CacheSize;
  uint64_t memPitch;
  uint32_t memoryBusWidth;
  uint32_t pciBusID;
  uint32_t pciDeviceID;
  uint32_t pciDomainID;
} libtcDeviceProps;

typedef struct {
  uint64_t free;
  uint64_t total;
} libtcCudaMemInfo;

typedef struct {
  cudaStream_t stream;
  int32_t device;
} libtcCudaStream;

typedef struct {
  cudaStream_t *stream;
  int32_t device;
  uint32_t nStreams;
} libtcCudaStreams;

const char *libtcCudaGetDeviceProps(libtcDeviceProps &ret, int32_t device);
const char *libtcCudaGetMemInfo(libtcCudaMemInfo &memInfo, int32_t device);

const char *libtcCudaCreateStream(libtcCudaStream &ret, int32_t device);
const char *libtcCudaDestroyStream(libtcCudaStream &stream);
const char *libtcCudaSyncStream(
    libtcCudaStream *stream, void (*callback)(const char *)
);

const char *libtcCudaAlloc(libtcCudaStream &stream, void **mem, uint64_t size);
const char *libtcCudaFree(libtcCudaStream &stream, void *ptr);
const char *libtcCudaMemcpy(
    libtcCudaStream &stream, void *dst, void *src, uint64_t size
);

typedef struct {
  uint32_t r;
  uint32_t c;
} Dim2;

typedef struct {
  uint32_t ch;
  uint32_t r;
  uint32_t c;

  __device__ __host__ Dim2 toDim2() { return {r, c}; };
} Dim3;

typedef enum PadMode : uint8_t {
  CONSTANT,
  CIRCULAR,
  REFLECT,
  REPLICATION
} PadMode;

const char *libtcCudaAdd2(
    libtcCudaStream &stream, double *out, const double *inp1, const double *inp2,
    uint64_t n
);
const char *libtcCudaSubtract2(
    libtcCudaStream &stream, double *out, const double *inp1, const double *inp2,
    uint64_t n
);
const char *libtcCudaMultiply2(
    libtcCudaStream &stream, double *out, const double *inp1, const double *inp2,
    uint64_t n
);
const char *libtcCudaDivide2(
    libtcCudaStream &stream, double *out, const double *inp1, const double *inp2,
    uint64_t n
);

const char *libtcCudaSum2d(
    libtcCudaStream &stream, void *out, void *inp, Dim2 inpS, dtype outType,
    dtype inpType
);

const char *libtcCudaMean2d(
    libtcCudaStream &stream, void *out, void *inp, Dim2 inpS,
    dtype outType, dtype inpType
);

const char *libtcCudaVariance2d(
    libtcCudaStream &stream, void *out, void *inp, Dim2 inpS,
    uint64_t correction, uint8_t calcStd, dtype outType, dtype inpType
);

const char *libtcCudaNormalize2d(
    libtcCudaStream &stream, void *out, void *inp, Dim2 inpS,
    double epsilon, dtype outType, dtype inpType
);

const char *libtcCudaTranspose2d(
    libtcCudaStream &stream, double *out, double *inp, Dim3 size
);

char const *libtcCudaMatMul(
    libtcCudaStream &stream, double *out, double *inp1, double *inp2,
    uint32_t m, uint32_t n, uint32_t k, uint32_t batches
);
char const *libtcCudaMatMulT(
    libtcCudaStream &stream, double *out, double *inp1, double *inp2T,
    uint32_t m, uint32_t n, uint32_t k, uint32_t batches
);

char const *libtcCudaMatMulCadd(
    libtcCudaStream &stream, double *out, double *inp1, double *inp2,
    double *add, uint32_t m, uint32_t n, uint32_t k, uint32_t batches
);
char const *libtcCudaMatMulTCadd(
    libtcCudaStream &stream, double *out, double *inp1, double *inp2T,
    double *add, uint32_t m, uint32_t n, uint32_t k, uint32_t batches
);

const char *libtcCudaMaxPool2d(
    libtcCudaStream &stream, double *out, double *inp, Dim2 kernS, Dim2 outS,
    Dim2 inpS, uint32_t matrices, Dim2 padding, Dim2 stride, Dim2 dilation
);

const char *libtcCudaConv2d(
    libtcCudaStream &stream, double *out, double *inp, double *kernel,
    uint32_t batches, Dim3 outS, Dim3 inpS, Dim2 kernS, uint32_t groups,
    Dim2 padding, PadMode padMode, double pad, Dim2 stride, Dim2 dilation
);

const char *libtcCudaPickRows(
    libtcCudaStream &stream, void *out, const void *inp, const void *indices,
    Dim2 size, dtype type, dtype itype
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LIBGPUC_CUDA_HPP