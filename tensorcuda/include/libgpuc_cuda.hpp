#ifndef LIBGPUC_CUDA_HPP
#define LIBGPUC_CUDA_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024U

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
  cudaStream_t* stream;
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

const char *libtcCudaSum2D(
    libtcCudaStream &stream, double *out, double *in, Dim2 inSize
);

const char *libtcCudaAdd2(
    libtcCudaStream &stream, double *out, const double *in1, const double *in2,
    uint64_t n
);
const char* libtcCudaSubtract2(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint64_t n);
const char* libtcCudaMultiply2(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint64_t n);
const char* libtcCudaDivide2(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint64_t n);

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

char const *libtcCudaCaddmm(
    libtcCudaStream &stream, double *out, double *inp1, double *inp2,
    double *add, uint32_t m, uint32_t n, uint32_t k, uint32_t batches
);

const char *libtcCudaMaxPool2D(
    libtcCudaStream &stream, double *out, double *inp, Dim2 kernS, Dim2 outS,
    Dim2 inpS, uint32_t matrices, Dim2 padding, Dim2 stride, Dim2 dilation
);

const char *libtcCudaConv2D(
    libtcCudaStream &stream, double *out, double *inp, double *kernel,
    uint32_t batches, Dim3 outS, Dim3 inpS, Dim2 kernS, uint32_t groups,
    Dim2 padding, PadMode padMode, double pad, Dim2 stride, Dim2 dilation
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LIBGPUC_CUDA_HPP